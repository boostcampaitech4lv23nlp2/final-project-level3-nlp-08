from torch import nn, distributions
from transformers import Trainer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import wandb


class BaseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


class BlendTrainer(Trainer):
    def __init__(self, kl_div_lambda=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_div_lambda = kl_div_lambda
        self.noise_sampler = distributions.normal.Normal(loc=0.0, scale=1e-5)
        self.batch = 0
        self.cul_origin_loss = 0
        self.cul_dual_loss = 0
        self.cul_origin_irr_loss = 0
        self.cul_dual_shuffle_loss = 0
        print('trainer kl_div_lambda is ', kl_div_lambda)

    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                reduction="sum",
            )
        ) / noised_logits.size(0)

    def compute_loss(self, model, inputs, return_outputs=False):
        def get_outputs(inputs, noise_inputs):
            # inputs
            outputs = model(inputs)
            # noise inputs
            noise_embeds = self.embedding_layer(noise_inputs["input_ids"])
            noise = self.noise_sampler.sample(sample_shape=noise_embeds.shape).to(
                noise_embeds
            )
            noise_inputs["inputs_embeds"] = noise_embeds.detach().clone() + noise
            noise_outputs = model(noise_inputs)
            symm_kl_loss = self._get_symm_kl(outputs["logits"], noise_outputs["logits"])
            return outputs, symm_kl_loss

        # eval
        if not "noise_input_ids" in inputs:
            outputs = model(inputs)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
        # Blend 사용
        else:
            self.embedding_layer = model.plm.get_input_embeddings()
            dual_inputs = {
                "input_ids": inputs["input_ids"],
                "decoder_input_ids": inputs["decoder_input_ids"],
                "labels": inputs["labels"],
            }
            dual_noise_inputs = {
                "input_ids": inputs["noise_input_ids"],
                "decoder_input_ids": inputs["noise_decoder_input_ids"],
                "labels": inputs["noise_labels"],
            }

            dual_outputs, dual_kl_symm_loss = get_outputs(
                dual_inputs, dual_noise_inputs
            )

            # TEST 용
            self.batch += 1
            self.cul_dual_loss += dual_outputs["loss"]
            self.cul_dual_shuffle_loss += dual_kl_symm_loss

            if self.batch % 25 == 0:
                logs = {
                    "dual_loss": self.cul_dual_loss / 25,
                    "dual_symm_kl_loss": self.cul_dual_shuffle_loss / 25,
                    "step": self.batch,
                }
                self.cul_dual_loss = 0
                self.cul_dual_shuffle_loss = 0
                wandb.log(logs)
            loss = dual_outputs["loss"] + dual_kl_symm_loss * self.kl_div_lambda

            return (loss, dual_outputs) if return_outputs else loss
