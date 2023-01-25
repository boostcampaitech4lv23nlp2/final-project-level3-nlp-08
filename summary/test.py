import pandas as pd
from rouge import Rouge
import torch
from transformers.models.bart import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tqdm import tqdm
from utlis.rdass import sim
from models import BaseModel

from omegaconf import OmegaConf
import argparse


def load_model(path, device):
    if not ".bin" in path:
        model = BartForConditionalGeneration.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        model = BaseModel("papari1123/summary_bart_dual_R3F_aihub")
        model.load_state_dict(torch.load(path))
        model = model.plm
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "papari1123/summary_bart_dual_R3F_aihub"
        )
    model.to(device)
    model.eval()
    return model, tokenizer


def test(model_name, dataset, test_num, name_header=""):
    rouge = Rouge()
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    model, tokenizer = load_model(model_name, device)

    rouge_1, rouge_2, rouge_L, rouge_L_r, rouge_L_p, sim_score = 0, 0, 0, 0, 0, 0

    for s, c in tqdm(list(zip(dataset["summary"], dataset["context"]))[:test_num]):
        input_ids = tokenizer.encode(c)
        input_ids = torch.tensor(input_ids).to(device)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(
            input_ids,
            eos_token_id=1,
            repetition_penalty=2.0,
            max_length=512,
            num_beams=15,
            use_cache=True,
        )
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(model_name + " out\n", output)
        print("GT\n", s)
        score = rouge.get_scores(output, s, avg=True)

        rouge_1 += score["rouge-1"]["f"]
        rouge_2 += score["rouge-2"]["f"]
        rouge_L += score["rouge-l"]["f"]
        rouge_L_r += score["rouge-l"]["r"]
        rouge_L_p += score["rouge-l"]["p"]
        sim_score += sim(output, s, c)

    rouge_1 /= test_num
    rouge_2 /= test_num
    rouge_L /= test_num
    rouge_L_r /= test_num
    rouge_L_p /= test_num
    sim_score /= test_num

    f = open(f"{name_header}_{model_name}_score_log.txt", "w")
    data = (
        "ROUGE-1: "
        + str(rouge_1)
        + " | ROUGE-2: "
        + str(rouge_2)
        + " | ROUGE-L: "
        + str(rouge_L)
        + " | ROUGE-L-p: "
        + str(rouge_L_p)
        + " | ROUGE-L-r: "
        + str(rouge_L_r)
        + " | SIM: "
        + str(sim_score)
    )
    f.write(data)
    f.close()

    print(
        f"ROUGE-1: {rouge_1} | ROUGE-2: {rouge_2} | ROUGE-L: {rouge_L} | "
        +"ROUGE-L-p: {rouge_L_p} | ROUGE-L-r: {rouge_L_r} | SIM: {sim_score}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--data", type=str)
    parser.add_argument("--model", type=str)
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f"./config/{args.config}.yaml")
    model_path = args.model if args.model else cfg.test.model_path
    data_path = args.data if args.data else cfg.path.predict_path

    dataset = pd.read_csv(data_path)
    test(model_path, dataset, len(dataset))
