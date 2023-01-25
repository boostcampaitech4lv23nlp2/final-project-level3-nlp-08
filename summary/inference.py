import pandas as pd
from rouge import Rouge
import torch
from transformers.models.bart import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tqdm import tqdm
from utlis.rdass import sim
from models import BaseModel


def load_model(name, device):
    if name == "ref":
        model = BartForConditionalGeneration.from_pretrained("alaggung/bart-rl")
        tokenizer = AutoTokenizer.from_pretrained("alaggung/bart-rl")
    elif name == "noR3F":
        model = BaseModel("gogamza/kobart-summarization")
        model.load_state_dict(
            torch.load("saved/dialog_dual_noR3F/checkpoint-100000/pytorch_model.bin")
        )
        model = model.plm
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "papari1123/summary_bart_dual_R3F_aihub"
        )
    elif name == "single":
        model = BartForConditionalGeneration.from_pretrained(
            "papari1123/summary_bart_single_aihub"
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "papari1123/summary_bart_dual_R3F_aihub"
        )
    else:
        model = BaseModel("gogamza/kobart-summarization")
        model.load_state_dict(
            torch.load(
                "saved/dialog_R3F_only_dual_new/checkpoint-100000/pytorch_model.bin"
            )
        )
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
        if model_name == "ref":
            c = (
                c.replace("</s> <s>", "[SEP]")
                .replace("<s>", "[BOS]")
                .replace("</s>", "[EOS]")
            )

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
        if model_name == "ref":
            output = output.split(".")[0]
        print(model_name + " out", output)
        print("sum", s)
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

    print(
        f"ROUGE-1: {rouge_1} | ROUGE-2: {rouge_2} | ROUGE-L: {rouge_L} | ROUGE-L-p: {rouge_L_p} | ROUGE-L-r: {rouge_L_r} | SIM: {sim_score}"
    )

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


if __name__ == "__main__":
    # single dataset
    test_data = pd.read_csv("datas/Test/all_05.csv")
    test_samples = 1500
    test("noR3F", test_data, test_samples, "single")
    test("single", test_data, test_samples, "single")
    test("dual", test_data, test_samples, "single")
    # dual dataset
    test_data = pd.read_csv("datas/Test/all_dual_05.csv")
    test("noR3F", test_data, test_samples, "dual")
    test("single", test_data, test_samples, "dual")
    test("dual", test_data, test_samples, "dual")
    # dual shuffle dataset
    test_data = pd.read_csv("datas/Test/all_dual_shuffle_05.csv")
    test("noR3F", test_data, test_samples, "dual_shuffle")
    test("single", test_data, test_samples, "dual_shuffle")
    test("dual", test_data, test_samples, "dual_shuffle")
