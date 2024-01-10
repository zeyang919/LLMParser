import torch
from peft import PeftModel
import transformers
import gradio as gr
import argparse
import csv
import os
from tqdm import tqdm
import numpy as np           

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="./llama-7b-hf")
# parser.add_argument('--train_weight', type=str, default="./trained/$param/$percentage")
parser.add_argument("--project", type=str, default="Mac")
parser.add_argument("--percentage", type=str, default="0.05")
args = parser.parse_args()

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

project = args.project
percentage = args.percentage
BASE_MODEL = args.base_model
LORA_WEIGHTS = "./trained/" + project + "/" + percentage
# if project=="BGL" or project=="Thunderbird":
max_length=1024
# else:
#     max_length=512
    
def eval(check_point,is_validation=False):
    # LORA_WEIGHTS = "./trained/" + project + "/" + percentage+"/checkpoint-"+check_point+"/pytorch_model.bin"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:
        pass

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model, LORA_WEIGHTS, torch_dtype=torch.float16, force_download=True
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )


    def generate_prompt(instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Response:"""


    if device != "cpu":
        model.half()
    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)


    def evaluate(
        instruction,
        input=None,
        temperature=0,
        top_p=0.9,
        top_k=30,
        num_beams=2,
        max_new_tokens=max_length,
        **kwargs,
    ):
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()


    def read_csv(filename):
        content_list = []
        event_template_list = []

        with open(filename, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                content_list.append(row["Content"])
                event_template_list.append(row["EventTemplate"])

        return content_list, event_template_list


    def keep_first_line(text):
        lines = text.split("\n")
        first_line = lines[0]
        return first_line


    def accuracy(string1, string2):
        # 去除空格
        string1 = string1.replace(" ", "")
        string2 = string2.replace(" ", "")

        # 比较字符串是否相等
        if string1 == string2:
            return True
        else:
            return False


    def get_accruacy_over_list(prediction, groundtruth):
        if type(groundtruth) == list:
            if len(groundtruth) == 0:
                return 0
            return np.max([accuracy(prediction, gt) for gt in groundtruth])
        return accuracy(prediction, groundtruth)


    def acc(predictions, data, metric="ACC"):
        if metric == "ACC":
            accs = []
            for prediction, dp in zip(predictions, data):
                accs.append(get_accruacy_over_list(prediction, dp))
            return np.mean(accs)

    if not is_validation:
        log_file = "../logs/" + project + "/" + project + "_2k.log_structured_corrected.csv"
    else:
        log_file = "../logs/" + project + "/0.2/validation.csv"
    # log_file = "./logs/" + project + "/" + project + "_2k.log_structured_corrected.csv"
    
    content, event_template = read_csv(log_file)
    instruction = "Parse the input log to log template."
    predictions = []
    des = "Predict " + project + " " + percentage
    for i in tqdm(range(len(content)), desc=des):
        prediction = evaluate(instruction=instruction, input=content[i])
        prediction = keep_first_line(prediction)
        predictions.append(prediction)
    if not is_validation:
        output_dir = "./diff/" + project + "/" + percentage + "/"
    else:
        output_dir = "./diff/" + project + "/" + percentage + "/validation_"+check_point+"/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir + "prediction.csv", "w",encoding='utf-8') as f:
        f = csv.writer(f)
        for i in range(len(predictions)):
            f.writerow([content[i], predictions[i], event_template[i]])
    with open(output_dir + "diff.csv", "w",encoding='utf-8') as f:
        f = csv.writer(f)
        for i in range(len(predictions)):
            if not accuracy(predictions[i], event_template[i]):
                f.writerow([content[i], predictions[i], event_template[i]])

    score = acc(predictions, event_template)
    with open(output_dir + "result.txt", "w") as f:
        f.write(str(score))
    return score

# check_points=[str(int(4000*float(percentage))),str(int(8000*float(percentage))),str(int(12000*float(percentage)))]

# for check_point in check_points:
#     validation_scores=[]
#     validation_scores.append(eval(check_point=check_point,is_validation=True))

# max_index = max(range(len(validation_scores)), key=validation_scores.__getitem__)

eval("300",is_validation=False)