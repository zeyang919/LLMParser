from transformers import AutoTokenizer,AutoModel
# from thuglm.modeling_chatglm import ChatGLMForConditionalGeneration
import torch
import argparse
import csv
import os
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType,PeftModel


parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="./chatglm6b-dddd")
# parser.add_argument('--train_weight', type=str, default="./trained/$param/$percentage")
parser.add_argument("--project", type=str, default="Mac")
parser.add_argument("--percentage", type=str, default="0.025")
args = parser.parse_args()


project = args.project
percentage = args.percentage
BASE_MODEL = args.base_model
LORA_WEIGHTS = "./trained/" + project + "/" + percentage+"/"

max_length=1024



def eval(check_point,is_validation=False):
    model = AutoModel.from_pretrained(
    BASE_MODEL, trust_remote_code=True).half().cuda()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['query_key_value',],
    )

    model = get_peft_model(model, peft_config)
    # if is_validation:
    peft_path = LORA_WEIGHTS+"checkpoint-"+check_point+"/chatglm-lora.pt"
        
    model.load_state_dict(torch.load(peft_path), strict=False)
    model.eval()


    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)



    def generate_prompt(instruction, input=None):
        if input:
            return f"""Instruction:
    {instruction}
    Input:
    {input}
    Response:"""
        else:
            return f"""Instruction:
    {instruction}
    Response:"""

    def evaluate(instruction,input=None):
        prompt = generate_prompt(instruction, input)
        with torch.autocast("cuda"):
            res, history = model.chat(tokenizer=tokenizer, query=prompt,max_length=max_length,temperature=0.1)
            return res
        

    def read_csv(filename):
        content_list = []
        event_template_list = []

        with open(filename, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                content_list.append(row["Content"])
                event_template_list.append(row["EventTemplate"])

        return content_list, event_template_list

    def accuracy(string1, string2):
        string1 = string1.replace(" ", "")
        string2 = string2.replace(" ", "")

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
    content, event_template = read_csv(log_file)
    instruction = "Parse the input log to log template."
    predictions = []
    des = "Predict " + project + " " + percentage
    for i in tqdm(range(len(content)), desc=des):
        prediction = evaluate(instruction=instruction, input=content[i])
        # prediction = keep_first_line(prediction)
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
    print(project,check_point,str(is_validation),score)

    with open(output_dir + "result.txt", "w") as f:
        f.write(str(score))
    return score


check_points=["50","100","150","200","250","300"
              ]

for check_point in check_points:
    validation_scores=[]
    validation_scores.append(eval(check_point=check_point,is_validation=True))

max_index = max(range(len(validation_scores)), key=validation_scores.__getitem__)

eval(check_points[max_index],is_validation=False)


