import torch
from peft import PeftModel
import transformers
import gradio as gr
import argparse
import csv
import os
import random
from tqdm import tqdm
import json
import numpy as np    
from datetime import datetime       

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="./llama-7b")
# parser.add_argument('--train_weight', type=str, default="./trained/$param/$percentage")
parser.add_argument("--project", type=str, default="Mac")
parser.add_argument("--batch", type=int, default=10)
parser.add_argument("--percentage", type=str, default="cross")
parser.add_argument("--lora", type=str, default="True")
args = parser.parse_args()

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig,LlamaTokenizerFast

# tokenizer = LlamaTokenizerFast.from_pretrained(args.base_model)
tokenizer = LlamaTokenizerFast.from_pretrained(args.base_model, unk_token="<unk>")

project = args.project
batch_size=args.batch
percentage = args.percentage
BASE_MODEL = args.base_model
if args.base_model=="True":
    LORA_WEIGHTS = "./trained/" + project + "/" + percentage+"_hf_fine"
else:
    LORA_WEIGHTS=False
# if project=="BGL" or project=="Thunderbird":
max_length=64
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
        if LORA_WEIGHTS:
            model = PeftModel.from_pretrained(
                model, LORA_WEIGHTS, torch_dtype=torch.float16, force_download=True
            )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if LORA_WEIGHTS:
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
    
    def read_json_file(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data


    def generate_prompt(project, instruction, input=None,output=None):
        few_shots = "../logs/" + project + "/0.001/train.json"
        few_shots_list = read_json_file(few_shots)
        examples=""
        for i in range(len(few_shots_list)):
            examples = examples + f"""
            ### Example {str(i+1)}:
            ### Instruction:
            {instruction}
            ### Input:
            {few_shots_list[i]["input"]}
            ### Response:
            {few_shots_list[i]["output"]}

            """
        if input:
            begin = f"""Below is an instruction that describes a task, paired with an input that provides further context.

            """
            if not output:
                tail = f"""
            Now, use the pattern from the examples to complete the request below:

            ### Instruction:
            {instruction}
            ### Input:
            {input}
            ### Response:"""
            else:
                tail = f"""
            Now, use the pattern from the examples to complete the request below:

            ### Instruction:
            {instruction}
            ### Input:
            {input}
            ### Response:
            {output}"""
            return begin + examples + tail
        
    def generate_prompt_list(project,content):
        res=[]
        batch_list=[]
        prompt_list=[]
        back_list=[]
        for i in content:
            prompt = generate_prompt(project,"Parse the input log to log template.", i)
            res.append(prompt)
        #     if len(back_list)<batch_size:
        #         back_list.append(prompt)
        #     else:
        #         # back_list.append(prompt)
        #         prompt_list.append(back_list)
        #         back_list=[]
        #         back_list.append(prompt)
        # if back_list!=[]:
        #     prompt_list.append(back_list)
        # # print(prompt_list)
        # # print('prompt generated',len(prompt_list))
        # # logging.info('prompt generated')
        # for i in prompt_list:
        #     # print(i)
        #     inputs = tokenizer(i, return_tensors="pt", padding=True).to(device)
        #     input_ids = inputs["input_ids"].to(device)

        return res
    
    # def generate_prompt_list(project,content):
    #     res=[]
    #     batch_list=[]
    #     prompt_list=[]
    #     back_list=[]
    #     for i in content:
    #         prompt = generate_prompt(project,"Parse the input log to log template.", i)
    #         res.append(prompt)
    #     return res
        



    if device != "cpu":
        model.half()
    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


    def evaluate(
        # instruction,
        # input=None,
        content_list,
        
        temperature=0,
        top_p=0.9,
        top_k=30,
        num_beams=2,
        max_new_tokens=max_length,
        **kwargs,
    ):
        # prompt = generate_prompt(project,instruction, input)
        # time1 = datetime.now()
        # for i in content_list:
        #     print(i)
        inputs = tokenizer(content_list, return_tensors="pt", padding=True).to(device)
        # print("Prompt",content_list)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        # time2 = datetime.now()
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        # time3 = datetime.now()
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        # for i in outputs:
        #     print(i)
        res=[]
        for i in range(len(outputs)):     
            output = outputs[i].replace(content_list[i], "").replace("<unk>", "").replace("</s>", "").replace("<s>", "").strip()
            # print(output)
            # print(type(output))
            output = keep_first_line(output).strip()
            res.append(output)
        # time4 = datetime.now()
        
        # print("Res: ",res)
        # print("duration:",(time2-time1).microseconds,(time3-time2).microseconds,(time4-time3).microseconds)
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
    def sample_list_pair(list1,list2,n):
        indices = list(range(len(list1)))

        # 随机抽样
        sample_indices = random.sample(indices, n)

        # 根据抽样的索引获取对应的元素
        sample_list1 = [list1[i] for i in sample_indices]
        sample_list2 = [list2[i] for i in sample_indices]
        return sample_list1,sample_list2

    if not is_validation:
        log_file = "../logs/" + project + "/" + project + "_2k.log_structured_corrected.csv"
    else:
        log_file = "../logs/" + project + "/0.2/validation.csv"
    # log_file = "./logs/" + project + "/" + project + "_2k.log_structured_corrected.csv"
    
    content, event_template = read_csv(log_file)
    # content, event_template=sample_list_pair(content,event_template,10)
    instruction = "Parse the input log to log template."
    predictions = []
    des = "Predict " + project + " " + percentage
    inputs_ids_list=generate_prompt_list(project,content=content)
    # inputs_ids_list=generate_prompt_list(project,content=content)
    for i in tqdm(range(0,len(inputs_ids_list),batch_size), desc=des):
        with torch.autocast("cuda"):
            prediction= evaluate(inputs_ids_list[i:i+batch_size])
        # print("prediction",prediction)
        # prediction = evaluate(instruction=instruction, input=content[i])
        # prediction = keep_first_line(prediction)
        predictions.extend(prediction)
        # predictions.append(prediction)
    if not is_validation:
        output_dir = "./diff/" + project + "/" + percentage + "_hf_fine/"
    else:
        output_dir = "./diff/" + project + "/" + percentage + "/validation_"+check_point+"/"
    if not LORA_WEIGHTS:
        output_dir = "./diff/" + project + "/few_shot_learning/"
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