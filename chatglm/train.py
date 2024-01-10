from transformers import Trainer, TrainingArguments
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional
import torch
import argparse
from glob import glob
import os
import pandas as pd
import shutil
from itertools import chain
from tqdm import tqdm
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--train_percentage", type=str, default=0.025)
parser.add_argument("--model", type=str, default="../LLMs/chatglm6b-dddd")
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--project", type=str, default="Mac")
args = parser.parse_args()


project = args.project
train_percentage = args.train_percentage
seed = 41
random.seed(seed)  
batch_size = args.batch_size
lr = args.learning_rate
num_epochs = args.num_epochs
output_dir = "../fine_tuned_model/chatglm/{}/{}/".format(project, train_percentage)
pretrainedmodel_path = args.model
data_path = "../logs/{}/{}/train.json".format(project, train_percentage)

tokenizer = AutoTokenizer.from_pretrained(pretrainedmodel_path
                                          , trust_remote_code=True
                                          )

model = (
    AutoModel.from_pretrained(pretrainedmodel_path
                              , trust_remote_code=True
                              )
    .half()
    .cuda()
)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    # ['dense','dense_h_to_4h','dense_4h_to_h'] # 'query_key_value',
    target_modules=[
        "query_key_value",
    ],
)
model = get_peft_model(model, peft_config)


class MyTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        def save_tunable_parameters(model, path):
            saved_params = {
                k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
            }
            # saved_params = model.state_dict()
            torch.save(saved_params, path)

        save_tunable_parameters(self.model, os.path.join(output_dir, "chatglm-lora.pt"))


dataset = load_dataset("json", data_files=data_path)


def get_masks_and_position_ids(
    seq, seq_len, context_length, device, gmask=False, position_encoding_2d=True
):
    mask_position = (
        seq_len - 2
    )  # is equal to `seq.index(mask_token)` or `seq.index(150001)`
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(150004)`
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat(
            (
                torch.zeros(seq_length, dtype=torch.long, device=device),
                torch.arange(
                    context_length - seq_length, dtype=torch.long, device=device
                )
                + 1,
            )
        )
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1 :] = mask_position
    return attention_mask, position_ids


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids) + 1
    input_ids = []
    attention_mask_list = []
    position_ids_list = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        labels = (
            [-100] * (seq_len - 1)
            + ids[(seq_len - 1) :]
            + [tokenizer.eop_token_id]
            + [-100] * (longest - ids_l - 1)
        )
        ids = ids + [tokenizer.eop_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        attention_mask, position_ids = get_masks_and_position_ids(
            ids, seq_len, longest, _ids.device, gmask=False
        )
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(attention_mask_list)
    position_ids = torch.stack(position_ids_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def format_example(example: dict) -> dict:
    context = f"""Instruction: 
    {example['instruction']}\n"""
    if example.get("input"):
        context += f"""Input: 
        {example['input']}\n"""
    context += f"Response: \n"
    target = example["output"]
    # {"context": context, "target": target}
    example["context"] = context
    example["target"] = target
    return example


max_seq_length = 1024


def preprocess(example):
    prompt = example["context"]
    target = example["target"]
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(
        target, max_length=max_seq_length, truncation=True, add_special_tokens=False
    )
    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def filter_nan(example):
    return example["target"] is not None and example["context"] is not None


column_names = dataset["train"].column_names

tokenized_datasets = dataset.shuffle().map(
    function=format_example, remove_columns=column_names
).filter(function=filter_nan)
tokenized_datasets = tokenized_datasets.map(function=preprocess)
print("length of the train dataset is ",len(tokenized_datasets["train"]))

from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl

save_steps=(len(tokenized_datasets["train"])/5)*5
save_steps=50
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    # per_device_eval_batch_size=1,
    # evaluation_strategy="steps",
    # eval_steps=50,
    logging_steps=10,
    gradient_accumulation_steps=1,
    num_train_epochs=num_epochs,
    weight_decay=0.1,
    # warmup_steps=1_000,
    # lr_scheduler_type="cosine",
    learning_rate=lr,
    save_steps=save_steps,
    fp16=True,
    push_to_hub=False,
    remove_unused_columns=False,
)

trainer = MyTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["valid"],
    # callbacks=[eccb]
)
trainer.train()

model.save_pretrained(output_dir)
