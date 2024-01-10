import os

import torch
import argparse
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="./llama-7b-hf")
parser.add_argument("--project", type=str, default="Mac")
parser.add_argument("--percentage", type=str, default="0.05")
args = parser.parse_args()

project = args.project
percentage = args.percentage
BASE_MODEL = args.base_model
LORA_WEIGHTS = "./trained/" + project + "/" + percentage

HF="./trained/" + project + "/" + percentage+"_1"

# assert (
#     BASE_MODEL
# ), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_WEIGHTS,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, HF, state_dict=deloreanized_sd, max_shard_size="400MB"
)

tokenizer.save_pretrained(HF)
