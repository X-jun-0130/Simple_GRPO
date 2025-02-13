# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Description:  GRPO训练
# Author:       XXJ
# Date:         2025/2/8
# Copyright 2025 The HuggingFace Team. All rights reserved.
# -------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes=5 /AAA-Nlp_2023/LLM-RL/Model_GRPO/grpo.py

#############################
#命令行中num_processes = 5 表示训练时使用5个GPU，vllm推理时默认使用剩下得最后一个GPU
#############################

import re
import os
import torch
from datasets import load_dataset
from rewards import accuracy_reward, format_reward, language_consistency_reward
from trl import ModelConfig, GRPOTrainer, GRPOConfig


#############################
# Initialize the Hyperparameters
#############################
training_args = GRPOConfig(output_dir="./Train_Result/Qwen2-3B-GRPO",
                            max_prompt_length = 512,
                            max_completion_length = 4096, 
                            max_steps = 572,
                            per_device_train_batch_size= 1,          # batch size per device during training
                            per_device_eval_batch_size = 4,           # batch size for evaluation
                            warmup_ratio=0.1,                       # number of warmup steps for learning rate scheduler
                            lr_scheduler_type ='cosine',
                            weight_decay=0.01,                      # strength of weight decay
                            logging_steps= 5,
                            save_strategy='steps',
                            eval_strategy = 'no',
                            save_steps = 286,
                            learning_rate= 5e-6,
                            bf16=True,
                            gradient_checkpointing=True,
                            gradient_checkpointing_kwargs= {'use_reentrant':False},
                            gradient_accumulation_steps = 2,
                            temperature = 0.6,
                            use_vllm = True,
                            vllm_device = 'auto',
                            vllm_gpu_memory_utilization = 0.8,
                            use_liger_kernel = True,
                            num_generations = 7,
                            deepspeed="./config_file/ds_config_grpo.json",
                            report_to='tensorboard',
                            )
#全局批次大小（ num_processes * per_device_train_batch_size ）必须能够整除 num_generations 

model_args = ModelConfig(model_name_or_path="/Model_WiNGPT_SFT/qwen-3B-coldstart/",
                        torch_dtype="bfloat16",
                        attn_implementation="flash_attention_2",
                        )

model_kwargs = dict(
    attn_implementation=model_args.attn_implementation,
    torch_dtype=torch.bfloat16,
    use_cache=False,
)
training_args.model_init_kwargs = model_kwargs

#############################
# Initialize the Reward Function
#############################
reward_funcs = [accuracy_reward, format_reward, language_consistency_reward]


#############################
# Initialize the Load Dataset
#############################
SYSTEM_PROMPT = '你是AI助手，回答问题时你应该一步一步地思考并反思。'

# Format into conversation
def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "solution": example["solution"],
        "task": example["task"],
        "languages": example["language"],
    }
    
# Load the dataset
dataset = load_dataset("json", data_files='./Model_GRPO/grpo_train_data.json', split="train", cache_dir="/workspace/cache_dir/")
dataset = dataset.map(make_conversation, num_proc=32, remove_columns=["question", "solution", "task", "language"])
print(dataset[0])
dataset = dataset.shuffle()

#############################
# Initialize the GRPO trainer
#############################
trainer = GRPOTrainer(
    model = model_args.model_name_or_path,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()