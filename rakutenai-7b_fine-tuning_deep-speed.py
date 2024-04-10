# Databricks notebook source
# MAGIC %md
# MAGIC # DeepspeedTorchDistributorを使ってRakutenAI-7Bをファインチューニングする
# MAGIC
# MAGIC 楽天株式会社の[Rakuten/RakutenAI-7B-chat](https://huggingface.co/Rakuten/RakutenAI-7B-chat) をファインチューニングの例を説明する。Apache SparkのDeepspeedTorchDistributorとHugging Face `transformers` ライブラリを利用する。 
# MAGIC
# MAGIC ## 環境: 
# MAGIC
# MAGIC - GPU、マルチノードクラスタ。[DeepSpeed](https://www.deepspeed.ai/)は現在CPU上での実行をサポートしていません。
# MAGIC - Databricks Runtime 15.0 ML GPU
# MAGIC - Azure上の`Standard_NC96ads_A100_v4`
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from dataclasses import field, dataclass
import json
import logging
import os
import numpy as np
from pathlib import Path
import torch
from typing import Optional, Union, Tuple

from datasets import Dataset, load_dataset
import transformers

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    IntervalStrategy,
    PreTrainedTokenizer,
    SchedulerType,
    Trainer,
    TrainingArguments,
    set_seed,
)

# COMMAND ----------

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/dbfs/tmp/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"

# COMMAND ----------

# MAGIC %md
# MAGIC ## DeepSpeed 構成の定義
# MAGIC
# MAGIC ディストリビュータに DeepSpeed 構成を渡すかどうかを選択できます。省略した場合は、既定の構成が適用されます。
# MAGIC
# MAGIC 構成は、Python 辞書または `json` 構成を含むファイル・パスを表す文字列として渡すことができます。

# COMMAND ----------

deepspeed_config = {
    "fp16": {
      "enabled": False
    },
    "bf16": {
      "enabled": True
    },
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": "auto",
        "betas": "auto",
        "eps": "auto",
        "weight_decay": "auto"
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": "auto",
        "warmup_max_lr": "auto",
        "warmup_num_steps": "auto"
      }
    },
    "zero_optimization": {
      "stage": 3,
      "overlap_comm": True,
      "contiguous_gradients": True,
      "sub_group_size": 1e9,
      "reduce_bucket_size": "auto",
      "stage3_prefetch_bucket_size": "auto",
      "stage3_param_persistence_threshold": "auto",
      "stage3_max_live_parameters": 1e9,
      "stage3_max_reuse_distance": 1e9,
      "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のConfigはZeROのCPU OffloadingをONにしている設定になります。したがって、A10などの内蔵メモリーが小さいGPUを使用する場合にご利用ください。

# COMMAND ----------

# DBTITLE 1,[Backup] ZeRO CPU Offloading使用版
deepspeed_config = {
    "fp16": {
      "enabled": False
    },
    "bf16": {
      "enabled": True
    },
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": "auto",
        "betas": "auto",
        "eps": "auto",
        "weight_decay": "auto"
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": "auto",
        "warmup_max_lr": "auto",
        "warmup_num_steps": "auto"
      }
    },
    "zero_optimization": {
      "stage": 3,
      "overlap_comm": True,
      "contiguous_gradients": True,
      "sub_group_size": 5e7,
      "reduce_bucket_size": "auto",
      "reduce_scatter": True,
      "stage3_max_live_parameters" : 1e9,
      "stage3_max_reuse_distance" : 1e9,
      "stage3_prefetch_bucket_size" : 5e8,
      "stage3_param_persistence_threshold" : 1e6,
      "stage3_gather_16bit_weights_on_model_save": True,
      "offload_param": {
      "device": "cpu",
      "pin_memory": True
      },
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True
      }
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 50,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニング関数を定義する
# MAGIC
# MAGIC データセットは下記二つのうちのいずれかを使用します。
# MAGIC - [yulanfmy/databricks-qa-ja](https://huggingface.co/datasets/yulanfmy/databricks-qa-ja)（データブリックス関連のQAデータです。１５００件ほど。A100x4で学習時間は1時間ほど。）
# MAGIC - [bbz662bbz/databricks-dolly-15k-ja-gozarinnemon](https://huggingface.co/datasets/bbz662bbz/databricks-dolly-15k-ja-gozarinnemon)(語尾を必ず「ござる」で終わらせるテキストデータです。ややふざけているようですが、ファインチューニングの効果がわかりやすいデータセットです。15000件ほど。A100x4で学習時間は30時間ほど。)

# COMMAND ----------

MODEL_PATH = "Rakuten/RakutenAI-7B-chat"
TOKENIZER_PATH = "Rakuten/RakutenAI-7B-chat"
DEFAULT_TRAINING_DATASET = "yulanfmy/databricks-qa-ja"
# DEFAULT_TRAINING_DATASET = "bbz662bbz/databricks-dolly-15k-ja-gozarinnemon"
LOCAL_OUTPUT_DIR = "/dbfs/RakutenAI-7B-chat/output"
DEFAULT_SEED = 68

class HFTrainingArguments:
    local_rank: str = "-1"
    dataset: str = DEFAULT_TRAINING_DATASET
    model: str = MODEL_PATH
    tokenizer: str = TOKENIZER_PATH
    max_seq_len: int = 512

    final_model_output_path: str = "/local_disk0/final_model"

    output_dir: str = "/local_disk0/output"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-6
    optim: str = "adamw_hf"
    num_train_epochs: int = 2
    max_steps: int = -1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    lr_scheduler_type: Union[SchedulerType, str] = "cosine"
    warmup_steps: int = 0
    weight_decay: float = 1
    logging_strategy: Union[str, IntervalStrategy] = IntervalStrategy.STEPS
    evaluation_strategy: Union[str, IntervalStrategy] = IntervalStrategy.STEPS
    save_strategy: Union[str, IntervalStrategy] = IntervalStrategy.STEPS
    fp16: bool = False
    bf16: bool = True
    save_steps: int = 100
    logging_steps: int = 10

args = HFTrainingArguments()

# COMMAND ----------

# MAGIC %md
# MAGIC データセットの準備。
# MAGIC
# MAGIC 使用するデータセットに応じて_reformat_data関数を編集ください。

# COMMAND ----------

def get_tokenizer(
    pretrained_name_or_path: str = TOKENIZER_PATH,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_name_or_path, 
        trust_remote_code=True, 
        add_eos_token=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_training_dataset(
  tokenizer,
  path_or_dataset: str = DEFAULT_TRAINING_DATASET,
  max_seq_len: int = 512,
  seed: int = DEFAULT_SEED,
) -> Dataset:
    print(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset(path_or_dataset)["train"]
    print(f"Found {dataset.num_rows} rows")

    def _reformat_data(rec):
        #################### "yulanfmy/databricks-qa-ja"の場合はこちら　####################
        if rec["context"]:
            return f"""Below is an instruction that describes a task. Write a response in Japanese that appropriately completes the request. USER:{rec["instruction"]} INPUT:{rec["context"]} ASSISTANT:{rec["response"]}""" 
        else:
            return f"""Below is an instruction that describes a task. Write a response in Japanese that appropriately completes the request. USER:{rec["instruction"]} ASSISTANT:{rec["response"]}"""
        
        #################### "bbz662bbz/databricks-dolly-15k-ja-gozarinnemon"の場合はこちら　####################
        # if rec["input"]:
        #     return f"""Below is an instruction that describes a task. Write a response in Japanese that appropriately completes the request. USER:{rec["instruction"]} INPUT:{rec["input"]} ASSISTANT:{rec["output"]}""" 
        # else:
        #     return f"""Below is an instruction that describes a task. Write a response in Japanese that appropriately completes the request. USER:{rec["instruction"]} ASSISTANT:{rec["output"]}"""

    def tokenize_function(allEntries):
        return tokenizer(
            _reformat_data(allEntries), 
            truncation=True, 
            padding=True,
            max_length=max_seq_len,
            return_overflowing_tokens=False,
            return_length=True,
        )

        for length, input_ids, attention_mask in zip(
            outputs["length"], outputs["input_ids"], outputs["attention_mask"]
        ):
            if length == max_seq_len:
                input_batch.append(input_ids)
                attention_masks.append(attention_mask)

        return {"input_ids": input_batch, "attention_mask": attention_masks}

    dataset = dataset.map(tokenize_function)
    split_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=seed)
    train_tokenized_dataset = split_dataset['train']
    eval_tokenized_dataset = split_dataset['test']

    return train_tokenized_dataset, eval_tokenized_dataset

tokenizer = get_tokenizer(args.tokenizer)
train_dataset, eval_dataset = load_training_dataset(
    tokenizer, 
    path_or_dataset=DEFAULT_TRAINING_DATASET, 
    max_seq_len=args.max_seq_len
)

# COMMAND ----------

def get_model(
    pretrained_name_or_path: str = MODEL_PATH
) -> AutoModelForCausalLM:
    print(f"Loading model: {pretrained_name_or_path}")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map= None,
    )

    model.config.use_cache = False

    return model

def fine_tune_rakutenai():
    set_seed(DEFAULT_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = get_model(pretrained_name_or_path=args.model)

    training_args = TrainingArguments(
        local_rank=args.local_rank,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim=args.optim,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_strategy=args.logging_strategy,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=deepspeed_config,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=False,
        disable_tqdm=True,
        report_to=["tensorboard"],
        # group_by_length=True,
        ddp_find_unused_parameters=False,
        # fsdp=["full_shard", "offload"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Training the model")
    trainer.train()

    print(f"Saving Model to {args.final_model_output_path}")
    trainer.save_model(output_dir=args.final_model_output_path)
    tokenizer.save_pretrained(args.final_model_output_path)

    print("Training finished.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## DeepSpeedディストリビュータの作成
# MAGIC
# MAGIC ディストリビュータを作成する際に、使用するノード数とノードあたりのGPU数を指定できます。

# COMMAND ----------

from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor

dist = DeepspeedTorchDistributor(
  numGpus=4,
  nnodes=1,
  localMode=True,  # Set False if use Distribute training across workers.
  deepspeedConfig=deepspeed_config)

dist.run(fine_tune_rakutenai)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 学習済みモデルを推論する

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Rakuten/RakutenAI-7B-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(args.final_model_output_path, torch_dtype="auto", device_map="auto")
model.eval()

requests = [
    "スマホが人間に与える害についてに簡潔に教えてください。",
]

system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {user_input} ASSISTANT:"

for req in requests:
    input_req = system_message.format(user_input=req)
    input_ids = tokenizer.encode(input_req, return_tensors="pt").to(device=model.device)
    tokens = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.decode(tokens[0][len(input_ids[0]):], skip_special_tokens=True)
    print("USER:\n" + req)
    print("ASSISTANT:\n" + out)
    print()
    print()


# COMMAND ----------


