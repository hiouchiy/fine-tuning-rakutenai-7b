# Databricks notebook source
# MAGIC %md
# MAGIC # QLoRAでRakutenAI-7B-chatをマルチGPUでデータ並列にファインチューニングする
# MAGIC
# MAGIC [RakutenAI-7B-chat](https://huggingface.co/Rakuten/RakutenAI-7B-chat)大規模言語モデル(LLM)は、70億個のパラメータを持つインストラクション・チューニングされた生成テキストモデルです。このモデルはMistral-7B-v0.1に基づいており、すべてのベンチマークでLlama 2 13Bを凌駕しています。
# MAGIC
# MAGIC このノートブックは、[bbz662bbz/databricks-dolly-15k-ja-gozarinnemon](https://huggingface.co/datasets/bbz662bbz/databricks-dolly-15k-ja-gozarinnemon)データセット上で[RakutenAI-7B-chat](https://huggingface.co/Rakuten/RakutenAI-7B-chat)モデルを微調整するためのものです。
# MAGIC
# MAGIC このノートブックの環境
# MAGIC - ランタイム: 15.0 LTS GPU ML Runtime
# MAGIC - インスタンス: Azure上の`Standard_NC96ads_A100_v4`。
# MAGIC
# MAGIC Hugging FaceのPEFTライブラリと、よりメモリ効率の良い微調整のためにQLoRAを活用しています。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 必要なパッケージのインストール
# MAGIC
# MAGIC 以下のセルを実行して、必要なライブラリをセットアップしてインストールする。今回の実験では、最近の[`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer)を活用するために、`accelerate`, `peft`, `transformers`, `datasets` とTRLが必要である。bitsandbytes`を使用して、ベースモデルを4bitに量子化する](https://huggingface.co/blog/4bit-transformers-bitsandbytes)。

# COMMAND ----------

# MAGIC %pip install bitsandbytes einops trl peft
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# パラメータを定義
model_output_location = "/local_disk0/rakutenai-7b-lora-fine-tune"
local_output_dir = "/local_disk0/results"

model_name = "Rakuten/RakutenAI-7B-chat"
revision = "b54331c1376938325bdcf6a61f48f171fd8b1594"

# COMMAND ----------

# MAGIC %md
# MAGIC ## データセット
# MAGIC
# MAGIC [bbz662bbz/databricks-dolly-15k-ja-gozarinnemon](https://huggingface.co/datasets/bbz662bbz/databricks-dolly-15k-ja-gozarinnemon)データセットを使用する。

# COMMAND ----------

from datasets import load_dataset

dataset_name = "bbz662bbz/databricks-dolly-15k-ja-gozarinnemon"
dataset = load_dataset(dataset_name, split="train")

# COMMAND ----------

# MAGIC %md
# MAGIC ## プロンプトのフォーマット

# COMMAND ----------

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_KEY = "USER:"
INPUT_KEY = "INPUT:"
RESPONSE_KEY = "ASSISTANT:"

PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}""".format(
  intro=INTRO_BLURB,
  instruction_key=INSTRUCTION_KEY,
  instruction="{instruction}",
  response_key=RESPONSE_KEY,
  response="{response}",
)

PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}""".format(
  intro=INTRO_BLURB,
  instruction_key=INSTRUCTION_KEY,
  instruction="{instruction}",
  input_key=INPUT_KEY,
  input="{input}",
  response_key=RESPONSE_KEY,
  response="{response}",
)

def apply_prompt_template(examples):
  instruction = examples["instruction"]
  response = examples["output"]
  context = examples.get("input")

  if context:
    full_prompt = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
  else:
    full_prompt = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
  return { "text": full_prompt }

dataset = dataset.map(apply_prompt_template)
split_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=45)
train_tokenized_dataset = split_dataset['train']
eval_tokenized_dataset = split_dataset['test']

# COMMAND ----------

train_tokenized_dataset["text"][0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルをトレーニングする

# COMMAND ----------

# MAGIC %md
# MAGIC それではモデルを訓練してみましょう！

# COMMAND ----------

import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def train_model():
  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
  from accelerate import PartialState
  from peft import LoraConfig
  from trl import SFTTrainer
  from transformers import TrainingArguments, DataCollatorForLanguageModeling

  tokenizer = AutoTokenizer.from_pretrained(
      model_name, 
      trust_remote_code=True, 
      add_eos_token=True, 
      use_fast=False)
  tokenizer.pad_token = tokenizer.unk_token
  tokenizer.padding_side = "right"

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant= False,
  )

  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      revision=revision,
      trust_remote_code=True,
      device_map={'': PartialState().process_index} # ポイントはここ。各GPUにモデルのコピーをロードする。
  )
  model.config.use_cache = False

  linear_layers = find_all_linear_names(model)
  print(f"Linear layers in the model: {linear_layers}")

  lora_alpha = 16
  lora_dropout = 0.1
  lora_r = 64

  peft_config = LoraConfig(
      lora_alpha=lora_alpha,
      lora_dropout=lora_dropout,
      r=lora_r,
      bias="none",
      task_type="CAUSAL_LM",
      target_modules=linear_layers,
  )

  per_device_train_batch_size = 4
  gradient_accumulation_steps = 4
  optim = "paged_adamw_32bit"
  logging_steps = 100
  learning_rate = 2e-4
  max_grad_norm = 0.3
  warmup_ratio = 0.03
  lr_scheduler_type = "constant"

  training_arguments = TrainingArguments(
      output_dir=local_output_dir,
      per_device_train_batch_size=per_device_train_batch_size,
      per_device_eval_batch_size=per_device_train_batch_size,
      gradient_accumulation_steps=gradient_accumulation_steps,
      optim=optim,
      logging_steps=logging_steps,
      learning_rate=learning_rate,
      weight_decay=0.001,
      fp16=False,
      bf16=True,
      max_grad_norm=max_grad_norm,
      warmup_ratio=warmup_ratio,
      group_by_length=True,
      lr_scheduler_type=lr_scheduler_type,
      ddp_find_unused_parameters=False,
      num_train_epochs=2,
      report_to=[],
      push_to_hub=False,  # DO NOT PUSH TO MODEL HUB FOR NOW,
      load_best_model_at_end=True, # RECOMMENDED
      metric_for_best_model="eval_loss", # RECOMMENDED
      evaluation_strategy="epoch", # RECOMMENDED
      save_strategy="epoch",
  )

  data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),

  max_seq_length = 512

  trainer = SFTTrainer(
      model=model,
      train_dataset=train_tokenized_dataset,
      eval_dataset=eval_tokenized_dataset,
      peft_config=peft_config,
      dataset_text_field="text",
      max_seq_length=max_seq_length,
      tokenizer=tokenizer,
      args=training_arguments,
      packing=False,
      # data_collator=data_collator
  )

  for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)
  
  trainer.train()

  return trainer.state.best_model_checkpoint

# COMMAND ----------

from pyspark.ml.torch.distributor import TorchDistributor
 
best_model_checkpoint = TorchDistributor(
  num_processes=4, 
  local_mode=True, 
  use_gpu=True).run(train_model)

print("The best model checkpoint is saved in " + best_model_checkpoint)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルを保存する

# COMMAND ----------

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

peft_model_id = best_model_checkpoint

tokenizer = AutoTokenizer.from_pretrained(model_name)

config = PeftConfig.from_pretrained(peft_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
  best_model_checkpoint, 
  return_dict=True, 
  load_in_4bit=True, 
  device_map={"":0},
)
# base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.eval()

# ストリーマーを定義
streamer = TextStreamer(
    tokenizer,
    skip_prompt=False, # 入力文(ユーザーのプロンプトなど)を出力するかどうか
    skip_special_tokens=False, # その他のデコード時のオプションもここで渡す
)

# COMMAND ----------



prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request. 
### USER:　肺がんの特徴を簡潔に教えてください。 
### ASSISTANT:"""
batch = tokenizer(prompt, add_special_tokens=True, return_tensors='pt').to('cuda')

with torch.cuda.amp.autocast():
  output_tokens = model.generate(
      input_ids = batch.input_ids, 
      max_new_tokens=300,
      temperature=0.7,
      top_p=0.7,
      num_return_sequences=1,
      do_sample=True,
      pad_token_id=tokenizer.eos_token_id,
      eos_token_id=tokenizer.eos_token_id,
      streamer=streamer,
  )

generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(generated_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ファインチューニングしたモデルをMLFlowに記録する

# COMMAND ----------

import torch
from peft import PeftModel, PeftConfig

peft_model_id = model_output_location
config = PeftConfig.from_pretrained(peft_model_id)

from huggingface_hub import snapshot_download
# Download the Mistral-7B-v0.1 model snapshot from huggingface
snapshot_location = snapshot_download(repo_id=config.base_model_name_or_path)


# COMMAND ----------

import mlflow
class Rakuten7BQLORA(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['repository'])
    self.tokenizer.pad_token = tokenizer.eos_token
    config = PeftConfig.from_pretrained(context.artifacts['lora'])
    base_model = AutoModelForCausalLM.from_pretrained(
      context.artifacts['repository'], 
      return_dict=True, 
      load_in_4bit=True, 
      device_map={"":0},
      trust_remote_code=True,
    )
    self.model = PeftModel.from_pretrained(base_model, context.artifacts['lora'])
  
  def predict(self, context, model_input):
    prompt = model_input["prompt"][0]
    temperature = model_input.get("temperature", [1.0])[0]
    max_tokens = model_input.get("max_tokens", [100])[0]
    batch = self.tokenizer(prompt, padding=True, truncation=True,return_tensors='pt').to('cuda')
    with torch.cuda.amp.autocast():
      output_tokens = self.model.generate(
          input_ids = batch.input_ids, 
          max_new_tokens=max_tokens,
          temperature=temperature,
          top_p=0.7,
          num_return_sequences=1,
          do_sample=True,
          pad_token_id=tokenizer.eos_token_id,
          eos_token_id=tokenizer.eos_token_id,
      )
    generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return generated_text

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import pandas as pd
import mlflow

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":[prompt], 
            "temperature": [0.5],
            "max_tokens": [100]})

with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Rakuten7BQLORA(),
        artifacts={'repository' : snapshot_location, "lora": peft_model_id},
        pip_requirements=["torch", "transformers", "accelerate", "einops", "loralib", "bitsandbytes", "peft"],
        input_example=pd.DataFrame({"prompt":[prompt], "temperature": [0.5],"max_tokens": [100]}),
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC MLFlowに記録されたモデルでモデル推論を実行する

# COMMAND ----------

import mlflow
import pandas as pd


prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
if one get corona and you are self isolating and it is not severe, is there any meds that one can take?

### Response: """
# Load model as a PyFuncModel.
run_id = run.info.run_id
logged_model = f"runs:/{run_id}/model"

loaded_model = mlflow.pyfunc.load_model(logged_model)

text_example=pd.DataFrame({
            "prompt":[prompt], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Predict on a Pandas DataFrame.
loaded_model.predict(text_example)
