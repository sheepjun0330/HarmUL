#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

model="Llama-3.2-1B-Instruct"
model_path="meta-llama/Llama-3.2-1B-Instruct"
methods=("GradAscent" "GradDiff" "NPO" "SimNPO" "RMU")

for trainer in "${methods[@]}"; do
  task_name="json_${trainer}"
  echo "Running $trainer with local JSON datasets"

  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=${trainer} \
  task_name=${task_name} \
  model=${model} \
  model.model_args.pretrained_model_name_or_path=${model_path} \
  data/datasets@data.forget=JSON_QA_forget \
  data/datasets@data.retain=JSON_QA_retain

done

# DPO requires alternate responses in forget set (question, answer, alternate)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/idk.yaml \
trainer=DPO \
task_name=json_DPO \
model=${model} \
model.model_args.pretrained_model_name_or_path=${model_path} \
data/datasets@data.forget=JSON_QA_forget_alt \
data/datasets@data.retain=JSON_QA_retain
