export MODEL_NAME="anhnct/audioldm2_gigaspeech"

accelerate launch --mixed_precision="fp16"  train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir "Train_data/metadata.txt" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="Checkpoint" \
  --num_train_epochs 7 \
  --use_ema
