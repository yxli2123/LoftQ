# LoftQ: train 4-bit 64-rank llama-2-7b with LoftQ on GSM8K using 8 A100s
# global batch_size=64
accelerate launch train_gsm8k.py \
  --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
  --learning_rate 3e-4 \
  --seed 11 \
  --expt_name gsm8k_llama2_7b_4bit_64rank_loftq_fake \
  --output_dir exp_results/ \
  --num_train_epochs 6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --do_train \
  --report_to tensorboard

# LoftQ: train 4-bit 64-rank llama-3-8b with LoftQ on GSM8K using 8 A100s
# global batch_size=64
accelerate launch train_gsm8k.py \
  --model_name_or_path LoftQ/Meta-Llama-3-8B-4bit-64rank \
  --learning_rate 5e-4 \
  --seed 11 \
  --expt_name gsm8k_llama3_8b_4bit_64rank_loftq_fake \
  --output_dir exp_results/ \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --do_train \
  --report_to tensorboard


# QLoRA: train 4-bit 64-rank llama-2-7b with QLoRA on GSM8K using 8 A100s
# global batch_size=64
python train_gsm8k.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --token YOUR_HF_TOKEN \
  --learning_rate 3e-4 \
  --seed 11 \
  --rank 64 --bits 4 --lora_alpha 16 --lora_init \
  --expt_name gsm8k_llama2_7b_4bit_64rank_qlora \
  --output_dir exp_results/ \
  --num_train_epochs 6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --do_train \
  --report_to tensorboard

# LoRA: train 16-bit 64-rank llama-2-7b with LoRA on GSM8K using 8 A100s
# global batch_size=64
accelerate launch train_gsm8k.py \
  --full_precision \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --token YOUR_HF_TOKEN \
  --learning_rate 3e-4 \
  --seed 11 \
  --rank 64 --lora_alpha 16 --lora_init \
  --expt_name gsm8k_llama2_7b_4bit_64rank_fp16 \
  --output_dir exp_results/ \
  --num_train_epochs 6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --do_train \
  --report_to tensorboard
