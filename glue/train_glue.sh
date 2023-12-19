export TASK_NAME=mnl9
export INT_BIT=2
export LR=1e-4
export LowRatio=8
export Seeds=0
CUDA_VISIBLE_DEVICES=0 python run_glue.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --num_train_epochs 5 \
  --learning_rate $LR \
  --output_dir output \
  --int_bit $INT_BIT \
  --seed $Seeds \
  --loftq \
  --reduced_rank 32 \
  --decomposed_pretrained_ckpt_path decomposed_pretrained_ckpt_path \
  --quant_embedding \
  --gradient_accumulation_steps 2 \
  --quant_method uniform \
  --num_warmup_steps 0 \
  --num_iter 5 \
  --decompose