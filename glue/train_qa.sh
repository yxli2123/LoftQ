export TASK_NAME=squad
export INT_BIT=2
export LR=5e-5
export LowRatio=8
export Seeds=0
## Use loftq to sepcify loftq training; use qlora to specify qlora training
## Specify the rank of low rank: reduced_rank
CUDA_VISIBLE_DEVICES=0 python run_qa.py \
--model_name_or_path microsoft/deberta-v3-base \
--dataset_name squad \
--do_train \
--do_eval \
--per_device_train_batch_size 8 \
--learning_rate 5e-5 \
--num_train_epochs 10 \
--output_dir output \
--evaluation_strategy steps \
--eval_steps 2500 \
--save_steps 2500 \
--overwrite_output_dir \
--int_bit $INT_BIT \
--seed $Seeds \
--loftq \
--reduced_rank 32 \
--quant_embedding \
--num_iter 5  \
--quant_method uniform \
--gradient_accumulation_steps 2  \
--decompose \
--decomposed_pretrained_ckpt_path decomposed-weight.bin \
--fp16 \
--fp16_opt_level o2 \
--fp16_full_eval