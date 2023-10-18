# train 3-bit 64-rank llama-2-7b on wikitext-2 using multiple GPUs
accelerate launch train_gsm8k.py \
--fake_quantization \
--model_name_or_path LoftQ/Llama-2-13b-hf-bit3-rank64 \
--output_dir exp_results/gsm8k/bit3-rank64_ft_fake \
--learning_rate 1e-4  \
--seed 202 \
--dataset_name gsm8k \
--dataset_config main \
--pad_to_max_length \
--max_source_length 128 \
--max_target_length 256 \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 8 \
--with_tracking \
--report_to tensorboard

# train 4-bit 64-rank llama-2-7b on wikitext-2 using one GPU
python train_gsm8k.py \
--model_name_or_path LoftQ/Llama-2-13b-hf-bit4-rank64 \
--output_dir /mnt/t-qingzhang/exp_results/gsm8k/bit4-rank64_ft_real \
--learning_rate 1e-4  \
--seed 202 \
--dataset_name gsm8k \
--dataset_config main \
--pad_to_max_length \
--max_source_length 128 \
--max_target_length 256 \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--with_tracking \
--report_to tensorboard
