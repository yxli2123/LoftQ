# train 2-bit 64-rank llama-2-7b on wikitext-2 using 1 GPU
python train_clm.py \
--model_name_or_path LoftQ/Llama-2-7b-hf-bit4-rank64 \
--output_dir exp_results/wikitext-2/bit4-rank64_ft \
--learning_rate 1e-4  \
--seed 888 \
--dataset_name wikitext \
--dataset_config wikitext-2-raw-v1 \
--num_train_epochs 2 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--do_train \
--do_eval \
--logging_steps 50 \
--evaluation_strategy epoch \
--report_to tensorboard \
--overwrite_output_dir \
--block_size 1024 \


# train 2-bit 64-rank llama-2-7b on wikitext-2 using multiple GPUs
accelerate launch train_clm.py \
--fake_quantization \
--model_name_or_path LoftQ/Llama-2-7b-hf-bit4-rank64 \
--output_dir exp_results/wikitext-2/bit4-rank64_ft \
--learning_rate 1e-4  \
--seed 888 \
--dataset_name wikitext \
--dataset_config wikitext-2-raw-v1 \
--num_train_epochs 2 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--do_train \
--do_eval \
--logging_steps 50 \
--evaluation_strategy epoch \
--report_to tensorboard \
--overwrite_output_dir \
--block_size 1024
