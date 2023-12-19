accelerate launch train_summarization.py \
--learning_rate 1e-4 \
--seed 11 \
--dataset_name cnn_dailymail \
--dataset_config "3.0.0" \
--pad_to_max_length \
--max_source_length 512 \
--output_dir exp_results/cnn_dailymail/lr1e-5 \
--num_train_epochs 15 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 4 \
--model_name_or_path LoftQ/bart-large-bit4-rank16 \
--do_train --do_eval --do_predict \
--logging_steps 100 \
--save_steps 10000 \
--evaluation_strategy epoch \
--report_to tensorboard \
--predict_with_generate

# lora fine-tune xsum
accelerate launch train_summarization.py \
--learning_rate 1e-4 \
--seed 11 \
--dataset_name xsum \
--dataset_config "3.0.0" \
--pad_to_max_length \
--max_source_length 512 \
--output_dir exp_results/xsum/lr1e-5 \
--num_train_epochs 25 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 4 \
--model_name_or_path LoftQ/bart-large-bit4-rank16 \
--do_train --do_eval --do_predict \
--logging_steps 100 \
--save_steps 10000 \
--evaluation_strategy epoch \
--report_to tensorboard \
--predict_with_generate
