# test 4-bit 64-rank llama-2-7b with LoftQ on GSM8K using one A100
python test_gsm8k.py \
  --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
  --ckpt_dir exp_results/gsm8k_llama2_7b_4bit_64rank_loftq/Llama-2-7b-hf-4bit-64rank/ep_8/lr_0.0003/seed_11/ \
  --batch_size 16

# test 4-bit 64-rank llama-2-7b with LoftQ on GSM8K using one A100,
# for adapters in HF subfolder
python test_gsm8k.py \
  --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
  --batch_size 16
