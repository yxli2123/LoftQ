import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig

HF_TOKEN = "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"
REPO_TOKEN = "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"

model = AutoModelForSeq2SeqLM.from_pretrained(
        'LoftQ/bart-large-bit4-iter1-rank64',
        load_in_4bit=True,
        load_in_8bit=False,
        device_map='auto',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4',
        ),
        trust_remote_code=True,
        token=REPO_TOKEN,
    )

print(model)
for k, v in model.named_parameters():
    print(k, v.shape)

