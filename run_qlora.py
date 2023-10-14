import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    AutoTokenizer
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType



HF_TOKEN = "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"
REPO_TOKEN = "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"

def main():
    model_name = 'LoftQ/Llama-2-7b-hf-bit4-iter5-rank64'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=REPO_TOKEN)
    tokenizer.pad_token_id = 0       # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    model_fp32 = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Llama-2-7b-hf',
            device_map='auto',
            trust_remote_code=True,
            use_auth_token=HF_TOKEN)
    model_bit4_fake = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=REPO_TOKEN)
    model_bit4_real = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            load_in_8bit=False,
            device_map='auto',
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
            ),
            trust_remote_code=True,
            use_auth_token=REPO_TOKEN,
        )

    model_bit4_fake = PeftModel.from_pretrained(model_bit4_fake,
                                                model_name,
                                                is_trainable=False,
                                                use_auth_token=REPO_TOKEN)
    model_bit4_real = PeftModel.from_pretrained(model_bit4_real,
                                                model_name,
                                                is_trainable=False,
                                                use_auth_token=REPO_TOKEN)
    model_bit4_fake = model_bit4_fake.to('cuda')
    model_bit4_real = model_bit4_real.to('cuda')

    sentence = ["you are beautiful", "you look perfect tonight, really?"]
    model_input = tokenizer(sentence, padding=True, return_tensors='pt')
    model_input = {k: v.to('cuda') for k, v in model_input.items()}

    logits_fp32 = model_fp32(**model_input)[0]
    logits_4bit_fake = model_bit4_fake(**model_input)[0]
    logits_4bit_real = model_bit4_real(**model_input)[0]

    for i in range(20):
        print(logits_fp32[0, 0, i])
        print(logits_4bit_fake[0, 0, i])
        print(logits_4bit_real[0, 0, i])


if __name__ == '__main__':
    main()
