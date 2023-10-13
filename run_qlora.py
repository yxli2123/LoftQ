import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    AutoTokenizer
)

HF_TOKEN = "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"
REPO_TOKEN = "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"

def main():
    model_name = 'LoftQ/bart-large-bit4-iter1-rank64'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    qmodel = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
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

    sentence = ["you are beautiful", "you look perfect tonight"]
    model_input = tokenizer(sentence, padding=True, return_tensors='pt')
    output_fp = model(**model_input)
    output_nf = qmodel(**model_input)

    logit_fp = output_fp[0][0, 0]
    logit_nf = output_nf[0][0, 0]
    print(logit_fp)
    print(logit_nf)

    kl_div_1 = torch.nn.functional.kl_div(logit_fp, logit_nf)
    kl_div_2 = torch.nn.functional.kl_div(logit_nf, logit_fp)

    print(kl_div_1, kl_div_2)


if __name__ == '__main__':
    main()
