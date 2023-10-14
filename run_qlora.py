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
    model_name = 'LoftQ/bart-large-bit4-iter0-rank64'
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=REPO_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    qmodel = AutoModelForSeq2SeqLM.from_pretrained(
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
            token=REPO_TOKEN,
        )
    qqmodel = AutoModelForSeq2SeqLM.from_pretrained(
        'facebook/bart-large',
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
        token=REPO_TOKEN,
    )

    sentence = ["you are beautiful", "you look perfect tonight"]
    model_input = tokenizer(sentence, padding=True, return_tensors='pt')
    output_fp = model(**model_input)
    output_q = qmodel(**model_input)
    output_qq = qqmodel(**model_input)

    logit_fp = output_fp[0][0, 0]
    logit_q = output_q[0][0, 0]
    logit_qq = output_qq[0][0, 0]

    logit_fp = torch.nn.functional.softmax(logit_fp.to(torch.float32))
    logit_q = torch.nn.functional.softmax(logit_q.to(torch.float32))
    logit_qq = torch.nn.functional.softmax(logit_qq.to(torch.float32))

    print(logit_fp)
    print(logit_q)
    print(logit_qq)

    kl_div_1 = torch.nn.functional.kl_div(logit_fp, logit_q)
    kl_div_2 = torch.nn.functional.kl_div(logit_q, logit_qq)
    kl_div_3 = torch.nn.functional.kl_div(logit_qq, logit_q)
    kl_div_4 = torch.nn.functional.kl_div(logit_fp, logit_qq)

    print("fp:ourq", kl_div_1)
    print("ourq:bnbq", kl_div_2)
    print("bnbq:ourq", kl_div_3)
    print("fp:bnbq", kl_div_4)


if __name__ == '__main__':
    main()
