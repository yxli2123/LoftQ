# <img src="asset/loftq_logo_mini.png" alt="LoftQ_logo" style="zoom:100%;" /> LoftQ: LoRA-Fine-Tuning-Aware Quantization

LoftQ helps you fine-tune LLMs with limited GPUs. 🚀 LoftQ finds good enough quantized LoRA initialization: quantized backbone Q and LoRA adapters A and B, given a pre-trained weight W.

This repo implements the paper 🔗: [LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/abs/2310.08659).

Our models are available on 🤗 [LoftQ Huggingface Hub](https://huggingface.co/LoftQ)

## News
- [04/20/2024] New `LLAMA-3-8B`results on GSM8K. See results [here](#llama-3-on-gsm8k). Check out LoftQ of 🦙 [LLAMA-3](https://huggingface.co/LoftQ/Meta-Llama-3-8B-4bit-64rank), [CodeLLAMA-7b](https://huggingface.co/LoftQ/CodeLlama-7b-hf-4bit-64rank), [CodeLLAMA=13b](https://huggingface.co/LoftQ/CodeLlama-13b-hf-4bit-64rank) on [Huggingface Hub](https://huggingface.co/LoftQ).

- [04/13/2024] New `phi-2` results on GSM8K. See results [here](#phi-2-on-gsm8k). Check out LoftQ of [Phi-2](https://huggingface.co/LoftQ/phi-2-4bit-64rank) on [Huggingface Hub](https://huggingface.co/LoftQ).

- [04/13/2024] Update `script/train_gsm8k.sh` to support data parallel of quantized models.


## Quick Start

### Requirements
We use [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) to implement the quantization. 
This package only support CUDA >= 11.0 and does not support CPU. 
However, we also provide fake quantization for fast and parallel training if GPUs are adequate.

`pip install -r requirements.txt`

### Steps

1. Apply LoftQ to a full-precision pre-trained weight and save.
2. Load LoftQ initialization and train.

For step 1, we have provided off-the-shelf LoftQ initializations (see [supported model list](#appendix-off-the-shelf-model-list)) 
in [Huggingface Hub LoftQ](https://huggingface.co/LoftQ).
If you want to do it yourself, jump to [LoftQ DIY](#loftq-diy).

For step 2, below is an example of loading 4bit Mistral-7B with 64rank LoRA adapters from Huggingface Hub.
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# fetch the MODEL_ID at https://huggingface.co/LoftQ
MODEL_ID = "LoftQ/Mistral-7B-v0.1-4bit-64rank"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16,  # you may change it with different models
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is recommended
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    ),
)
peft_model = PeftModel.from_pretrained(
    base_model,
    MODEL_ID,
    subfolder="loftq_init",
    is_trainable=True,
)

# Do training with peft_model ...
```

## LoftQ DIY

### Apply LoftQ and save
We provide [quantize_save.py](quantize_save.py) as an example to apply LoftQ with 
different bits(`--bits`), ranks(`--rank`), and alternating steps (`--iter`, a hyper-parameter in LoftQ, see Algorithm 1 in [LoftQ paper](https://arxiv.org/abs/2310.08659)). Currently, this example supports
`llama-2`, `falcon`, `mistral`, `bart`, `t5`, `deberta`, `bert`, `roberta`.

Below is an example of obtaining 4bit LLAMA-2-7b with 16-rank LoRA adapters by 5 alternating steps.
```sh
SAVE_DIR="model_zoo/loftq/"
python quantize_save_load.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \  # high-precision model id in HF
    --token HF_TOKEN \  # your HF token if the model is private, e.g., llama-2
    --bits 4 \
    --iter 5 \
    --rank 16 \
    --save_dir $SAVE_DIR
```

The above commands end up with creating the model directory under `$SAVE_DIR`. 
Specifically, the model directory is named as 

`MODEL_DIR = SAVE_DIR + f"{args.model_name_or_path.split('/')[-1]}-{args.bits}bits-{args.rank}rank"`

In this example, `MODEL_DIR="model_zoo/loftq/Llama-2-7b-hf-4bit-16rank"`, where the backbone is stored in `$MODEL_DIR`
and the LoRA adapters are at the sub-folder `$MODEL_DIR/loftq_init`.

### Load and train
Similar to loading from Huggingface Hub, we only need to change the `MODEL_ID` to the `MODEL_DIR`.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_DIR = "model_zoo/loftq/Llama-2-7b-hf-4bit-16rank"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, 
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    ),
)
peft_model = PeftModel.from_pretrained(
    base_model,
    MODEL_DIR,
    subfolder="loftq_init",
    is_trainable=True,
)
# Do training with peft_model ...
```

## LoftQ Fine-tuning

We also provide an example to fine-tune LLAMA-7b with LoftQ on GSM8K. 

```shell
python train_gsm8k.py \
    --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
    --learning_rate 3e-4 \
    --seed 11 \
    --expt_name gsm8k_llama2_7b_4bit_64rank_loftq \
    --output_dir exp_results/ \
    --num_train_epochs 6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --do_train \
    --report_to tensorboard
```

### Other training Files

* GLUE: `glue/run_glue.py`
* Question Answering: `glue/run_qa.py`
* Summarization: `train_summarization.py`
* WikiText-2: `train_clm.py`
* GSM8K: `train_gsm8k.py`

More example scripts are in [scripts](scripts).

## Quick Evaluation
Here is the command to test GSM8K with adapters we have fine-tuned. It is stored in the `subfolder='gsm8k'` 
of the target model in [LoftQ Huggingface hub](https://huggingface.co/LoftQ).
```shell
python test_gsm8k.py \
    --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
    --batch_size 16
```

```shell
python test_gsm8k.py \
    --model_name_or_path LoftQ/phi-2-4bit-64rank \
    --batch_size 16
```
Feel free to change `batch_size` to accommodate to your machine.


## Main Results

### LLAMA-2 on WikiText-2 and GSM8K

| Bit  | WikiText-2 | WikiText-2  | GSM8K      | GSM8K       |
| ---- | ---------- | ----------- | ---------- | ----------- |
|      | LLAMA-2-7b | LLAMA-2-13b | LLAMA-2-7b | LLAMA-2-13b |
| 16   | 5.08       | 5.12        | 36.9       | 43.1        |
| 4    | 5.24       | 5.16        | 35.0       | 45.0        |
| 3    | 5.63       | 5.13        | 32.9       | 44.4        |
| 2.5  | 5.78       | 5.22        | 31.1       | 41.1        |
| 2.25 | 6.13       | 5.45        | 26.5       | 38.1        |
| 2    | 7.85       | 7.69        | 20.9       | 25.4        |

Models are fine-tuned through causal language modeling on training sets and are tested on validation/test sets.

### Phi-2 on GSM8K

| Model   | Bits | Rank | LoRA Initial           | GSM8K     |
| --------| ---- | ---- | ---------------------- | --------- |
| Phi-2   | 16   | -    | Full model fine-tuning | 66.8±1.2  |
| Phi-2   | 16   | 64   | Gaussian + 0           | 64.8±0.5  |
| Phi-2   | 4    | 64   | Gaussian + 0 (QLoRA)   | 60.2±0.6  |
| Phi-2   | 4    | 64   | LoftQ                  | 64.1±0.7  |

### LLAMA-3 on GSM8K

| Model      | Bits | Rank | LoRA Initial           | GSM8K     |
| -----------| ---- | ---- | ---------------------- | --------- |
| LLAMA-3-8B | 16   | -    | Full model fine-tuning | 70.4±0.7  |
| LLAMA-3-8B | 16   | 64   | Gaussian + 0 (LoRA)    | 69.3±1.5  |
| LLAMA-3-8B | 4    | 64   | Gaussian + 0 (QLoRA)   | 67.4±1.0  |
| LLAMA-3-8B | 4    | 64   | LoftQ                  | 68.0±0.6  |


Models are fine-tuned through causal language modeling on (reformatted) training sets and are tested on validation/test sets.

### BART-large on CNN/DailyMail and XSum

| Bit     | Rank | XSum              | CNN/DailyMail     |
|---------|------|-------------------|-------------------|
| Lead-3* |      | 16.30/1.60/11.95  | 40.42/17.62/36.67 |
| 16      | 16   | 43.95/20.72/35.68 | 45.03/21.84/42.15 |
| 4       | 16   | 44.51/21.14/36.18 | 43.96/21.06/40.96 |
| 2       | 16   | 40.81/17.85/32.80 | 42.52/19.81/39.51 |
| 16      | 8    | 43.40/20.20/35.20 | 44.72/21.58/41.84 |
| 4       | 8    | 44.08/20.72/35.89 | 43.81/20.95/40.84 |
| 2       | 8    | 39.63/16.65/31.62 | 42.24/19.44/29.04 |

*: Using the first 3 sentences in the document as the summary

### DeBERTa-V3-base on GLUE using Normal Float Datatype

| Bit | **Rank**| **MNLI** | **QNLI** | **RTE** | **SST** | **MRPC** | **CoLA** | **QQP** | **STSB** | **SQuAD** | **ANLI** |
|----------|------------|----------|----------|---------|---------|----------|----------|---------|----------|-----------|----------|
|         |           | m / mm   | Acc      | Acc     | Acc     | Acc      | Acc      | Mcc     | P/S Corr | EM/F1     | Acc      |
|    16     |  16   | 90.5/90.6 | 94.0     | 82.0    | 95.3    | 89.5/93.3 | 69.2     | 92.4/89.8 | 91.6/91.1 | 88.5/92.8 | 59.8     |
| 2        | 16     | **84.7/85.1** | **86.6** | **61.4** | **90.2** | **83.8/88.6** | **37.4** | **90.3/86.9** | **87.1/86.9** | **81.5/88.6** | **47.1** |
| 2        | 32    | **86.0/86.1** | **89.9** | **61.7** | **92.0** | **83.6/87.2** | **47.5** | **91.0/87.9** | **87.5/87.0** | **82.9/89.8** | **49.0** |

### DeBERTa-V3-base on GLUE using Uniform Quantization Datatype

| **Bit** | **Rank** | **MNLI** | **QNLI** | **RTE** | **SST** | **MRPC** | **CoLA** | **QQP** | **STSB** | **SQuAD** |
|----------|------------|----------|----------|---------|---------|----------|----------|---------|----------|-----------|
|         |           | m / mm   | Acc      | Acc     | Acc     | Acc      | Acc      | Mcc     | P/S Corr | Em/F1     |
|  16       | 16    | 90.5/90.6 | 94.0     | 82.0    | 95.3    | 89.5/93.3 | 69.2     | 92.4/89.8 | 91.6/91.1 | 88.5/92.8 |
| 2        | 16     | **87.3/87.1** | **90.6** | **61.1** | **94.0** | **87.0/90.6** | **59.1** | **90.9/88.0** | **87.9/87.6** | **84.4/91.2** |
| 2        | 32     | **88.0/88.1** | **92.2** | **63.2** | **94.7** | **87.5/91.2** | **60.5** | **91.3/88.3** | **89.5/89.2** | **85.2/91.6** |


## Citation
```bibtext
@article{li2023loftq,
  title={Loftq: Lora-fine-tuning-aware quantization for large language models},
  author={Li, Yixiao and Yu, Yifan and Liang, Chen and He, Pengcheng and Karampatziakis, Nikos and Chen, Weizhu and Zhao, Tuo},
  journal={arXiv preprint arXiv:2310.08659},
  year={2023}
}
```

## Appendix: Off-the-shelf Model List
| Model Name  | Bits | Ranks |
|---------------| ---- | ----- |
| LLAMA-3-8B    | 4    | 64    |
| CodeLLAMA-7b  | 4    | 64    |
| CodeLLAMA-13b | 4    | 64    |
| Phi-2         | 4    | 64    |
| LLAMA-2-7b    | 4    | 64    |
| LLAMA-2-13b   | 4    | 64    |
| LLAMA-2-70b   | 4    | 64    |
| Mistral       | 4    | 64    |
| Mistral       | 4    | 32    |
| BART-large    | 4    | 8     |
| BART-large    | 4    | 16    |
| BART-large    | 4    | 32    |
| BART-large    | 2    | 8     |
