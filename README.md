# <img src="asset/loftq_logo_mini.png" alt="LoftQ_logo" style="zoom:100%;" /> LoftQ: LoRA-Fine-Tuning-Aware Quantization

This repo implements [LoftQ: LoRA-Fine-Tuning-Aware Quantization](https://arxiv.org/abs/2310.08659).

## Overview

A novel quantization framework that simultaneously quantizes an LLM and finds a proper low-rank initialization for LoRA fine-tuning. 
This framework alternatively apply quantization and SVD to obtain the initialization of the quantized backbone and low-rank adapters.

## Main Results

**LLAMA-2 on WikiText-2 and GSM8K**

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

**BART-large on CNN/DailyMail and XSum**

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

**DeBERTa-V3-base on GLUE**

## LoftQ
We use huggingface 🤗 as our training code scripts. See examples [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch)

### Requirements
We use [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) to implement the quantization. 
This package only support CUDA >= 11.0 and does not support CPU. 
However, we also provide fake quantization for fast and parallel training if GPUs are adequate.

`pip install -r requirements.txt`

### Quantize Models
Given a pre-trained model `pretrained_model`, simply call 
```python
import utils
utils.replace_module(
        pretrained_model
        prename='model',
        allow_name=['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1_proj', 'fc2_proj'],
        block_name=['LayerNorm', 'classifier', 'lm_head'],
        reduced_rank=32,
        num_bits=4,
        num_iter=1,
        enable_lora=True,
        num_layers=32,
        empty_init=True,
        quant_method='normal',
        fake_quant=True,
)
```

* *module*: have to inherit nn.Module
* *prename*: previous name, used to iteratively obtain parameters name
* *allow_name*: allowed nn.Linear to quantize
* *block_name*: blocked nn.Linear to quantize
* *reduced_rank*: low-rank rank
* *num_bits*: low-precision bits. 2,4,8 as expected, float number between (2, 4) enables mixed precision
* *num_iter*: alternating steps
* *enable_lora*: whether enable lora part in forward pass
* *num_layers*: total number of layers. can be obtained by the model config file
* *empty_init*: True for the first time decomposition, False for loading model from checkpoints
* *quant_method*: choose in \['normal', 'uniform'\], other quantization method not supported
* *fake_quant*: True for fake quantization where values change but memory not saved; False for real quant

Examples of quantizing LLAMA-2, BART, DeBERTa-V3 are in `quantize.py`. 

### Download Quantized Model
We provide quantized models with LoRA adapters obtained by LoftQ. 
These models are available on https://huggingface.co/LoftQ.
To use these models, for example, a 2-bit 64-rank LLAMA-2-13b, call
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
                'LoftQ/Llama-2-13b-hf-bit2-rank64',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type='nf4',
                ),
            )
```

### Training Files

* GLUE: 
* Question Answering: 
* Summarization: `train_summarization.py`
* WikiText-2: `train_clm.py`
* GSM8K: `train_gsm8k.py`

Example scripts are in [scripts](scripts).

