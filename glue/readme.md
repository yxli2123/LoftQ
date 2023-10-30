# Implementation for GLUE experiments using LoftQ with DeBERTa-V3-base
We implement the glue experiments here. Currently we only support fake quantization.


## Main results:
**DebBERTa-V3-base on GLUE using uniform quantization with 2-bits precision**

| **Rank** | **Method** | **MNLI** | **QNLI** | **RTE** | **SST** | **MRPC** | **CoLA** | **QQP** | **STSB** | **SQuAD** | **ANLI** |
|----------|------------|----------|----------|---------|---------|----------|----------|---------|----------|-----------|----------|
|         |           | m / mm   | Acc      | Acc     | Acc     | Acc      | Matt      | Acc     | P/S Corr | EM/F1     | Acc      |
|         | Full FT    | 90.5/90.6 | 94.0     | 82.0    | 95.3    | 89.5/93.3 | 69.2     | 92.4/89.8 | 91.6/91.1 | 88.5/92.8 | 59.8     |
| 16       | LoRA       | 90.4/90.5 | 94.6     | 85.1    | 95.1    | 89.9/93.6 | 69.9     | 92.0/89.4 | 91.7/91.1 | 87.3/93.1 | 60.2     |
| 16        | LoftQ     | **84.7/85.1** | **86.6** | **61.4** | **90.2** | **83.8/88.6** | **37.4** | **90.3/86.9** | **87.1/86.9** | **81.5/88.6** | **47.1** |
| 32        | LoftQ     | **86.0/86.1** | **89.9** | **61.7** | **92.0** | **83.6/87.2** | **47.5** | **91.0/87.9** | **87.5/87.0** | **82.9/89.8** | **49.0** |

| **Rank** | **Method** | **MNLI** | **QNLI** | **RTE** | **SST** | **MRPC** | **CoLA** | **QQP** | **STSB** | **SQuAD** |
|----------|------------|----------|----------|---------|---------|----------|----------|---------|----------|-----------|
|         |           | m / mm   | Acc      | Acc     | Acc     | Acc      | Matt      | Acc     | P/S Corr | Em/F1     |
|         | Full FT    | 90.5/90.6 | 94.0     | 82.0    | 95.3    | 89.5/93.3 | 69.2     | 92.4/89.8 | 91.6/91.1 | 88.5/92.8 |
| 16       | LoRA       | 90.4/90.5 | 94.6     | 85.1    | 95.1    | 89.9/93.6 | 69.9     | 92.0/89.4 | 91.7/91.1 | 87.3/93.1 |
| 16        | OurAlg     | **87.3/87.1** | **90.6** | **61.1** | **94.0** | **87.0/90.6** | **59.1** | **90.9/88.0** | **87.9/87.6** | **84.4/91.2** |
| 32        | OurAlg     | **88.0/88.1** | **92.2** | **63.2** | **94.7** | **87.5/91.2** | **60.5** | **91.3/88.3** | **89.5/89.2** | **85.2/91.6** |


### Quantize Models
We use the following the codes to quantize the model, initialize the low rank adapters using LoftQ, and prepare for the training
```python
import utils
allow_name = ['query', 'key', 'value',
                  'q_proj', 'k_proj', 'v_proj',
                  'query_proj', 'key_proj', 'value_proj',
                  'out_proj', 'dense', 'attention', 'fc1', 'fc2']
block_name = ['pooler', 'classifier', 'LayerNorm']

utils.replace_module(model,
                     allow_name=allow_name,
                     block_name=block_name,
                     reduced_rank=32,
                     decomposition=True,
                     quant_method=uniform,
                     int_bit=2,
                     args=args,
                     )
utils.show_model_stats(model, mark_only_lora_as_trainable=True)
```

### Training file
See example training file for GLUE task and SQuAD in ```run_glue.py``` and  ```run_qa.py```

### Scripts
Sample script is given in the ```train_qa.sh``` and ```train_glue.sh```

**Arguments**
- `--reduced_rank`: rank of low rank adapters
- `--int_bit`: integer bit precision
- `--qlora`: whether use qlora to init low rank adapters
- `--loftq`: whether use loftq to init low rank adapters
- `--decompose`: whether quantize and decompose the model; if not used then the model will load pre-decomposed quantizedckpt
- `--decomposed_pretrained_ckpt_path`: used when decompose is false; the path of predecomposed quantized model. 
- `--quant_embedding`: whether quantize embedding
- `--quant_method`: which quantization method used
- `--num_iter`: number of iteration used in loftq
- `--eval`: whether doing eval


**Sample Checkpoints**

You can easily access the sample checkpoints using wget command

| Model Name      | TASK      |  Rank |  Performance(Acc) |
| ------------    | --------- | ---------------- |--------------|
| [deberta_mnli_loftq_uniform_32](https://www.dropbox.com/scl/fi/5gprwddymu8ukw15ld29z/deberta_base_r32_mnli.bin?rlkey=vp8ygxlqe5zz42l7kesby93oi) | MNLI      |  32              |  88.0
| [deberta_mnli_loftq_uniform_16](https://www.dropbox.com/scl/fi/rt9ckeowrf4k9pd0swwpd/deberta_base_r16_mnli.bin?rlkey=j8m84g6sxty2dxpa6y4e5ux1e) | MNLI      |  16              |  87.3
| [deberta_mrpc_loftq_uniform_32](https://www.dropbox.com/scl/fi/flv2rmxi726vatzbuddug/deberta_base_r32_mrpc.bin?rlkey=ci18guma3fs4wfgc6r15wbrl6) | MRPC     |  32              |   87.5