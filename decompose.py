import torch.cuda
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)

import utils
import argparse
import os
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from huggingface_hub import Repository, create_repo

HF_TOKEN = "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"
REPO_TOKEN = "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"


def main(args):
    # create repo
    ckpt_dir = os.path.join(args.path_to_model_zoo,
                            args.model_name.split('/')[-1],
                            f"bit{args.num_bits}",
                            f"iter{args.num_iter}",
                            f"rank{args.reduced_rank}")

    args.num_bits = int(args.num_bits) if args.num_bits - int(args.num_bits) == 0 else args.num_bits
    repo_name = "LoftQ/" + args.model_name.split('/')[-1] + f"-bit{args.num_bits}" + f"-rank{args.reduced_rank}"
    repo_id = create_repo(repo_name, exist_ok=True, token=REPO_TOKEN).repo_id
    repo = Repository(ckpt_dir, clone_from=repo_id, token=REPO_TOKEN)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=HF_TOKEN)
    config = AutoConfig.from_pretrained(args.model_name, use_auth_token=HF_TOKEN)
    print(config)

    # bart
    if 'bart' in args.model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2', 'out_proj']
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings', 'lora']
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                                 inference_mode=False,
                                 r=args.reduced_rank,
                                 lora_alpha=args.reduced_rank,
                                 lora_dropout=0.1,
                                 target_modules=target_modules
                                 )

    # llama
    elif 'llama' in args.model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                     use_auth_token=HF_TOKEN,
                                                     device_map='auto')
        block_name = ['lm_head', 'norm', 'embed_tokens', 'lora']
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                 inference_mode=False,
                                 r=args.reduced_rank,
                                 lora_alpha=args.reduced_rank,
                                 lora_dropout=0.1,
                                 target_modules=target_modules
                                 )
    elif 'deberta' in args.model_name.lower():
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        target_modules = ['query_proj', 'key_proj', 'value_proj', 'dense', 'embeddings']
        block_name = ['pooler', 'classifier', 'LayerNorm', 'lora']
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                                 inference_mode=False,
                                 r=args.reduced_rank,
                                 lora_alpha=args.reduced_rank,
                                 lora_dropout=0.1,
                                 target_modules=target_modules
                                 )

    else:
        raise NotImplementedError("model not supported")

    if args.fake_quant:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        utils.replace_module(
            model,
            allow_name=target_modules,
            block_name=block_name,
            prename='model',
            reduced_rank=args.reduced_rank,
            num_bits=args.num_bits,
            num_iter=args.num_iter,
            enable_lora=True,
            num_layers=config.num_hidden_layers,
            empty_init=False,
            quant_method=args.method,
            fake_quant=args.fake_quant,
        )

        model.base_model.save_pretrained(ckpt_dir)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

        repo.push_to_hub(commit_message="Upload decomposed weights", auto_lfs_prune=True)

    else:
        pass

    for name, param in model.named_parameters():
        print(name, param.shape, param.max(), param.min(), param.mean(), param.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='normal', choices=['normal', 'uniform'])
    parser.add_argument('--path_to_model_zoo', type=str, default='./yixiaoli_model_zoo_hf/')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base',
                        help='tiiuae/falcon-7b, meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-7b-chat-hf, facebook/bart-large')
    parser.add_argument('--num_bits', type=float, default=4)
    parser.add_argument('--reduced_rank', type=int, default=64)
    parser.add_argument('--num_iter', type=int, default=0)
    parser.add_argument('--fake_quant', action='store_true')

    args = parser.parse_args()

    # edit_lora_alpha(args)
    main(args)
    # lora_only(args)

    # from accelerate import init_empty_weights
    # with init_empty_weights():
    #     config = AutoConfig.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=HF_TOKEN)
    #     my_model = AutoModel.from_config(config)
    #     for k, v in my_model.named_parameters():
    #         print(k, v.shape)
