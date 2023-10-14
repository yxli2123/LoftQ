import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import utils
import argparse
import os
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import json
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

    # bart
    if 'bart' in args.model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2', 'out_proj']
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
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                 inference_mode=False,
                                 r=args.reduced_rank,
                                 lora_alpha=args.reduced_rank,
                                 lora_dropout=0.1,
                                 target_modules=target_modules
                                 )

    else:
        raise NotImplementedError("model not supported")

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = model.to('cuda')

    # This is only for decomposition
    if args.num_bits in [2, 4]:
        utils.substitute_layer_weights(model,
                                       num_bits=args.num_bits,
                                       num_iter=args.num_iter,
                                       reduced_rank=args.reduced_rank,
                                       allow_name=target_modules,
                                       method=args.method)
    elif args.num_bits in [2.25, 2.5, 3]:
        bit_to_layer = {2.25: 4, 2.5: 8, 3: 16}
        lora_weight = {'name': None, 'lora_A': None, 'lora_B': None}
        for name, param in model.named_parameters():
            if 'lora' not in name and any(an in name for an in target_modules):
                num_bits = 4 if any(f"layers.{i}" in name for i in range(bit_to_layer[args.num_bits])) else 2

                weight, lora_A, lora_B = utils.qlora_init(weight=param,
                                                          num_bits=num_bits,
                                                          num_iter=args.num_iter,
                                                          reduced_rank=args.reduced_rank,
                                                          method=args.method)

                param.data = weight
                lora_weight['name'] = name.replace('.weight', '')
                lora_weight['lora_A'] = lora_A
                lora_weight['lora_B'] = lora_B
            elif lora_weight['name'] is not None and lora_weight['name'] in name and 'lora_A' in name:
                param.data = lora_weight['lora_A']
            elif lora_weight['name'] is not None and lora_weight['name'] in name and 'lora_B' in name:
                param.data = lora_weight['lora_B']
            else:
                print("Warning: decomposed weight does not match its layer.\n")

    for name, param in model.named_parameters():
        print(name, param.shape, param.max(), param.min(), param.mean(), param.requires_grad)

    model.base_model.save_pretrained(ckpt_dir)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    repo.push_to_hub(commit_message="Upload decomposed weights", auto_lfs_prune=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='normal', choices=['normal', 'uniform'])
    parser.add_argument('--path_to_model_zoo', type=str, default='./yixiaoli_model_zoo_hf/')
    parser.add_argument('--model_name', type=str, default='facebook/bart-large',
                        help='tiiuae/falcon-7b, meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-7b-chat-hf, facebook/bart-large')
    parser.add_argument('--num_bits',       type=float, default=4)
    parser.add_argument('--reduced_rank',   type=int, default=64)
    parser.add_argument('--num_iter',       type=int, default=0)

    args = parser.parse_args()

    # edit_lora_alpha(args)
    main(args)
    # lora_only(args)
