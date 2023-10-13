import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import utils
import argparse
import os
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import json
from huggingface_hub import Repository, create_repo

HF_TOKEN = "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"


def main(args):
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
    print(model)
    #model = model.to('cuda')

    # This is only for decomposition
    utils.substitute_layer_weights(model,
                                   num_bits=args.num_bits,
                                   num_iter=args.num_iter,
                                   reduced_rank=args.reduced_rank,
                                   allow_name=target_modules,
                                   method=args.method)
    #
    # lora_weight = {'name': None, 'lora_A': None, 'lora_B': None}
    # for name, param in model.named_parameters():
    #     if 'lora' not in name and any(an in name for an in target_modules):
    #         num_bits = 4 if any(f"layers.{i}" in name for i in range(args.high_bit_layer)) else 2
    #
    #         weight, lora_A, lora_B = utils.qlora_init(weight=param,
    #                                                   num_bits=num_bits,
    #                                                   num_iter=args.num_iter,
    #                                                   reduced_rank=args.reduced_rank,
    #                                                   method=args.method)
    #
    #         param.data = weight
    #         lora_weight['name'] = name.replace('.weight')
    #         lora_weight['lora_A'] = lora_A
    #         lora_weight['lora_B'] = lora_B
    #     elif lora_weight['name'] in name and 'lora_A' in name:
    #         param.data = lora_weight['lora_A']
    #     elif lora_weight['name'] in name and 'lora_B' in name:
    #         param.data = lora_weight['lora_B']
    #     else:
    #         print("Warning: decomposed weight does not match its layer.\n")

    for name, param in model.named_parameters():
        print(name, param.shape, param.max(), param.min(), param.mean(), param.requires_grad)

    ckpt_dir = os.path.join(args.path_to_model_zoo,
                            args.model_name.split('/')[-1],
                            f"bit{args.num_bits}",
                            f"iter{args.num_iter}",
                            f"rank{args.reduced_rank}")
    # ckpt_dir = os.path.join(args.path_to_model_zoo,
    #                         args.model_name.split('/')[-1],
    #                         f"high_bit_layer{args.high_bit_layer}",
    #                         f"iter{args.num_iter}",
    #                         f"rank{args.reduced_rank}")

    # save
    repo_name = "LoftQ/" + args.model_name.split('/')[-1] + f"-bit{args.num_bits}" + f"-iter{args.num_iter}" + f"-rank{args.reduced_rank}"
    repo_id = create_repo(repo_name, exist_ok=True, token=REPO_TOKEN).repo_id
    # Clone repo locally
    repo = Repository(ckpt_dir, clone_from=repo_id, token=REPO_TOKEN)

    base_model = model.get_base_model()
    model.save_pretrained(ckpt_dir)
    base_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    repo.push_to_hub(commit_message="Upload decomposed weights", auto_lfs_prune=True)


def edit_lora_alpha(args):
    ckpt_dir = os.path.join(args.path_to_model_zoo,
                            args.model_name.split('/')[-1],
                            f"bit{args.num_bits}",
                            f"iter{args.num_iter}",
                            f"rank{args.reduced_rank}")
    file_path = os.path.join(ckpt_dir, "adapter_config.json")
    with open(file_path, "r") as fp:
        config = json.load(fp)
        fp.close()

    config['lora_alpha'] = args.reduced_rank

    with open(file_path, "w") as outfile:
        json.dump(config, outfile, indent=4)


def lora_only(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=HF_TOKEN)

    # bart
    if 'bart' in args.model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2', 'out_proj']
        model = utils.replace_lora_only(model, target_modules, None, reduced_rank=args.reduced_rank, load_mode=False)

    # llama
    elif 'llama' in args.model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                     use_auth_token=HF_TOKEN,
                                                     device_map='auto')
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
        model = utils.replace_lora_only(model, target_modules, None, reduced_rank=args.reduced_rank, load_mode=False)

    else:
        raise NotImplementedError("model not supported")

    print(model)
    model = model.to('cuda')

    for n, p in model.named_parameters():
        p.requires_grad = False
        print(n, p.size(), p.min(), p.max(), p.device, p.requires_grad)

    ckpt_dir = os.path.join(args.path_to_model_zoo,
                            args.model_name.split('/')[-1],
                            f"low_rank_only"
                            f"rank{args.reduced_rank}")

    # save
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='normal', choices=['normal', 'uniform'])
    parser.add_argument('--path_to_model_zoo', type=str, default='./yixiaoli_model_zoo_hf/')
    parser.add_argument('--model_name', type=str, default='facebook/bart-large',
                        help='tiiuae/falcon-7b, meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-7b-chat-hf, facebook/bart-large')
    parser.add_argument('--num_bits',       type=int, default=4)
    parser.add_argument('--reduced_rank',   type=int, default=64)
    parser.add_argument('--num_iter',       type=int, default=1)
    parser.add_argument('--high_bit_layer', type=int, default=4)

    args = parser.parse_args()

    # edit_lora_alpha(args)
    main(args)
    # lora_only(args)
