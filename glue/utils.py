import torch
import math
import random
from torch import nn
from utils_qaunt import weight_quant_fn
import torch.nn.functional as F

def explore_grad(weight):
    if weight.requires_grad is True:
        print(weight.shape)
        print(weight)
        print(weight.grad)


def low_rank_decomposition(weight, reduced_rank=0.15):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param      rank_ratio: rank_of_decomposed_matrix
    :return: L, R
    """

    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"

    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    reduced_rank = int(reduced_rank)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return L, R


class LinearQuantLoRA(nn.Module):
    def __init__(self, in_feature, out_feature, reduced_rank, has_bias=True, args=None):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.reduced_rank = reduced_rank
        self.has_bias = has_bias
        self.quant = nn.Linear(in_feature, out_feature, bias=False)

        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(out_feature, requires_grad=True))

        self.has_svd_adapter = args.loftq
        self.has_lora_adapter = args.qlora
        if self.has_svd_adapter:
            self.right = nn.Linear(in_feature, reduced_rank, bias=False)
            self.left = nn.Linear(reduced_rank, out_feature, bias=False)
            print(f"low rank adapter with rank {reduced_rank} using LoftQ")
        if self.has_lora_adapter:
            print(f"low rank adapter with rank {reduced_rank} using QLoRA")
            self.lora_A = nn.Linear(in_feature,reduced_rank, bias=False)
            self.lora_B = nn.Linear(reduced_rank, out_feature, bias=False)


    def forward(self, x):
        right_output = self.right(x) if self.has_svd_adapter else 0
        LRX = self.left(right_output) if self.has_svd_adapter else 0
        HX = self.quant(x)
        Y = HX + LRX + self.bias if self.has_bias else HX + LRX
        if self.has_lora_adapter:
            lora_A_output = self.lora_A(x)
            Y += self.lora_B(lora_A_output)

        return Y

    def initialize_weight(self, quant_weight, left_weight, right_weight, sparse_weight=None, bias=None):
        self.quant.weight = nn.Parameter(quant_weight, requires_grad=False)  # Freeze the backbone

        if self.has_bias:
            self.bias = nn.Parameter(bias, requires_grad=True)
        if self.has_svd_adapter:
            self.left.weight = nn.Parameter(left_weight, requires_grad=True)
            self.right.weight = nn.Parameter(right_weight, requires_grad=True)

        if self.has_lora_adapter:
            lora_A_weight = nn.Parameter(self.quant.weight.new_zeros((self.reduced_rank, self.in_feature)),
                                         requires_grad=True)
            lora_B_weight = nn.Parameter(self.quant.weight.new_zeros((self.out_feature, self.reduced_rank),
                                         requires_grad=True))
            nn.init.kaiming_uniform_(lora_A_weight, a=math.sqrt(5))
            nn.init.zeros_(lora_B_weight)
            self.lora_A.weight = lora_A_weight
            self.lora_B.weight = lora_B_weight


def quant_first_iter(weight, L, R, reduced_rank, int_bit, quant_method,
                     **kwargs):
    low_rank_product = L @ R if torch.is_tensor(L) else 0
    residual = weight - low_rank_product
    quant_w = weight_quant_fn(residual,
                              num_bits=int_bit,
                              quant_method=quant_method)

    output = low_rank_decomposition(weight - quant_w, reduced_rank=reduced_rank )
    L, R = output[0], output[1]
    final_residual = weight - quant_w - L @ R
    return weight, L, R, quant_w, final_residual


def replace_module(module,
                             allow_name=None,
                             block_name=None,
                             reduced_rank=32,
                             decomposition=True,
                             quant_method='uniform',
                             int_bit=4,
                             args=None,
                             **kwargs):
    """
    :param         int_bit: integer bit, 8, 4, 2 for example
    :param    quant_method: quantization method to use
    :param   decomposition: whether quantize
    :param          module: an nn.Module class
    :param      block_name: do not continue to iterate when the module's name is in the block_name
    :param      allow_name: replace the module if its name is in the allow_name
    :param reduced_rank: rank of low rank adapters
    :return: None
    """

    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query', 'key', 'value', 'dense', 'attention']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if args.quant_embedding and type(target_attr) == nn.Embedding and "token" not in attr_str:
            print("====================================================")
            print(attr_str, target_attr)
            if decomposition:
                weights = target_attr.weight
                L, R = 0, 0
                for i in range(args.num_iter):
                    weights, L, R, quant_w, final_residual = quant_first_iter(weights, L, R, reduced_rank,
                                                                              int_bit, quant_method,
                                                                              **kwargs)
                if args.num_iter==0:
                    quant_w = weight_quant_fn(weights,
                              num_bits=int_bit,
                              quant_method=quant_method)

                new_embedding = LinearQuantEmbedding(target_attr.weight.shape[0], target_attr.weight.shape[1],
                                                     r=int(reduced_rank), args=args)
                new_embedding.initialize_weight(quant_w, L, R, 0, 0)
            else:
                weights = target_attr.weight
                H, W = target_attr.weight.shape
                L = torch.zeros(H, int(reduced_rank), requires_grad=True)
                R = torch.zeros(int(reduced_rank), W, requires_grad=True)
                new_embedding = LinearQuantEmbedding(target_attr.weight.shape[0], target_attr.weight.shape[1],
                                                     r=int(reduced_rank), args=args)
                new_embedding.initialize_weight(weights, L, R, 0, 0)
            setattr(module, attr_str, new_embedding)
        if type(target_attr) == nn.Linear and any(attr_str in an for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr.weight.shape[0])
            if decomposition:
                weights = target_attr.weight
                L, R = 0,0
                for i in range(args.num_iter):
                    weights, L, R, quant_w, final_residual = quant_first_iter(weights,L,R,reduced_rank,int_bit,quant_method,**kwargs)
                if args.num_iter==0:
                    quant_w = weight_quant_fn(weights,
                              num_bits=int_bit,
                              quant_method=quant_method)
                linear_loras = LinearQuantLoRA(target_attr.in_features, target_attr.out_features, reduced_rank=int(reduced_rank),
                                               has_bias=True,  args=args)
                linear_loras.initialize_weight(quant_w, L, R, 0, target_attr.bias)

            else:
                H, W = target_attr.weight.shape
                reduced_rank = int(reduced_rank)
                L = torch.zeros(H, reduced_rank, requires_grad=True)
                R = torch.zeros(reduced_rank, W, requires_grad=True)
                S = torch.zeros(H, W, requires_grad=True)
                quant_weight = torch.zeros(H, W, requires_grad=False)

                # Create a nn.Module and assign decomposed weights to the parameters
                linear_loras = LinearQuantLoRA(target_attr.in_features, target_attr.out_features, reduced_rank=int(reduced_rank),
                                               has_bias=True, args=args)

                linear_loras.initialize_weight(quant_weight, L, R, S, target_attr.bias)

            setattr(module, attr_str, linear_loras)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            replace_module(immediate_child_module, allow_name, block_name, reduced_rank,
                                      decomposition, quant_method, int_bit, args=args, **kwargs)

def show_model_stats(model,mark_only_lora_as_trainable=True):
    total = 0
    lr_adapter = 0
    if mark_only_lora_as_trainable:
        for n, m in model.deberta.named_parameters():
            if 'lora' in n or 'left' in n or 'right' in n:
                m.requires_grad = True
                lr_adapter += m.numel()
            else:
                if "quant" in n or "word_embeddings.weight" in n:
                    print(n, m)
                m.requires_grad = False
            print(n, m.shape, m.requires_grad)
            total += m.numel()

    else:
        for n, m in model.deberta.named_parameters():
            if "quant" in n or "word_embeddings.weight" in n:
                print(n, m)
            if m.requires_grad:
                lr_adapter += m.numel()
                print(lr_adapter)
            total += m.numel()
    print(f"Total trainable parameters {lr_adapter}")
    print(f"We finetune about {lr_adapter / total} ratio of percentages")


class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LinearQuantEmbedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            r: int = 0,
            lora_alpha: int = 1,
            merge_weights: bool = True,
            args=None,
            **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters

        self.weight.requires_grad = False
        self.has_svd_adapter = args.loftq
        self.has_lora_adapter = args.qlora
        if self.has_svd_adapter:
            self.left = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.right = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            print(f"low rank adapter with rank {r} using LoftQ")
        if self.has_lora_adapter:
            print(f"low rank adapter with rank {r} using QLoRa")
            self.lora_A = nn.Parameter(self.weight.new_zeros((num_embeddings, r)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((r, embedding_dim)))



    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        # if hasattr(self, 'lora_A'):
        #     # initialize A the same way as the default for nn.Linear and B to zero
        #     nn.init.zeros_(self.lora_A)
        #     nn.init.normal_(self.lora_B)

    def initialize_weight(self, quant_weight, left_weight, right_weight, sparse_weight=None, bias=None):
        self.weight = nn.Parameter(quant_weight, requires_grad=False)
        if self.has_svd_adapter:
            self.left = nn.Parameter(left_weight, requires_grad=True)
            self.right = nn.Parameter(right_weight, requires_grad=True)

        if self.has_lora_adapter:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)


    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)

    def forward(self, x: torch.Tensor):
        result = nn.Embedding.forward(self, x)
        if self.has_svd_adapter:
            after_left = F.embedding(
                x, self.left
            )
            result += (after_left @ self.right)
        if self.has_lora_adapter:
            after_A = F.embedding(
                x, self.lora_A
            )
            result += (after_A @ self.lora_B)
        return result

