import os
import torch
from torch import Tensor
import math
import random
from torch import nn
import torch.nn.functional as F
from scipy.stats import norm
from torch import optim


def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
    variations = 2 ** num_bits

    if symmetric:
        print("symmetric nf4")
        v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
        values = []
        for index in range(len(v) - 1):
            values.append(0.5 * v[index] + 0.5 * v[index + 1])
        v = values
    else:
        # one more positive value, this is an asymmetric type
        print("asymmetric nf4")
        v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
        # print(torch.linspace(offset, 0.5, 9)[:-1])
        # print(v1)
        v2 = [0]
        # v2 = [0]*(256-15) ## we have 15 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
        # print(torch.linspace(offset, 0.5, 8)[:-1])
        # print(v3)
        v = v1 + v2 + v3

    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()
    # print(values)
    return values
    # assert values.


def low_rank_decomposition(weight, reduced_rank=32):
    """
    :param          weight: The matrix to decompose, of shape (H, W)
    :param    reduced_rank: the final rank
    :return:
    """

    """parameter_ratio = rank * (H + W) / (H * W)"""
    """rank_ratio = """
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"
    H, W = weight.size()

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    rank = torch.count_nonzero(S)
    is_full_rank = rank == min(H, W)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    print(f"W: ({H},{W}) | Rank: {rank} | U:{U.shape} | S:{S.shape} | Vh:{Vh.shape}")
    print(f"Reduced Rank: {reduced_rank} | Num Parameters: {(H + W) * reduced_rank}")
    print(f"L: {L.shape} | R: {R.shape}")

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, 'reduced_rank': reduced_rank}


class NFQuantizer:
    def __init__(self, num_bits=2, device='cuda', method='normal', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.device = device
        if method == 'normal':
            self.norm_lookup_table = self.create_normal_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        elif method == 'uniform':
            self.norm_lookup_table = self.create_uniform_map(num_bits=self.num_bits)
            self.norm_lookup_table = self.norm_lookup_table.to(device)
        else:
            raise NotImplementedError("Other quantization methods not supported yet.")

    @staticmethod
    def create_uniform_map(symmetric=False, num_bits=4):
        if symmetric:
            negative = torch.linspace(-1, 0, 2 ** (num_bits - 1))
            positive = torch.linspace(0, 1, 2 ** (num_bits - 1))
            table = torch.cat([negative, positive[1:]])
        else:
            table = torch.linspace(-1, 1, 2 ** num_bits)
        return table

    @staticmethod
    def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
        variations = 2 ** num_bits

        if symmetric:
            print("symmetric nf4")
            v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
            values = []
            for index in range(len(v) - 1):
                values.append(0.5 * v[index] + 0.5 * v[index + 1])
            v = values
        else:
            # one more positive value, this is an asymmetric type
            print("asymmetric nf4")
            v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
            # print(torch.linspace(offset, 0.5, 9)[:-1])
            # print(v1)
            v2 = [0]
            # v2 = [0]*(256-15) ## we have 15 non-zero values in this data type
            v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
            # print(torch.linspace(offset, 0.5, 8)[:-1])
            # print(v3)
            v = v1 + v2 + v3

        values = torch.Tensor(v)
        values = values.sort().values
        values /= values.max()
        # print(values)
        return values
        # assert values.

    def quantize_tensor(self, weight):
        max_abs = torch.abs(weight).max()
        weight_normed = weight / max_abs

        weight_normed_expanded = weight_normed.unsqueeze(-1)

        # Reshape L to have the same number of dimensions as X_expanded
        L_reshaped = torch.tensor(self.norm_lookup_table).reshape(1, -1)

        # Calculate the absolute difference between X_expanded and L_reshaped
        abs_diff = torch.abs(weight_normed_expanded - L_reshaped)

        # Find the index of the minimum absolute difference for each element
        qweight = torch.argmin(abs_diff, dim=-1)
        # print(min_index)
        return qweight, max_abs

    def dequantize_tensor(self, qweight, max_abs):
        qweight_flatten = qweight.flatten()

        weight_normed = self.norm_lookup_table[qweight_flatten]
        weight = weight_normed * max_abs

        weight = weight.reshape(qweight.shape)

        return weight

    def quantize_block(self, weight, block_size=64, method='normal'):
        assert len(weight.shape) == 2 and weight.shape[0] * weight.shape[1] % block_size == 0
        M, N = weight.shape
        device = weight.device

        # Quantization
        weight_flatten = weight.flatten()  # (M*N, )
        weight_block = weight_flatten.reshape(-1, block_size)  # (L, B), L = M * N / B
        if method == 'normal':
            weight_max = weight_block.abs().max(dim=-1)[0]  # (L, 1)
        elif method == 'uniform':
            weight_max = weight_block.mean(dim=-1) + 2.5 * weight_block.std(dim=-1)
        else:
            raise NotImplementedError("Method not supported yet.")
        weight_max = weight_max.unsqueeze(-1)
        weight_divabs = weight_block / weight_max  # (L, B)
        weight_divabs = weight_divabs.unsqueeze(-1)  # (L, B, 1)
        L_reshaped = self.norm_lookup_table.reshape(1, -1)  # (1, 2**K)

        abs_diff = torch.abs(weight_divabs - L_reshaped)  # (L, B, 2**K)
        qweight = torch.argmin(abs_diff, dim=-1)  # (L, B)

        # Pack multiple k-bit into uint8
        qweight = qweight.reshape(-1, 8 // self.num_bits)
        qweight_pack = torch.zeros((M * N // 8 * self.num_bits, 1), dtype=torch.uint8, device=device)

        # data format example:
        # [1, 0, 3, 2] or [01, 00, 11, 10]  -> [10110001], LIFO
        for i in range(8 // self.num_bits):
            qweight[:, i] = qweight[:, i] << i * self.num_bits
            qweight_pack[:, 0] |= qweight[:, i]

        return qweight_pack, weight_max, weight.shape

    def dequantize_block(self, qweight, weight_max, weight_shape, block_size=64):
        # unpack weight
        device = qweight.device
        weight = torch.zeros((qweight.shape[0], 8 // self.num_bits), dtype=torch.float32, device=device)
        for i in range(8 // self.num_bits):
            lookup_table_idx = qweight.to(torch.long) % 2 ** self.num_bits  # get the most right 2 bits
            lookup_table_idx = lookup_table_idx.to(torch.int)
            weight[:, i] = self.norm_lookup_table[lookup_table_idx].squeeze()
            qweight = qweight >> self.num_bits  # right shift 2 bits of the original data

        weight_block = weight.reshape(-1, block_size)
        weight = weight_block * weight_max
        weight = weight.reshape(weight_shape)

        return weight


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ret = input @ self.weight.T
        if self.bias is None:
            return ret
        else:
            return ret + self.bias


class QLinearLR(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 reduced_rank: int,
                 num_bits: int,
                 block_size=64,
                 enable_lora=True,
                 bias=None,
                 device='cuda',
                 ):
        super().__init__()
        self.num_bits = num_bits
        self.enable_lora = enable_lora
        self.quantizer = NFQuantizer(num_bits=num_bits)

        self.register_buffer('qweight', torch.empty((in_features * out_features // 8 * num_bits, 1), dtype=torch.uint8,
                                                    device=device))
        self.register_buffer('absmax', torch.empty((in_features * out_features // block_size, 1), dtype=torch.float32,
                                                   device=device))
        self.lora_A = nn.Parameter(torch.empty((reduced_rank, in_features), dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.empty((out_features, reduced_rank), dtype=torch.float32, device=device))

        self.bias = bias

        self.weight_size = torch.Size([out_features, in_features])
        self.weight_type = torch.float32
        self.block_size = block_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.quantizer.dequantize_block(self.qweight, self.absmax, self.weight_size, self.block_size)
        ret = input @ weight.T
        lora = (input @ self.lora_A.T) @ self.lora_B.T if self.enable_lora else 0

        return ret + lora + self.bias if self.bias is not None else ret + lora

    def initial_backbone(self, qweight, absmax):
        self.qweight = qweight
        self.absmax = absmax

    def initial_lora(self, lora_A, lora_B):
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B


class LinearLR(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 reduced_rank: int,
                 bias=False,
                 ):
        super().__init__()
        self.lora_A = nn.Linear(in_features, reduced_rank, bias=False)
        self.lora_B = nn.Linear(reduced_rank, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lora = self.lora_B(self.lora_A(input)) + self.bias
        return lora

    def initial_lora(self, lora_A, lora_B):
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B


def substitute_layer_weights_iter_quant(module,
                                        allow_name=None,
                                        block_name=None,
                                        reduced_rank=32,
                                        num_bits=4,
                                        num_iter=5,
                                        load=False,
                                        enable_lora=True):
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h',
                      'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']
    assert (num_bits == 8 or num_bits == 4 or num_bits == 2) and num_iter >= 0

    allow_module = [nn.Linear, Linear]

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if any(isinstance(target_attr, module) for module in allow_module) and any(an in attr_str for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr)
            linear_loras = QLinearLR(target_attr.in_features, target_attr.out_features,
                                     reduced_rank,
                                     num_bits,
                                     block_size=64,
                                     enable_lora=enable_lora,
                                     bias=target_attr.bias,
                                     device='cuda')

            if not load:
                weight = target_attr.weight.data
                out_feature, in_feature = weight.size()
                device = weight.device
                calibration = False
                quantizer = NFQuantizer(num_bits=num_bits, device=device, differentiable=calibration)
                res = weight.clone()

                for i in range(num_iter):
                    # Quantization
                    quantized_weight, max_abs, shape = quantizer.quantize_block(res)
                    dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape)
                    res = weight - dequantized_weight

                    # Decompose the residual by SVD
                    output = low_rank_decomposition(res, reduced_rank=reduced_rank)
                    L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']
                    res = weight - torch.mm(L, R)

                if num_iter == 0:
                    quantized_weight, max_abs, shape = quantizer.quantize_block(res)
                    R = torch.randn((reduced_rank, in_feature), device=device)
                    L = torch.zeros((out_feature, reduced_rank), device=device)
                linear_loras.initial_backbone(quantized_weight, max_abs)
                linear_loras.initial_lora(R, L)

            delattr(module, attr_str)
            torch.cuda.empty_cache()
            setattr(module, attr_str, linear_loras)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            substitute_layer_weights_iter_quant(immediate_child_module,
                                                allow_name=allow_name,
                                                block_name=block_name,
                                                reduced_rank=reduced_rank,
                                                num_bits=num_bits,
                                                num_iter=num_iter,
                                                load=load,
                                                enable_lora=enable_lora)


def replace_lora_only(module,
                      allow_name=None,
                      block_name=None,
                      reduced_rank=32,
                      load_mode=False,
                      ):
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h',
                      'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']

    allow_module = [nn.Linear, Linear]

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if any(isinstance(target_attr, module) for module in allow_module) and any(an in attr_str for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr)
            linear_loras = LinearLR(target_attr.in_features,
                                    target_attr.out_features,
                                    reduced_rank,
                                    bias=False if target_attr.bias is None else True)

            if not load_mode:
                output = low_rank_decomposition(target_attr.weight, reduced_rank=reduced_rank)
                L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']

                linear_loras.initial_lora(R, L)

            delattr(module, attr_str)
            torch.cuda.empty_cache()
            setattr(module, attr_str, linear_loras)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            replace_lora_only(immediate_child_module,
                              allow_name=allow_name,
                              block_name=block_name,
                              reduced_rank=reduced_rank,
                              load_mode=load_mode,
                              )


def qlora_init(weight, num_bits, reduced_rank, num_iter, method='normal'):
    out_feature, in_feature = weight.size()
    device = weight.device

    quantizer = NFQuantizer(num_bits=num_bits, device=device, method=method)
    if method == 'normal':
        block_size = 64
    else:
        # TODO: change
        block_size = 1024
    res = weight.clone()
    for i in range(num_iter):
        # Quantization
        quantized_weight, max_abs, shape = quantizer.quantize_block(res, block_size=block_size, method=method)
        dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape, block_size=block_size)
        res = weight - dequantized_weight

        # Decompose the residual by SVD
        output = low_rank_decomposition(res, reduced_rank=reduced_rank)
        L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']
        res = weight - torch.mm(L, R)

    if num_iter == 0:
        quantized_weight, max_abs, shape = quantizer.quantize_block(res, block_size=block_size)
        dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape, block_size=block_size)
        R = torch.randn((reduced_rank, in_feature), device=device)
        L = torch.zeros((out_feature, reduced_rank), device=device)

    lora_A, lora_B = R, L

    quant_error = (weight - dequantized_weight).power(2).mean().sqrt()
    print(quant_error)

    return dequantized_weight, lora_A, lora_B


def substitute_layer_weights(module,
                             allow_name=None,
                             block_name=None,
                             reduced_rank=8,
                             num_bits=4,
                             num_iter=5,
                             method='normal'
                             ):
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h',
                      'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']
    assert (num_bits == 8 or num_bits == 4 or num_bits == 2) and num_iter >= 0

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if any(an in attr_str for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr)

            weight, lora_A, lora_B = qlora_init(weight=target_attr.weight,
                                                num_bits=num_bits,
                                                num_iter=num_iter,
                                                reduced_rank=reduced_rank,
                                                method=method)
            target_attr.weight.data = weight
            target_attr.lora_A['default'].weight.data = lora_A
            target_attr.lora_B['default'].weight.data = lora_B

            torch.cuda.empty_cache()

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            substitute_layer_weights(immediate_child_module,
                                     allow_name=allow_name,
                                     block_name=block_name,
                                     reduced_rank=reduced_rank,
                                     num_bits=num_bits,
                                     num_iter=num_iter,
                                     method=method)


class DQLinearLoRA(nn.Module):
    def __init__(self, in_features, out_features, reduced_rank, num_bits=4, num_iter=1, num_step=4, lr=1e-3, bias=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.reduced_rank = reduced_rank
        self.num_bits = num_bits
        self.num_iter = num_iter
        self.num_step = num_step
        self.block_size = 64 if (num_bits == 4 or num_bits == 8) else 32
        self.lr = lr

        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.bias = bias
        self.lora_A = nn.Parameter(torch.randn(reduced_rank, in_features), requires_grad=False)
        self.lora_B = nn.Parameter(torch.randn(out_features, reduced_rank), requires_grad=False)

        self.lookup_table = self.create_normal_map(num_bits=num_bits)
        self.lookup_table = nn.Parameter(self.lookup_table, requires_grad=False)
        self.max_val = self.init_max_val()

        self.optimizer = optim.AdamW([self.max_val], lr=lr)
        self.loss_fn = nn.MSELoss()

    def initialize(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def init_max_val(self):
        weight_flatten = self.weight.flatten()  # (M*N, )
        weight_block = weight_flatten.reshape(-1, self.block_size)  # (L, B), L = M * N / B
        max_val = weight_block.abs().max(dim=-1)[0]  # (L, 1)
        max_val = nn.Parameter(max_val)
        return max_val

    def quantize(self, weight):
        assert len(weight.shape) == 2 and weight.shape[0] * weight.shape[1] % self.block_size == 0
        M, N = weight.shape

        # Quantization
        weight_flatten = weight.flatten()  # (M*N, )
        weight_block = weight_flatten.reshape(-1, self.block_size)  # (L, B), L = M * N / B

        weight_max = self.max_val.unsqueeze(-1)
        weight_divabs = weight_block / weight_max  # (L, B)
        weight_divabs = weight_divabs.unsqueeze(-1)  # (L, B, 1)
        L_reshaped = self.lookup_table.reshape(1, -1)  # (1, 2**K)

        abs_diff = torch.abs(weight_divabs - L_reshaped)  # (L, B, 2**K)
        qweight = torch.argmin(abs_diff, dim=-1)  # (L, B)

        qweight = self.lookup_table[qweight].squeeze()

        qweight = qweight * weight_max
        qweight = qweight.reshape(M, N)

        return qweight

    def forward_qlora(self, x, qweight):
        lora = (x @ self.lora_A.T) @ self.lora_B.T
        net = x @ qweight.T
        return lora + net + self.bias if self.bias is not None else lora + net

    def forward_backbone(self, x, qweight):
        net = x @ qweight.T
        return net + self.bias if self.bias is not None else net

    def forward(self, x):
        torch.cuda.empty_cache()
        y_gold = x @ self.weight.T + self.bias if self.bias is not None else x @ self.weight.T

        res = self.weight.clone()
        for j in range(self.num_iter):
            for i in range(self.num_step):
                # Quantization
                qweight = self.quantize(res)
                y_pred = self.forward_backbone(x, qweight)
                loss = self.loss_fn(y_pred, y_gold)
                loss.backward(retain_graph=True)
                print(f"Before updating lookup table: {self.max_val}")
                print(f"loss: {loss}")
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f"After updating lookup table: {self.max_val}")

            res = self.weight - qweight
            print(f"L2 norm for quantization: {res.pow(2).mean().sqrt().item()}")

            # Decompose the residual by SVD
            output = low_rank_decomposition(res, reduced_rank=self.reduced_rank)
            L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']
            res = self.weight - torch.mm(L, R)
            print(f"L2 norm for SVD: {res.pow(2).mean().sqrt().item()}")

        if self.num_iter == 0:
            qweight = self.quantize(self.weight)
            R = torch.randn((self.reduced_rank, self.in_features))
            L = torch.zeros((self.out_features, self.reduced_rank))

        self.lora_A.data = R
        self.lora_B.data = L
        self.weight.data = qweight
        torch.cuda.empty_cache()

        return y_gold

    @staticmethod
    def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
        variations = 2 ** num_bits

        if symmetric:
            print("symmetric nf4")
            v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
            values = []
            for index in range(len(v) - 1):
                values.append(0.5 * v[index] + 0.5 * v[index + 1])
            v = values
        else:
            # one more positive value, this is an asymmetric type
            print("asymmetric nf4")
            v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
            # print(torch.linspace(offset, 0.5, 9)[:-1])
            # print(v1)
            v2 = [0]
            # v2 = [0]*(256-15) ## we have 15 non-zero values in this data type
            v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
            # print(torch.linspace(offset, 0.5, 8)[:-1])
            # print(v3)
            v = v1 + v2 + v3

        values = torch.Tensor(v)
        values = values.sort().values
        values /= values.max()
        # print(values)
        return values


class QLinear(nn.Module):
    def __init__(self, num_bits, block_size, bias=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bits = num_bits
        self.block_size = block_size

        self.lookup_table = create_normal_map(num_bits=num_bits, symmetric=True)
        self.lookup_table = nn.Parameter(self.lookup_table, requires_grad=False)
        self.max_val = None

    def init_max_val(self, weight):
        weight_flatten = weight.flatten()  # (M*N, )
        weight_block = weight_flatten.reshape(-1, self.block_size)  # (L, B), L = M * N / B
        max_val = weight_block.abs().max(dim=-1)[0]  # (L, 1)
        self.max_val = nn.Parameter(max_val, requires_grad=True)

    def quantize(self, weight):
        assert len(weight.shape) == 2 and weight.shape[0] * weight.shape[1] % self.block_size == 0
        M, N = weight.shape

        # Quantization
        weight_flatten = weight.flatten()  # (M*N, )
        weight_block = weight_flatten.reshape(-1, self.block_size)  # (L, B), L = M * N / B

        weight_max = self.max_val.unsqueeze(-1)
        weight_divabs = weight_block / weight_max  # (L, B)
        weight_divabs = weight_divabs.unsqueeze(-1)  # (L, B, 1)
        L_reshaped = self.lookup_table.reshape(1, -1)  # (1, 2**K)

        abs_diff = torch.abs(weight_divabs - L_reshaped)  # (L, B, 2**K)
        qweight = torch.argmin(abs_diff, dim=-1)  # (L, B)

        qweight = self.lookup_table[qweight].squeeze()

        qweight = qweight * weight_max
        qweight = qweight.reshape(M, N)

        return qweight

    def forward(self, x, weight, bias=None):
        qweight = self.quantize(weight)
        net = x @ qweight.T
        y = net + bias if bias is not None else net
        return y, qweight


class LearnableQuantizer:
    def __init__(self, in_features, out_features, reduced_rank, num_bits=4, num_iter=1, num_step=4, lr=1e-3, bias=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.reduced_rank = reduced_rank

        self.qlora = QLinear(in_features, out_features, bias, num_bits)
        self.num_iter = num_iter
        self.num_step = num_step
        self.lr = lr

        self.lora_A = nn.Parameter(torch.randn(reduced_rank, in_features), requires_grad=False)
        self.lora_B = nn.Parameter(torch.randn(out_features, reduced_rank), requires_grad=False)

        self.optimizer = optim.AdamW([self.qlora.max_val], lr=lr)
        self.loss_fn = nn.MSELoss()

    def init_weight(self, weight, bias):
        self.qlora.init_weight(weight, bias)

    def update(self, x, y):
        device = x.device
        torch.cuda.empty_cache()

        weight = self.qlora.weight.clone()
        res = self.qlora.weight.clone()
        for j in range(self.num_iter):
            for i in range(self.num_step):
                # Quantization
                self.qlora.weight = res
                y_pred = self.qlora(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                print(
                    f"Before: max: {self.qlora.max_val.max()} | min: {self.qlora.max_val.min()} | mean: {self.qlora.max_val.mean()}")
                print(f"loss: {loss}")
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(
                    f"After: max: {self.qlora.max_val.max()} | min: {self.qlora.max_val.min()} | mean: {self.qlora.max_val.mean()}")

            res = weight - self.qlora.qweight
            print(f"L2 norm for quantization: {res.pow(2).mean().sqrt().item()}")

            # Decompose the residual by SVD
            output = low_rank_decomposition(res, reduced_rank=self.reduced_rank)
            L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']
            res = weight - torch.mm(L, R)
            print(f"L2 norm for SVD: {res.pow(2).mean().sqrt().item()}")

        # if self.num_iter == 0:
        #     qweight = self.quantize(self.weight)
        #     R = torch.randn((self.reduced_rank, self.in_features))
        #     L = torch.zeros((self.out_features, self.reduced_rank))

        self.lora_A.data = R
        self.lora_B.data = L
        torch.cuda.empty_cache()

    def get_weights(self):
        return self.qlora.qweight, self.lora_A, self.lora_B

    @staticmethod
    def create_normal_map(offset=0.9677083, symmetric=False, num_bits=2):
        variations = 2 ** num_bits

        if symmetric:
            print("symmetric nf4")
            v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
            values = []
            for index in range(len(v) - 1):
                values.append(0.5 * v[index] + 0.5 * v[index + 1])
            v = values
        else:
            # one more positive value, this is an asymmetric type
            print("asymmetric nf4")
            v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
            # print(torch.linspace(offset, 0.5, 9)[:-1])
            # print(v1)
            v2 = [0]
            # v2 = [0]*(256-15) ## we have 15 non-zero values in this data type
            v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
            # print(torch.linspace(offset, 0.5, 8)[:-1])
            # print(v3)
            v = v1 + v2 + v3

        values = torch.Tensor(v)
        values = values.sort().values
        values /= values.max()
        # print(values)
        return values


def replace_linear(module,
                   allow_name=None,
                   block_name=None,
                   reduced_rank=32,
                   num_bits=4,
                   num_iter=5,
                   num_step=3,
                   lr=1e-3,
                   ):
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h',
                      'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']
    assert (num_bits == 8 or num_bits == 4 or num_bits == 2) and num_iter >= 0 and num_step >= 0

    allow_module = [nn.Linear, Linear]

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if any(isinstance(target_attr, module) for module in allow_module) and any(an in attr_str for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr)
            in_features = target_attr.in_features
            out_features = target_attr.out_features
            bias = target_attr.bias
            weight = target_attr.weight

            linear_dqlora = DQLinearLoRA(in_features=in_features,
                                         out_features=out_features,
                                         reduced_rank=reduced_rank,
                                         num_bits=num_bits,
                                         num_iter=num_iter,
                                         num_step=num_step,
                                         lr=lr,
                                         bias=bias)
            linear_dqlora.initialize(weight, bias)

            delattr(module, attr_str)
            torch.cuda.empty_cache()
            setattr(module, attr_str, linear_dqlora)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            replace_linear(immediate_child_module,
                           allow_name=allow_name,
                           block_name=block_name,
                           reduced_rank=reduced_rank,
                           num_bits=num_bits,
                           num_iter=num_iter,
                           num_step=num_step,
                           lr=lr,
                           )


def replace_linear_with_observer(module,
                                 allow_name=None,
                                 block_name=None,
                                 ):
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h',
                      'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2',
                      'o_proj', 'gate_proj', 'down_proj']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']

    allow_module = [nn.Linear, Linear]

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if any(isinstance(target_attr, module) for module in allow_module) and any(an in attr_str for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr)

            linear_observer = LinearObserver(in_features=target_attr.in_features,
                                             out_features=target_attr.out_features)

            delattr(module, attr_str)
            torch.cuda.empty_cache()
            setattr(module, attr_str, linear_observer)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            replace_linear_with_observer(immediate_child_module,
                                         allow_name=allow_name,
                                         block_name=block_name,
                                         )


def replace_linear_with_learnable_quantization(module,
                                               allow_name=None,
                                               block_name=None,
                                               reduced_rank=32,
                                               num_bits=4,
                                               num_iter=5,
                                               num_step=3,
                                               lr=1e-3,
                                               ):
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h',
                      'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']
    assert (num_bits == 8 or num_bits == 4 or num_bits == 2) and num_iter >= 0 and num_step >= 0

    allow_module = [LinearObserver]

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if any(isinstance(target_attr, module) for module in allow_module) and any(an in attr_str for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr)
            in_features = target_attr.in_features
            out_features = target_attr.out_features
            bias = target_attr.bias
            weight = target_attr.weight
            input = target_attr.input
            output = target_attr.output
            device = weight.device

            block_size = 64 if (num_bits == 4 or num_bits == 8) else 32
            qlinear = QLinear(num_bits, block_size)
            qlinear.init_max_val(weight)

            # move to device
            qlinear = qlinear.to(device)
            input, output = input.to(device), output.to(device)

            loss_l2 = nn.MSELoss()
            loss_kl = nn.KLDivLoss()
            optimizer = optim.AdamW([qlinear.max_val], lr=lr)

            res = weight.clone()
            for j in range(num_iter):
                for i in range(num_step):
                    # Quantization
                    y_pred, qweight = qlinear(input, res)
                    loss = loss_l2(y_pred, output)
                    loss.backward()
                    print(
                        f"Before: max: {qlinear.max_val.max()} | min: {qlinear.max_val.min()} | mean: {qlinear.max_val.mean()}")
                    print(f"loss: {loss}")
                    optimizer.step()
                    optimizer.zero_grad()
                    print(
                        f"After: max: {qlinear.max_val.max()} | min: {qlinear.max_val.min()} | mean: {qlinear.max_val.mean()}")

                res = weight - qweight
                print(f"L2 norm for quantization: {res.pow(2).mean().sqrt().item()}")

                # Decompose the residual by SVD
                output = low_rank_decomposition(res, reduced_rank=reduced_rank)
                L, R, reduced_rank = output['L'], output['R'], output['reduced_rank']
                res = weight - torch.mm(L, R)
                print(f"L2 norm for SVD: {res.pow(2).mean().sqrt().item()}")

            linear_lora = LinearLoRA(in_features, out_features, reduced_rank)
            linear_lora.initialize(qweight, L, R, bias)
            linear_lora.to('cpu')  # save memory

            delattr(module, attr_str)
            torch.cuda.empty_cache()
            # os.system('nvidia-smi')
            setattr(module, attr_str, linear_lora)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            replace_linear_with_learnable_quantization(immediate_child_module,
                                                       allow_name=allow_name,
                                                       block_name=block_name,
                                                       reduced_rank=reduced_rank,
                                                       num_bits=num_bits,
                                                       num_iter=num_iter,
                                                       num_step=num_step,
                                                       lr=lr,
                                                       )


def replace_linear_with_shell(module,
                              allow_name=None,
                              block_name=None,
                              reduced_rank=32,
                              ):
    # Default allow name and block name lists
    if allow_name is None:
        allow_name = ['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h',
                      'q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    if block_name is None:
        block_name = ['pooler', 'classifier', 'LayerNorm', 'embeddings']

    allow_module = [LinearObserver, nn.Linear]

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if any(isinstance(target_attr, module) for module in allow_module) and any(an in attr_str for an in allow_name):
            print("====================================================")
            print(attr_str, target_attr)
            in_features = target_attr.in_features
            out_features = target_attr.out_features
            bias = target_attr.bias
            weight = target_attr.weight
            L = nn.Parameter(torch.randn(reduced_rank, in_features), requires_grad=False)
            R = nn.Parameter(torch.randn(out_features, reduced_rank), requires_grad=False)

            linear_lora = LinearLoRA(in_features, out_features, reduced_rank)

            linear_lora.initialize(weight, L, R, bias)

            delattr(module, attr_str)
            torch.cuda.empty_cache()
            # os.system('nvidia-smi')
            setattr(module, attr_str, linear_lora)

    for name, immediate_child_module in module.named_children():
        # do not continue to iterate when the module's name is in the block_name
        if not any(name in bn for bn in block_name):
            replace_linear_with_shell(immediate_child_module,
                                      allow_name=allow_name,
                                      block_name=block_name,
                                      reduced_rank=reduced_rank,
                                      )


def one_pass(model, dataloader, args):
    allow_name = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    replace_linear(model,
                   allow_name=allow_name,
                   reduced_rank=args.reduced_rank,
                   num_bits=args.num_bits,
                   num_iter=args.num_iter,
                   num_step=args.num_step,
                   lr=args.lr)
    for i, batch in enumerate(dataloader):
        output = model(**batch)

    torch.save(model.state_dict(), args.ckpt_path)


class LinearLoRA(nn.Linear):
    def __init__(self, in_features: int, out_features: int, reduced_rank: int):
        super().__init__(in_features, out_features)
        self.reduced_rank = reduced_rank
        self.lora_A = nn.Parameter(torch.randn(reduced_rank, in_features), requires_grad=True)
        self.lora_B = nn.Parameter(torch.randn(out_features, reduced_rank), requires_grad=True)

    def initialize(self, weight, lora_A, lora_B, bias=None):
        self.weight.data = weight
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        if self.bias is not None:
            self.bias.data = bias

    def forward(self, input: Tensor) -> Tensor:
        dense = F.linear(input, self.weight, self.bias)
        lora = (input @ self.lora_A.T) @ self.lora_B.T
        return dense + lora


class LinearObserver(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.output = None
        self.input = None

    def forward(self, input: Tensor) -> Tensor:
        self.input = nn.Parameter(input.clone(), requires_grad=False)
        y = F.linear(input, self.weight, self.bias)
        self.output = nn.Parameter(y.clone(), requires_grad=False)
        return y
