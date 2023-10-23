import torch
def weight_quant_fn(weight,  num_bits, quant_method, num_std = 2):
    if quant_method=="normal_float":
        return quant_nf4_block(weight,num_bits=num_bits)
    elif quant_method == "uniform":
        mean, std = weight.mean(), weight.std()
        clip_val = (mean - num_std * std, mean + num_std * std)
        clip_val = torch.tensor(list(clip_val))
        return quant_uniform(weight,num_bits,clip_val)
    else:
        raise ValueError("")


#The default choice of normal float quantization is asymmetric. We follow the default setting of qlora that we use asymmetric normal float with block size 64
from scipy.stats import norm
def create_normal_map(offset=0.9677083, symmetric=False, num_bits = 4):
    variations = 2**num_bits
    print("Doing normal float quantization")
    if symmetric == True:
        v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
        values = []
        for index in range(len(v) - 1):
            values.append(0.5 * v[index] + 0.5 * v[index + 1])
        v = values
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
        v2 = [0]
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
        v = v1 + v2 + v3


    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()
    return values
    # assert values.

def quantize_tensor(X, L):
    X_expanded = X.unsqueeze(-1)

    # Reshape L to have the same number of dimensions as X_expanded
    L_reshaped = torch.tensor(L).reshape(1, -1)

    # Calculate the absolute difference between X_expanded and L_reshaped
    abs_diff = torch.abs(X_expanded - L_reshaped)

    # Find the index of the minimum absolute difference for each element
    min_index = torch.argmin(abs_diff, dim=-1)
    # print(min_index)
    return L[min_index]

def quant_nf4(weight,num_bits=2):
    print(f"using normal float {num_bits}bits")
    max_abs = torch.abs(weight).max()
    weight_divabs = weight/max_abs
    data_format = create_normal_map(num_bits=num_bits)
    weights_divabs = quantize_tensor(weight_divabs, data_format)
    return weights_divabs*max_abs

def quant_nf4_block(weight, block_size=64, num_bits=2):
    def quant_nf4(weight, num_bits=num_bits):
        max_abs = torch.abs(weight).max()
        weight_divabs = weight / max_abs
        data_format = create_normal_map(num_bits=num_bits)
        weights_divabs = quantize_tensor(weight_divabs, data_format)
        return weights_divabs * max_abs
    print(f"nf4 quantize by block with block size {block_size} using {num_bits}bits")
    weight_resize = weight.resize(weight.shape[0]*weight.shape[1]//block_size,block_size)
    quant_block = torch.vmap(quant_nf4, out_dims=0)
    return quant_block(weight_resize).view(weight.shape[0],weight.shape[1])


def quant_uniform(input, num_bits=2, clip_val = None):
    if clip_val!=None:
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
    print(f"uniform quant with {num_bits}bits")
    alpha = (input.max() - input.min()).detach()
    beta = input.min().detach()
    input_normalized = (input - beta) / (alpha + 1e-8)  # map to 0 to 1
    s = (2 ** num_bits - 1)
    quant_input = torch.round(input_normalized * s).div(s)  # map to int between 0 and s(2**num_bits-1)
    output = quant_input * (alpha + 1e-8) + beta  #
    return output
