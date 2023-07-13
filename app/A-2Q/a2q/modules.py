import torch
from torch import nn as nn
from torch.nn import functional as F

from a2q.functions import (STELocalGradientGradBasedIndexingQuantizationFunction, LocalGradientSignAndGradBasedQuantizationFunction, STELocalGradientSignBasedQuantizationFunction,
                           STELocalGradientGradBasedQuantizationFunction, LocalGradientGradBasedQuantizationFunction)
from a2q.utils.get_scale_index import get_deg_index, get_scale_index, get_scale_index_naive, get_scale_index_uniform


class FeatureQuantizationV1(nn.Module):
    """
    feature uniform quantization.
    """

    def __init__(self, input_dim, bit, gama_init=0.001, gama_std=0.001, is_quantizing_features=True):
        super(FeatureQuantizationV1, self).__init__()
        self.is_quantizing_features = is_quantizing_features
        if is_quantizing_features:
            self.bit = bit
            self.feature_quantization_function = STELocalGradientSignBasedQuantizationFunction
            self.gama = torch.nn.Parameter(torch.Tensor(input_dim, 1))
            torch.nn.init.normal_(self.gama, mean=gama_init, std=gama_std)
            self.gama = torch.nn.Parameter(self.gama.abs())
            self.bit = torch.nn.Parameter(torch.Tensor(input_dim, 1))
            torch.nn.init.constant_(self.bit, bit)

    def forward(self, x):
        if self.is_quantizing_features:
            x = self.feature_quantization_function.apply(x, self.gama, self.bit)
        return x


class FeatureQuantizationV2(nn.Module):
    def __init__(self, input_dim, bit, gama_init=0.001, gama_std=0.001, uniform=False, is_naive=False, is_quantizing_features=True, init="norm"):
        super(FeatureQuantizationV2, self).__init__()
        self.bit = bit
        self.quant_fea = is_quantizing_features
        self.uniform = uniform
        self.is_naive = is_naive
        self.input_dim = input_dim

        self.quant_fea_func = STELocalGradientGradBasedIndexingQuantizationFunction
        self.quant_fea_func_no_index = STELocalGradientGradBasedIndexingQuantizationFunction
        self.gama = torch.nn.Parameter(torch.Tensor(input_dim, 1))
        if init == "norm":
            torch.nn.init.normal_(self.gama, mean=gama_init, std=gama_std)
            self.gama = torch.nn.Parameter(self.gama.abs())
        else:
            torch.nn.init.uniform_(self.gama, 0, 1)

        self.bit = torch.nn.Parameter(torch.Tensor(torch.Tensor(input_dim, 1)))
        _init_bit = bit
        torch.nn.init.constant_(self.bit, _init_bit)

    """
    Features uniform quantization.
    """

    def forward(self, fea, edge_index):
        if edge_index is not None and not self.uniform and not self.is_naive:
            deg_index = get_deg_index(fea=fea, edge_index=edge_index)
            scale_index = get_scale_index(fea=fea, deg_index=deg_index, scale=self.gama, bit=self.bit)
            unique_index = torch.unique(scale_index)
            bit_sum = fea.size(1) * self.bit[unique_index].sum() / 8.0 / 1024.0
            fea_q = self.quant_fea_func.apply(fea, self.gama, self.bit, scale_index)
        elif edge_index is not None and self.uniform and not self.is_naive:
            scale_index = get_scale_index_uniform(fea=fea, scale=self.gama, bit=self.bit)
            unique_index = torch.unique(scale_index)
            bit_sum = fea.size(1) * self.bit[unique_index].sum() / 8.0 / 1024.0
            fea_q = self.quant_fea_func.apply(fea, self.gama, self.bit, scale_index)
        elif edge_index is not None and self.is_naive:
            scale_index = get_scale_index_naive(fea, edge_index, self.input_dim)
            unique_index = torch.unique(scale_index)
            bit_sum = fea.size(1) * self.bit[unique_index].sum() / 8.0 / 1024.0
            fea_q = self.quant_fea_func.apply(fea, self.gama, self.bit, scale_index)
        else:
            fea_q = self.quant_fea_func_no_index.apply(fea, self.gama, self.bit)
            bit_sum = fea_q.new_zeros(1)
        if not self.quant_fea:
            fea_q = fea
            bit_sum = 0
        return fea_q, bit_sum


class FeatureWeightQuantizationV1(nn.Module):
    """
    xw uniform quantization.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            bit,
            alpha_init=0.1,
            alpha_std=0.1,
    ):
        super(FeatureWeightQuantizationV1, self).__init__()
        self.bit = bit
        self.quant_weight_func = STELocalGradientSignBasedQuantizationFunction
        _alpha = torch.Tensor(out_channels, 1)
        self.alpha = torch.nn.Parameter(_alpha)
        torch.nn.init.normal_(self.alpha, mean=alpha_init, std=alpha_std)
        self.alpha = torch.nn.Parameter(self.alpha.abs())
        _init_bit = bit
        self.bit = torch.nn.Parameter(torch.Tensor(out_channels, 1))
        torch.nn.init.constant_(self.bit, _init_bit)

    def forward(self, fea):
        # quantization
        fea_q = self.quant_weight_func.apply(fea, self.alpha, self.bit)
        return fea_q


class FeatureWeightQuantizationV2(nn.Module):
    """
    The result of XW uniform quantization.
    """

    def __init__(self, in_channels, out_channels, bit, alpha_init=0.01, alpha_std=0.01):
        super(FeatureWeightQuantizationV2, self).__init__()
        self.bit = bit
        self.quant_weight_func = LocalGradientSignAndGradBasedQuantizationFunction
        _alpha = torch.Tensor(out_channels, 1)
        self.alpha = torch.nn.Parameter(_alpha)
        torch.nn.init.normal_(self.alpha, mean=alpha_init, std=alpha_std)
        self.alpha = torch.nn.Parameter(self.alpha.abs())
        _init_bit = bit
        self.bit = torch.nn.Parameter(torch.Tensor(out_channels, 1))
        torch.nn.init.constant_(self.bit, _init_bit)

    def forward(self, fea):
        # quantization
        fea_q = self.quant_weight_func.apply(fea, self.alpha, self.bit)
        return fea_q


class WeightQuantizationV1(nn.Module):
    """
    weight uniform quantization.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            bit,
            alpha_init=0.1,
            alpha_std=0.1,
    ):
        super(WeightQuantizationV1, self).__init__()
        self.bit = bit
        self.quant_weight_func = STELocalGradientGradBasedQuantizationFunction
        # initialize the step size by normal distribution
        _alpha = torch.Tensor(out_channels, 1)
        self.alpha = torch.nn.Parameter(_alpha)
        torch.nn.init.normal_(self.alpha, mean=alpha_init, std=alpha_std)
        self.alpha = torch.nn.Parameter(self.alpha.abs())
        _init_bit = bit
        self.bit = torch.nn.Parameter(torch.Tensor(out_channels, 1))
        torch.nn.init.constant_(self.bit, _init_bit)

    def forward(self, weight):
        # quantization
        weight_q = self.quant_weight_func.apply(weight, self.alpha, self.bit)
        return weight_q


class WeightQuantizationV2(nn.Module):
    """
    weight uniform quantization.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            bit,
            alpha_init=0.01,
            alpha_std=0.01,
    ):
        super(WeightQuantizationV2, self).__init__()
        self.bit = bit
        # quantizer

        self.quant_weight_func = LocalGradientGradBasedQuantizationFunction
        _alpha = torch.Tensor(out_channels, 1)
        self.alpha = torch.nn.Parameter(_alpha)
        torch.nn.init.normal_(self.alpha, mean=alpha_init, std=alpha_std)
        self.alpha = torch.nn.Parameter(self.alpha.abs())
        _init_bit = bit
        self.bit = torch.nn.Parameter(torch.Tensor(out_channels, 1))
        torch.nn.init.constant_(self.bit, _init_bit)

    def forward(self, weight):
        # quantization
        weight_q = self.quant_weight_func.apply(weight, self.alpha, self.bit)
        return weight_q


class LinearQuantizationV1(nn.Linear):
    """
    Quantized linear layers.
    """

    def __init__(
            self,
            in_features,
            out_features,
            num_nodes,
            bit,
            gama_init=1e-3,
            bias=True,
            all_positive=False,
            para_dict={"alpha_init": 0.01, "alpha_std": 0.02, "gama_init": 0.1, "gama_std": 0.2},
            quant_fea=True,
    ):
        super(LinearQuantizationV1, self).__init__(in_features, out_features, bias)
        self.bit = bit
        # if the value is all positive then we use the unsign quantization
        if all_positive:
            bit_fea = bit + 1
        else:
            bit_fea = bit
        alpha_init = para_dict["alpha_init"]
        gama_init = para_dict["gama_init"]
        alpha_std = para_dict["alpha_std"]
        gama_std = para_dict["gama_std"]
        # weight quantization module
        self.weight_quant = WeightQuantizationV1(
            in_features, out_features, bit, alpha_init=alpha_init, alpha_std=alpha_std
        )
        # features quantization module
        self.fea_quant = FeatureQuantizationV1(
            num_nodes,
            bit_fea,
            gama_init=gama_init,
            gama_std=gama_std,
        )
        if not quant_fea:
            # Do not quantize the feature when the value is 0 or 1
            self.fea_quant = nn.Identity()
        # glorot(self.weight)

    def forward(self, x):
        # weight quantization
        weight_q = self.weight_quant(self.weight)
        # weight_q  = self.weight
        fea_q = self.fea_quant(x)
        # fea_q = x
        return F.linear(fea_q, weight_q, self.bias)


class LinearQuantizationV2(nn.Linear):
    """
    Quantized linear layers.
    """

    def __init__(
            self,
            in_features,
            out_features,
            num_nodes,
            bit,
            gama_init=1e-3,
            bias=True,
            all_positive=False,
            para_dict={"alpha_init": 0.01, "alpha_std": 0.02, "gama_init": 0.1, "gama_std": 0.2},
            quant_fea=True,
            uniform=False,
            is_naive=False,
            init="norm",
    ):
        super(LinearQuantizationV2, self).__init__(in_features, out_features, bias)
        self.bit = bit
        if all_positive:
            bit_fea = bit + 1
        else:
            bit_fea = bit
        alpha_init = para_dict["alpha_init"]
        gama_init = para_dict["gama_init"]
        alpha_std = para_dict["alpha_std"]
        gama_std = para_dict["gama_std"]
        # weight quantization module
        self.weight_quant = WeightQuantizationV2(
            in_features,
            out_features,
            bit,
            alpha_init=alpha_init,
            alpha_std=alpha_std,
        )
        self.fea_quant = FeatureQuantizationV2(
            num_nodes, bit_fea, gama_init=gama_init, gama_std=gama_std, uniform=uniform, is_naive=is_naive, init=init
        )
        if not quant_fea:
            self.fea_quant = nn.Identity()

    def forward(self, x, edge_index, bit_sum):
        # weight quantization
        weight_q = self.weight_quant(self.weight)
        # weight_q = self.weight
        if isinstance(self.fea_quant, nn.Identity):
            fea_q = self.fea_quant(x)
            bit_sum_layer = 0
        else:
            fea_q, bit_sum_layer = self.fea_quant(x, edge_index)
        bit_sum += bit_sum_layer
        # fea_q = x
        return F.linear(fea_q, weight_q, self.bias), edge_index, bit_sum
