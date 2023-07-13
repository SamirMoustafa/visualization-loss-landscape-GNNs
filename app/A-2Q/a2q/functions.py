import math

import torch
from torch.autograd import Function
from torch_scatter import scatter_add


class STELocalGradientSignBasedQuantizationFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha, bit):
        bit = bit.abs().round()
        q_max = 2 ** (bit - 1) - 1
        x_div = torch.clamp(x / alpha, -q_max, q_max)
        x_q = torch.round(x_div)
        x_max = q_max * alpha
        ctx.save_for_backward(x, x_div, x_q, x_max, alpha, torch.sign(alpha))
        ctx.q_max = q_max
        return x_q * alpha

    @staticmethod
    def backward(ctx, grad_output):
        x0, x, x_q, x_max, alpha, alpha_sign = ctx.saved_variables
        grad_x = grad_output
        grad_x[x < -ctx.q_max] = 0
        grad_x[x > ctx.q_max] = 0
        x_q_sign = torch.sign(x_q.mul(alpha) - x0)
        i = (x0.abs() <= x_max).float()
        grad_alpha = (x_q_sign * ((x_q - x) * i + (1 - i) * ctx.q_max * torch.sign(x))).mean(1).reshape(-1, 1)
        grad_b = (x_q_sign * (1 - i) * torch.sign(x0) * (ctx.q_max + 1) * math.log(2) * alpha).mean(1).reshape(-1, 1)
        return grad_x, grad_alpha, grad_b


class STELocalGradientGradBasedQuantizationFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha, bit):
        bit = bit.abs().round()
        q_max = 2 ** (bit - 1) - 1
        x_div = torch.clamp(x / alpha, -q_max, q_max)
        x_q = torch.round(x_div)
        x_max = q_max * alpha
        ctx.save_for_backward(x, x_div, x_q, x_max, alpha, torch.sign(alpha))
        ctx.q_max = q_max
        return x_q * alpha

    @staticmethod
    def backward(ctx, grad_output):
        x0, x, x_q, x_max, alpha, alpha_sign = ctx.saved_variables
        grad_x = grad_output
        grad_x[x < -ctx.q_max] = 0
        grad_x[x > ctx.q_max] = 0
        i = (x0.abs() <= x_max).float()
        grad_alpha = (grad_output * ((x_q - x) * i + (1 - i) * ctx.q_max * torch.sign(x0))).sum(1).reshape(-1, 1)
        grad_b = (grad_output * (1 - i) * torch.sign(x) * (ctx.q_max + 1) * math.log(2) * alpha).sum(1).reshape(-1, 1)
        return grad_x, grad_alpha, grad_b


class LocalGradientGradBasedQuantizationFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha, bit):
        bit = bit.abs().round()
        q_max = 2 ** (bit - 1) - 1
        x_div = torch.clamp(x / alpha, -q_max, q_max)
        x_q = torch.round(x_div)
        x_max = q_max * alpha
        ctx.save_for_backward(x, x_div, x_q, x_max, alpha, torch.sign(alpha))
        ctx.q_max = q_max
        return x_q * alpha

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        x0, x, x_q, x_max, alpha, alpha_sign = ctx.saved_variables
        i = (x0.abs() <= x_max).float()
        grad_alpha = (grad_output * ((x_q - x) * i + (1 - i) * ctx.q_max * torch.sign(x0))).sum(1).reshape(-1, 1)
        grad_b = (grad_output * (1 - i) * torch.sign(x) * (ctx.q_max + 1) * math.log(2) * alpha).sum(1).reshape(-1, 1)
        return grad_x, grad_alpha, grad_b


class LocalGradientSignAndGradBasedQuantizationFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha, bit):
        bit = bit.abs().round()
        q_max = 2 ** (bit - 1) - 1
        x_div = torch.clamp(x / alpha, -q_max, q_max)
        x_q = torch.round(x_div)
        x_max = q_max * alpha
        ctx.save_for_backward(x, x_div, x_q, x_max, alpha, torch.sign(alpha))
        ctx.q_max = q_max
        return x_q * alpha

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        x0, x, x_q, x_max, alpha, alpha_sign = ctx.saved_variables
        i = (x0.abs() <= x_max).float()
        x_q_sign = torch.sign(x_q.mul(alpha) - x0)
        grad_alpha = (x_q_sign * ((x_q - x) * i + (1 - i) * ctx.q_max * torch.sign(x0))).mean(1).reshape(-1, 1)
        grad_b = (grad_x * (1 - i) * torch.sign(x) * (ctx.q_max + 1) * math.log(2) * alpha).sum(1).reshape(-1, 1)
        return grad_x, grad_alpha, grad_b


class STELocalGradientGradBasedIndexingQuantizationFunction(Function):
    @staticmethod
    def forward(ctx, x, gama, bit, scale_index):
        ctx.size = gama.size()[0]
        gama = gama[scale_index]
        bit = bit[scale_index]
        bit = torch.round(bit.abs())
        q_max_ = (2 ** (bit - 1) - 1).expand_as(x)
        gama = gama.abs()
        x_max = q_max_ * gama
        gama_sign = torch.sign(gama)
        x_sign = torch.sign(x)
        x_div = x.div(gama)
        x_div[x_div > q_max_] = q_max_[x_div > q_max_]
        x_div[x_div < -q_max_] = -q_max_[x_div < -q_max_]
        x_q = torch.round(x_div.abs()) * x_sign
        dq_x = x_q.mul(gama)

        ctx.save_for_backward(x, x_div, x_q, x_max, gama, gama_sign)
        ctx.q_max = q_max_
        ctx.index = scale_index
        return dq_x

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output  # grad for features will not be clipped
        # grad alpha
        x0, x, x_q, x_max, gama, gama_sign = ctx.saved_variables
        grad_x[x < -ctx.q_max] = 0
        grad_x[x > ctx.q_max] = 0
        i = (x0.abs() <= x_max).float()

        grad_gama = (grad_output * ((x_q - x) * i + (1 - i) * ctx.q_max * torch.sign(x0))).sum(1).reshape(-1, 1)
        grad_gama_out = grad_gama.new_zeros((ctx.size, 1))
        grad_gama_out = scatter_add(grad_gama, ctx.index, dim=0, out=grad_gama_out)

        grad_b = (grad_output * (1 - i) * torch.sign(x0) * (ctx.q_max + 1) * math.log(2) * gama).sum(1).reshape(-1, 1)
        grad_b_out = grad_b.new_zeros((ctx.size, 1))
        grad_b_out = scatter_add(grad_b, ctx.index, dim=0, out=grad_b_out)
        return grad_x, grad_gama_out, grad_b_out, None, None, None
