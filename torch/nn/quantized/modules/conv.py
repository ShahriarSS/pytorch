# coding=utf-8
r"""Quantized convolution modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import torch
from torch.nn.modules.utils import _pair


from torch.nn.modules.conv import _ConvNd

"""Computes the output shape given convolution parameters."""
def _conv_output_shape(input_size, kernel_size, padding, stride, dilation,
                       output_padding=0):
    return np.floor((input_size + 2 * padding - kernel_size - (kernel_size - 1)
                    * (dilation - 1)) / stride) + 2 * output_padding + 1


class _ConvNd(Module):
    def __init__(self, weight, bias, scale, zero_point, dtype,
                 stride, padding, dilation, transposed, output_padding,
                 groups, padding_mode):
        if transposed:
            raise NotImplementedError("Transposed convolution not implemented!")
        super(_ConvNd, self).__init__()
        self.in_channels = weight.shape[1] * groups
        self.out_channels = weight.shape[0]
        self.kernel_size = weight.shape[2:]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        self._weight = weight  # Store the original weight for future reference.
        self._packed_weight = ops.quantized.fbgemm_conv_prepack(self._weight,
                                                                self.groups)
        self.bias = bias
        self.bias.requires_grad = False  # Inference only!

        self.dtype = dtype
        self.scale = scale
        self.zero_point = zero_point

    @property
    def weight(self):
        return self._packed_weight

    @weight.setter
    def weight(self, w):
        self._weight = w
        self._packed_weight = ops.quantized.fbgemm_conv_prepack(self._weight,
                                                                self.groups)

    @weight.deleter
    def weight(self):
        del self._weight
        del self._packed_weight

    def extra_repr(self):
        s = super(_ConvNd, self).extra_repr()
        s += ', scale={scale}, zero_point={zero_point}, dtype={dtype}'
        return s.format(**self.__dict__)


class Conv2d(_ConvNd):
    def __init__(self, weight, bias, scale, zero_point, dtype,
                 stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros'):
        if padding_mode != 'zeros':
            raise NotImplementedError(
                "Currently only zero-padding is supported!")
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        kernel_size = _pair(kernel_size)
        transposed = False
        output_padding = _pair(0)
        super(Conv2d, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     transposed=transposed,
                                     output_padding=output_padding,
                                     groups=groups,
                                     bias=True,
                                     padding_mode=padding_mode)
        del self.weight
        del self.bias

        qweight = torch._empty_affine_quantized(
            [out_channels, kernel_size[0], kernel_size[1],
             in_channels // self.groups],
            scale=1, zero_point=0, dtype=torch.qint8)
        qbias = torch._empty_affine_quantized([out_channels],
                                              scale=1, zero_point=0,
                                              dtype=torch.qint32)
        self.register_buffer('_packed_weight',
                             torch.ops.quantized.fbgemm_conv_prepack(qweight, self.groups))
        self.register_buffer('bias', qbias)
        self.register_buffer('_scale', torch.Tensor([1]))
        self.register_buffer('_zero_point', torch.Tensor([0]).to(torch.int))

    @property
    def weight(self):
        return self._packed_weight
        # return torch.ops.quantized.fbgemm_conv_unpack(self._packed_weight)

    @weight.setter
    def weight(self, w):
        self._packed_weight = torch.ops.quantized.fbgemm_conv_prepack(w, self.groups)

    @property
    def scale(self):
        return self._scale.item()

    @scale.setter
    def scale(self, s):
        if isinstance(s, torch.Tensor):
            self._scale = s
        else:
            self._scale = torch.Tensor([s])

    @property
    def zero_point(self):
        return self._zero_point.item()

    @zero_point.setter
    def zero_point(self, zp):
        if isinstance(zp, torch.Tensor):
            self._zero_point = zp
        else:
            self._zero_point = torch.Tensor([zp]).to(torch.int)

    def forward(self, input):
        if input.ndim != 4:
            raise ValueError("Input shape must be `(N, C, H, W)`!")
        return ops.quantized.fbgemm_conv2d(input,
                                           self.weight, self.bias,
                                           self.stride, self.padding,
                                           self.dilation, self.groups,
                                           self.scale, self.zero_point)
