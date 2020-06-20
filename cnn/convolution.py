import os

import torch
from cnn.config import config
from cnn.relu import Relu

floatX = config['floatX']
device = config['device']


# 增强版 im2col
def im2col_enhanced(im: torch.Tensor, kernel_size, stride, inner_stride=(1, 1)) -> torch.Tensor:
    kh, kw = kernel_size
    sh, sw = stride
    ish, isw = inner_stride
    b, h, w, c = im.shape
    assert (h - kh * ish) % sh == 0
    assert (w - kw * isw) % sw == 0
    out_h = (h - kh * ish) // sh + 1
    out_w = (w - kw * isw) // sw + 1
    out_size = (b, out_h, out_w, kh, kw, c)
    s = im.stride()
    out_stride = (s[0], s[1] * sh, s[2] * sw, s[1] * ish, s[2] * isw, s[3])
    col_img = im.as_strided(size=out_size, stride=out_stride)
    return col_img


class Convolution:
    batch: int
    in_h: int
    in_w: int
    real_in_h: int
    real_in_w: int
    out_h: int
    out_w: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    in_channel: int
    out_channel: int
    learning_rate: float

    with_pad: bool

    in_data: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor
    grad_weight: torch.Tensor

    def __init__(self, input_shape, out_channel, kernel_size, stride, learning_rate, activate_func: str = None):
        self.in_h, self.in_w, self.in_channel = input_shape
        self.learning_rate = learning_rate
        self.out_channel = out_channel
        self.kernel_h, self.kernel_w = kernel_size
        self.stride_h, self.stride_w = stride

        # padding
        self.real_in_h = self.in_h
        self.real_in_w = self.in_w
        if (self.in_h - self.kernel_h) % self.stride_h != 0:
            self.in_h += self.stride_h - (self.in_h - self.kernel_h) % self.stride_h
            assert (self.in_h - self.kernel_h) % self.stride_h == 0
        if (self.in_w - self.kernel_w) % self.stride_w != 0:
            self.in_w += self.stride_w - (self.in_w - self.kernel_w) % self.stride_w
            assert (self.in_w - self.kernel_w) % self.stride_w == 0
        self.with_pad = (self.in_h != self.real_in_h) or (self.in_w != self.real_in_w)

        self.out_h = (self.in_h - self.kernel_h) // self.stride_h + 1
        self.out_w = (self.in_w - self.kernel_w) // self.stride_w + 1

        self.weight = torch.randn(
            (self.kernel_h, self.kernel_w, self.in_channel, out_channel),
            dtype=floatX, device=device)
        self.bias = torch.randn((self.out_channel,), dtype=floatX, device=device)
        self.grad_weight = torch.empty(
            (self.kernel_h, self.kernel_w, self.in_channel, out_channel),
            dtype=floatX, device=device)

        if activate_func == 'relu':
            self.activation = Relu()
        else:
            self.activation = None

    # 测试通过
    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        self.batch = in_data.shape[0]
        assert in_data.shape == (self.batch, self.real_in_h, self.real_in_w, self.in_channel)
        # padding
        if self.with_pad:
            new_in_data = torch.zeros((self.batch, self.in_h, self.in_w, self.in_channel), dtype=floatX, device=device)
            new_in_data[:, :self.real_in_h, :self.real_in_w, :] = in_data
            in_data = new_in_data

        self.in_data = in_data
        col_img = im2col_enhanced(in_data, (self.kernel_h, self.kernel_w), (self.stride_h, self.stride_w))
        out = torch.tensordot(col_img, self.weight, dims=[(3, 4, 5), (0, 1, 2)]) \
              + self.bias.reshape((1, 1, 1, self.out_channel))
        if self.activation is not None:
            out = self.activation.forward(out)
        return out

    # 测试通过
    def backward(self, eta: torch.Tensor) -> torch.Tensor:
        assert eta.shape == (self.batch, self.out_h, self.out_w, self.out_channel)
        if self.activation is not None:
            eta = self.activation.backward(eta)
        # filters 梯度
        col_img = im2col_enhanced(self.in_data, (self.kernel_h, self.kernel_w), (self.stride_h, self.stride_w))
        self.grad_weight[:, :, :, :] = 0
        for b in range(self.batch):
            self.grad_weight += torch.tensordot(
                col_img[b], eta[b], dims=[(0, 1), (0, 1)]
            )
        # biases 梯度
        grad_bias = eta.sum(dim=(0, 1, 2))
        # in_data 梯度
        # 这部分的实现参照 PPT
        padding_eta = torch.zeros(
            (self.batch,
             2 * (self.kernel_h - 1) + (self.out_h - 1) * self.stride_h + 1,
             2 * (self.kernel_w - 1) + (self.out_w - 1) * self.stride_w + 1,
             self.out_channel), dtype=floatX, device=device)
        pad_h = self.kernel_h - 1
        pad_w = self.kernel_w - 1
        padding_eta[:, pad_h:-pad_h:self.stride_h, pad_w:-pad_w:self.stride_w, :] = eta  # padding_eta 其他部分为0
        filters_flip = self.weight.flip(dims=(0, 1))
        # 进行卷积运算
        col_eta = im2col_enhanced(padding_eta, (self.kernel_h, self.kernel_w), (1, 1))
        assert col_eta.shape == (self.batch, self.in_h, self.in_w, self.kernel_h, self.kernel_w, self.out_channel)
        next_eta = torch.tensordot(col_eta, filters_flip, dims=[(3, 4, 5), (0, 1, 3)])
        assert next_eta.shape == (self.batch, self.in_h, self.in_w, self.in_channel)
        # 更新
        self.weight -= self.learning_rate * self.grad_weight / self.batch
        self.bias -= self.learning_rate * grad_bias / self.batch
        # 去 padding
        if self.with_pad:
            next_eta = next_eta[:, :self.real_in_h, :self.real_in_w, :]
        return next_eta

    def save(self, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.weight, os.path.join(folder_path, 'weight.bin'))
        torch.save(self.bias, os.path.join(folder_path, 'bias.bin'))

    def load(self, folder_path: str):
        self.weight = torch.load(os.path.join(folder_path, 'weight.bin'))
        self.bias = torch.load(os.path.join(folder_path, 'bias.bin'))

    def copy_from_other(self, other):
        self.weight = other.weight.clone()
        self.bias = other.bias.clone()
