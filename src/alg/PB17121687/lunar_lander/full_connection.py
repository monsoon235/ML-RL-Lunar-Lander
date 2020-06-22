import os

import torch
from cnn.config import config
from src.alg.PB17121687.lunar_lander.relu import Relu

floatX = config['floatX']
device = config['device']


class FullConnection:
    batch: int
    dim_in: int
    dim_out: int

    weight: torch.Tensor
    bias: torch.Tensor
    in_data: torch.Tensor

    def __init__(self, dim_in, dim_out, activate_func: str = None) -> None:
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = torch.randn((dim_out, dim_in), dtype=floatX, device=device) / 500
        self.bias = torch.randn((dim_out,), dtype=floatX, device=device) / 500

        if activate_func == 'relu':
            self.activation = Relu()
        else:
            self.activation = None

    def forward(self, in_data: torch.testing) -> torch.Tensor:
        self.batch = in_data.shape[0]
        assert in_data.shape == (self.batch, self.dim_in)
        self.in_data = in_data
        out_data = torch.add(
            torch.mul(
                in_data.reshape((self.batch, 1, self.dim_in)),
                self.weight.reshape((1, self.dim_out, self.dim_in))
            ).sum(dim=2),
            self.bias.reshape((1, self.dim_out))
        )
        assert out_data.shape == (self.batch, self.dim_out)
        if self.activation is not None:
            out_data = self.activation.forward(out_data)
        return out_data

    def backward(self, eta: torch.Tensor, learning_rate) -> torch.Tensor:
        assert eta.shape == (self.batch, self.dim_out)
        if self.activation is not None:
            eta = self.activation.backward(eta)
        # 本次更新的梯度
        grad_biases = eta.sum(dim=0)
        grad_params = torch.mul(
            eta.reshape((self.batch, self.dim_out, 1)),
            self.in_data.reshape((self.batch, 1, self.dim_in))
        ).sum(dim=0)
        # 下一步传递的梯度
        next_eta = torch.tensordot(eta, self.weight, dims=1)
        assert next_eta.shape == (self.batch, self.dim_in)
        self.weight -= learning_rate / self.batch * grad_params
        self.bias -= learning_rate / self.batch * grad_biases
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
