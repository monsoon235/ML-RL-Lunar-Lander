import torch.tensor
from src.alg.PB17121687.lunar_lander.config import config

floatX = config['floatX']


class Relu:
    mask: torch.Tensor

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        self.mask = in_data > 0
        return in_data * self.mask

    def backward(self, eta: torch.Tensor) -> torch.Tensor:
        return eta * self.mask
