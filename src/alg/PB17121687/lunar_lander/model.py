import os

import torch.tensor

from src.alg.PB17121687.lunar_lander.full_connection import FullConnection
from src.alg.PB17121687.lunar_lander.config import config

floatX = config['floatX']
device = config['device']


class DuelingDoubleDQN:
    bs: int
    fc1: FullConnection
    fc2: FullConnection
    fc_v1: FullConnection
    fc_v2: FullConnection
    fc_a1: FullConnection
    fc_a2: FullConnection

    def __init__(self) -> None:
        self.fc1 = FullConnection(**config['params']['fc1'])
        self.fc2 = FullConnection(**config['params']['fc2'])
        self.fc_v1 = FullConnection(**config['params']['fc_v1'])
        self.fc_v2 = FullConnection(**config['params']['fc_v2'])
        self.fc_a1 = FullConnection(**config['params']['fc_a1'])
        self.fc_a2 = FullConnection(**config['params']['fc_a2'])

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        self.bs = in_data.shape[0]
        out_data = self.fc1.forward(in_data)
        out_data = self.fc2.forward(out_data)
        v = self.fc_v1.forward(out_data)
        v = self.fc_v2.forward(v)
        a = self.fc_a1.forward(out_data)
        a = self.fc_a2.forward(a)
        return v + (a - a.mean(dim=1, keepdim=True))

    def backward(self, eta: torch.Tensor, learning_rate: float) -> torch.Tensor:
        dim_out = config['params']['fc_a2']['dim_out']
        eta_v = eta.sum(dim=1, keepdim=True)
        o_by_a = torch.empty(size=(self.bs, dim_out, dim_out), dtype=floatX, device=device)
        o_by_a[:, :, :] = -1 / dim_out
        for i in range(dim_out):
            o_by_a[:, i, i] = (dim_out - 1) / dim_out
        eta_a = (o_by_a * eta.reshape((self.bs, dim_out, 1))).sum(dim=1)
        eta_v = self.fc_v2.backward(eta_v, learning_rate)
        eta_v = self.fc_v1.backward(eta_v, learning_rate)
        eta_a = self.fc_a2.backward(eta_a, learning_rate)
        eta_a = self.fc_a1.backward(eta_a, learning_rate)
        eta = eta_a + eta_v
        eta = self.fc2.backward(eta, learning_rate)
        eta = self.fc1.backward(eta, learning_rate)
        return eta

    def save(self, folder_path: str):
        self.fc1.save(os.path.join(folder_path, 'fc1'))
        self.fc2.save(os.path.join(folder_path, 'fc2'))
        self.fc_v1.save(os.path.join(folder_path, 'fc_v1'))
        self.fc_v2.save(os.path.join(folder_path, 'fc_v2'))
        self.fc_a1.save(os.path.join(folder_path, 'fc_a1'))
        self.fc_a2.save(os.path.join(folder_path, 'fc_a2'))

    def load(self, folder_path: str):
        self.fc1.load(os.path.join(folder_path, 'fc1'))
        self.fc2.load(os.path.join(folder_path, 'fc2'))
        self.fc_v1.load(os.path.join(folder_path, 'fc_v1'))
        self.fc_v2.load(os.path.join(folder_path, 'fc_v2'))
        self.fc_a1.load(os.path.join(folder_path, 'fc_a1'))
        self.fc_a2.load(os.path.join(folder_path, 'fc_a2'))
