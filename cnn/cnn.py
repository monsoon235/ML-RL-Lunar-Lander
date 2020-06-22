import torch
import os

from cnn.config import config
from cnn.convolution import Convolution
from src.alg.PB17121687.lunar_lander.full_connection import FullConnection

floatX = config['floatX']
device = config['device']
torch.set_num_threads(config['thread_num'])


# max_batch = 200
# img_size = 119
# deepid_dim = 1000
# cross_dim = 500
#
# # 1e-3 发散
# # 2e-4 发散
#
# # 1e-5 迭代 300 即可
#
# conv1_learning_rate = 1e-5 / max_batch
# conv2_learning_rate = 1e-5 / max_batch
# conv3_learning_rate = 1e-5 / max_batch
# conv4_learning_rate = 1e-5 / max_batch
# fc1_learning_rate = 1e-5 / max_batch
# fc2_learning_rate = 1e-5 / max_batch
# fc_cross_learning_rate = 1e-5 / max_batch
# fc_softmax_learning_rate = 1e-5 / max_batch
#
# ks1 = 4
# ks2 = 3
# ks3 = 3
# ks4 = 2
#
# ps1 = 2
# ps2 = 2
# ps3 = 2
#
# ci = 3
# oc1 = 20
# oc2 = 40
# oc3 = 60
# oc4 = 80
#
# stride1 = 1
# stride2 = 1
# stride3 = 1
# stride4 = 1
#
# test_imgs: torch.Tensor
# test_labels: torch.Tensor


class DuelingDQN:
    batch: int
    conv1: Convolution
    conv2: Convolution
    conv3: Convolution
    fc_v1: FullConnection
    fc_v2: FullConnection
    fc_a1: FullConnection
    fc_a2: FullConnection

    def __init__(self) -> None:
        self.conv1 = Convolution(**config['params']['conv1'])
        self.conv2 = Convolution(**config['params']['conv2'])
        self.conv3 = Convolution(**config['params']['conv3'])
        self.fc_v1 = FullConnection(**config['params']['fc_v1'])
        self.fc_v2 = FullConnection(**config['params']['fc_v2'])
        self.fc_a1 = FullConnection(**config['params']['fc_a1'])
        self.fc_a2 = FullConnection(**config['params']['fc_a2'])

    def forward(self, in_data: torch.Tensor) -> torch.Tensor:
        self.batch = in_data.shape[0]
        out_data = self.conv1.forward(in_data)
        out_data = self.conv2.forward(out_data)
        out_data = self.conv3.forward(out_data)
        out_data = out_data.flatten(start_dim=1, end_dim=-1)
        v = self.fc_v1.forward(out_data)
        v = self.fc_v2.forward(v)
        a = self.fc_a1.forward(out_data)
        a = self.fc_a2.forward(a)
        return v + (a - a.mean(dim=1, keepdim=True))

    def backward(self, eta: torch.Tensor) -> torch.Tensor:
        dim_out = config['params']['fc_a2']['dim_out']
        eta_v = eta.sum(dim=1, keepdim=True)
        o_by_a = torch.empty(size=(self.batch, dim_out, dim_out), dtype=floatX, device=device)
        o_by_a[:, :, :] = -1 / dim_out
        for i in range(dim_out):
            o_by_a[:, i, i] = (dim_out - 1) / dim_out
        eta_a = (o_by_a * eta.reshape((self.batch, dim_out, 1))).sum(dim=1)

        eta_v = self.fc_v2.backward(eta_v)
        eta_v = self.fc_v1.backward(eta_v)

        eta_a = self.fc_a2.backward(eta_a)
        eta_a = self.fc_a1.backward(eta_a)

        eta = eta_a + eta_v
        eta = self.conv3.backward(eta.reshape(self.batch, 22, 16, 64))
        eta = self.conv2.backward(eta)
        eta = self.conv1.backward(eta)

        return eta

    def save(self, folder_path: str):
        self.conv1.save(os.path.join(folder_path, 'conv1'))
        self.conv2.save(os.path.join(folder_path, 'conv2'))
        self.conv3.save(os.path.join(folder_path, 'conv3'))
        self.fc_v1.save(os.path.join(folder_path, 'fc_v1'))
        self.fc_v2.save(os.path.join(folder_path, 'fc_v2'))
        self.fc_a1.save(os.path.join(folder_path, 'fc_a1'))
        self.fc_a2.save(os.path.join(folder_path, 'fc_a2'))

    def load(self, folder_path: str):
        self.conv1.load(os.path.join(folder_path, 'conv1'))
        self.conv2.load(os.path.join(folder_path, 'conv2'))
        self.conv3.load(os.path.join(folder_path, 'conv3'))
        self.fc_v1.load(os.path.join(folder_path, 'fc_v1'))
        self.fc_v2.load(os.path.join(folder_path, 'fc_v2'))
        self.fc_a1.load(os.path.join(folder_path, 'fc_a1'))
        self.fc_a2.load(os.path.join(folder_path, 'fc_a2'))

    def copy_from_other(self, other):
        self.conv1.copy_from_other(other.conv1)
        self.conv2.copy_from_other(other.conv2)
        self.conv3.copy_from_other(other.conv3)
        self.fc_v1.copy_from_other(other.fc_v1)
        self.fc_v2.copy_from_other(other.fc_v2)
        self.fc_a1.copy_from_other(other.fc_a1)
        self.fc_a2.copy_from_other(other.fc_a2)

# class LightCNN:
#     batch: int
#     conv1: convolution.Convolution
#     conv2: convolution.Convolution
#     conv3: convolution.Convolution
#     conv4: convolution.Convolution
#     pool1: pooling.Pooling
#     pool2: pooling.Pooling
#     pool3: pooling.Pooling
#     fc1: full_connection.FullConnection
#     fc2: full_connection.FullConnection
#
#     def __init__(self) -> None:
#         size = img_size
#         self.conv1 = convolution.Convolution(input_shape=(size, size, ci), out_channel=oc1,
#                                              kernel_size=(ks1, ks1), stride=(stride1, stride1),
#                                              learning_rate=conv1_learning_rate, activate_func='relu')
#         assert (size - ks1) % stride1 == 0
#         size = (size - ks1) // stride1 + 1
#         self.pool1 = pooling.Pooling(input_shape=(size, size, oc1), pool_size=ps1)
#         assert size % 2 == 0
#         size //= 2
#         self.conv2 = convolution.Convolution(input_shape=(size, size, oc1), out_channel=oc2,
#                                              kernel_size=(ks2, ks2), stride=(stride2, stride2),
#                                              learning_rate=conv2_learning_rate, activate_func='relu')
#         assert (size - ks2) % stride2 == 0
#         size = (size - ks2) // stride2 + 1
#         self.pool2 = pooling.Pooling(input_shape=(size, size, oc2), pool_size=ps2)
#         assert size % 2 == 0
#         size //= 2
#         self.conv3 = convolution.Convolution(input_shape=(size, size, oc2), out_channel=oc3,
#                                              kernel_size=(ks3, ks3), stride=(stride3, stride3),
#                                              learning_rate=conv3_learning_rate, activate_func='relu')
#         assert (size - ks3) % stride3 == 0
#         size = (size - ks3) // stride3 + 1
#         self.pool3 = pooling.Pooling(input_shape=(size, size, oc3), pool_size=ps3)
#         assert size % 2 == 0
#         size //= 2
#         self.conv4 = convolution.Convolution(input_shape=(size, size, oc3), out_channel=oc4,
#                                              kernel_size=(ks4, ks4), stride=(stride4, stride4),
#                                              learning_rate=conv4_learning_rate, activate_func='relu')
#         assert (size - ks4) % stride4 == 0
#         size4 = (size - ks4) // stride4 + 1
#         self.fc1 = full_connection.FullConnection(dim_in=size * size * oc3,
#                                                   dim_out=deepid_dim // 2, learning_rate=fc1_learning_rate,
#                                                   activate_func='relu')
#         self.fc2 = full_connection.FullConnection(dim_in=size4 * size4 * oc4,
#                                                   dim_out=deepid_dim // 2, learning_rate=fc2_learning_rate,
#                                                   activate_func='relu')
#
#     def forward(self, in_data: torch.Tensor) -> torch.Tensor:
#         self.batch = in_data.shape[0]
#         assert in_data.shape == (self.batch, img_size, img_size, 3)
#         out3 = self.pool3.forward(self.conv3.forward(
#             self.pool2.forward(self.conv2.forward(
#                 self.pool1.forward(self.conv1.forward(
#                     in_data
#                 ))))))
#         out4 = self.conv4.forward(out3)
#         out3 = out3.flatten(start_dim=1, end_dim=-1)
#         out4 = out4.flatten(start_dim=1, end_dim=-1)
#
#         id = torch.empty((self.batch, deepid_dim), dtype=floatX, device=device)
#         id[:, :deepid_dim // 2] = self.fc1.forward(out3)
#         id[:, deepid_dim // 2:] = self.fc2.forward(out4)
#         return id
#
#     def backward(self, eta: torch.Tensor) -> torch.Tensor:
#         assert eta.shape == (self.batch, deepid_dim)
#         shape3 = (self.pool3.batch, self.pool3.height // self.pool3.pool_size,
#                   self.pool3.width // self.pool3.pool_size, oc3)
#         shape4 = (self.conv4.batch, self.conv4.out_h, self.conv4.out_w, self.conv4.out_channel)
#         eta3 = self.fc1.backward(eta[:, :deepid_dim // 2]).reshape(shape3)
#         eta4 = self.fc2.backward(eta[:, deepid_dim // 2:])
#         eta4 = self.conv4.backward(eta4.reshape(shape4))
#         next_eta = self.conv1.backward(self.pool1.backward(
#             self.conv2.backward(self.pool2.backward(
#                 self.conv3.backward(self.pool3.backward(
#                     eta3 + eta4
#                 ))))))
#         return next_eta

# def save(self, folder_path: str):
#     self.conv1.save(os.path.join(folder_path, 'conv1'))
#     self.conv2.save(os.path.join(folder_path, 'conv2'))
#     self.conv3.save(os.path.join(folder_path, 'conv3'))
#     self.conv4.save(os.path.join(folder_path, 'conv4'))
#     self.pool1.save(os.path.join(folder_path, 'pool1'))
#     self.pool2.save(os.path.join(folder_path, 'pool2'))
#     self.pool3.save(os.path.join(folder_path, 'pool3'))
#     self.fc1.save(os.path.join(folder_path, 'fc1'))
#     self.fc2.save(os.path.join(folder_path, 'fc2'))
#
# def load(self, folder_path: str):
#     self.conv1.load(os.path.join(folder_path, 'conv1'))
#     self.conv2.load(os.path.join(folder_path, 'conv2'))
#     self.conv3.load(os.path.join(folder_path, 'conv3'))
#     self.conv4.load(os.path.join(folder_path, 'conv4'))
#     self.pool1.load(os.path.join(folder_path, 'pool1'))
#     self.pool2.load(os.path.join(folder_path, 'pool2'))
#     self.pool3.load(os.path.join(folder_path, 'pool3'))
#     self.fc1.load(os.path.join(folder_path, 'fc1'))
#     self.fc2.load(os.path.join(folder_path, 'fc2'))
