from cnn.cnn import DuelingDQN
import torch
import time

if __name__ == '__main__':
    d1 = DuelingDQN()
    d2 = DuelingDQN()
    start = time.time()
    for i in range(1000):
        print(i)
        d1.forward(torch.randn((5, 199, 152, 3), device='cuda'))
        d1.backward(torch.randn((5, 18), device='cuda'))
        d2.copy_from_other(d1)
    print(f'{time.time() - start} s')
