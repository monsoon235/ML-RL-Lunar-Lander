import torch

config = {
    'device': 'cuda',
    'floatX': torch.float32,
    'thread_num': 8,
    'params': {
        'conv1': {
            'input_shape': (199, 152, 3),
            'out_channel': 32,
            'kernel_size': (8, 8),
            'stride': (4, 4),
            'learning_rate': 0.0001,
            'activate_func': 'relu'
        },
        'conv2': {
            'input_shape': (49, 37, 32),
            'out_channel': 64,
            'kernel_size': (4, 4),
            'stride': (2, 2),
            'learning_rate': 0.0001,
            'activate_func': 'relu'
        },
        'conv3': {
            'input_shape': (24, 18, 64),
            'out_channel': 64,
            'kernel_size': (3, 3),
            'stride': (1, 1),
            'learning_rate': 0.0001,
            'activate_func': 'relu'
        },
        'fc_v1': {
            'dim_in': 22 * 16 * 64,
            'dim_out': 256,
            'learning_rate': 0.0001,
            'activate_func': 'relu'
        },
        'fc_v2': {
            'dim_in': 256,
            'dim_out': 1,
            'learning_rate': 0.1,
            'activate_func': 'none'
        },
        'fc_a1': {
            'dim_in': 22 * 16 * 64,
            'dim_out': 256,
            'learning_rate': 0.0001,
            'activate_func': 'relu'
        },
        'fc_a2': {
            'dim_in': 256,
            'dim_out': 18,
            'learning_rate': 0.0001,
            'activate_func': 'none'
        },
    }
}
