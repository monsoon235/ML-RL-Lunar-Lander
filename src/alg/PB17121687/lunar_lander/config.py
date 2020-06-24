import torch

config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'floatX': torch.float64,
    'params': {
        'fc1': {
            'dim_in': 8,
            'dim_out': 64,
            'activate_func': 'relu'
        },
        'fc2': {
            'dim_in': 64,
            'dim_out': 64,
            'activate_func': 'relu'
        },
        'fc_v1': {
            'dim_in': 64,
            'dim_out': 32,
            'activate_func': 'relu'
        },
        'fc_v2': {
            'dim_in': 32,
            'dim_out': 1,
            'activate_func': 'none'
        },
        'fc_a1': {
            'dim_in': 64,
            'dim_out': 32,
            'activate_func': 'relu'
        },
        'fc_a2': {
            'dim_in': 32,
            'dim_out': 4,
            'activate_func': 'none'
        },
    }
}
