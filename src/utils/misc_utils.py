def get_params_from_file(filepath, params_name='params'):
    import importlib
    module = importlib.import_module(filepath)
    params = getattr(module, params_name)
    return params
