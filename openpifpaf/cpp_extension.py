def register_ops():
    import importlib
    import os
    import torch

    lib_dir = os.path.dirname(__file__)
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_cpp")
    if ext_specs is None:
        raise ImportError
    torch.ops.load_library(ext_specs.origin)
