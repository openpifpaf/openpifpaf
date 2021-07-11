import importlib
import os
import torch


def register_ops():
    lib_dir = os.path.dirname(__file__)
    if hasattr(os, 'add_dll_directory'):  # for Windows
        import ctypes  # pylint: disable=import-outside-toplevel

        kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
        if hasattr(kernel32, 'AddDllDirectory'):
            kernel32.AddDllDirectory.restype = ctypes.c_void_p

        os.add_dll_directory(lib_dir)  # pylint: disable=no-member

    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        importlib.machinery.EXTENSION_SUFFIXES
    )

    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec("_cpp")
    if ext_specs is None:
        raise ImportError
    torch.ops.load_library(ext_specs.origin)
