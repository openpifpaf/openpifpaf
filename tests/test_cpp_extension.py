import torch

import openpifpaf


def test_op_int64():
    print(openpifpaf.__version__)  # make sure C++ extension is loaded
    torch.ops.openpifpaf_decoder.test_op_int64(42)


def test_op_double():
    print(openpifpaf.__version__)  # make sure C++ extension is loaded
    torch.ops.openpifpaf_decoder.test_op_double(42.0)


def test_class_int64():
    torch.classes.openpifpaf_decoder.TestClass().op_int64(42)


def test_class_double():
    torch.classes.openpifpaf_decoder.TestClass().op_double(42.0)
