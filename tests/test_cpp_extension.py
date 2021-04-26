import torch

import openpifpaf


def test_op_int64():
    print(openpifpaf.__version__)  # make sure C++ extension is loaded
    torch.ops.openpifpaf_decoder.test_op_int64(42)


def test_op_double():
    print(openpifpaf.__version__)  # make sure C++ extension is loaded
    torch.ops.openpifpaf_decoder.test_op_double(42.0)
