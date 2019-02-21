import numpy as np
import openpifpaf


def test_normalize_paf():
    connection_intensity_fields = np.array([
        [[0.0, 1.0], [1.0, 0.0]],  # field 0
        [[0.0, 0.5], [0.5, 0.0]],  # field 1
    ])
    j1_fields = np.array([
        [  # field 0
            [[0.0, 1.0], [1.0, 0.0]],  # x
            [[0.0, 1.0], [1.0, 0.0]],  # y
        ],
        [  # field 1
            [[0.0, 0.5], [0.5, 0.0]],  # x
            [[0.0, 0.5], [0.5, 0.0]],  # y
        ],
    ])
    j2_fields = np.array([
        [  # field 0
            [[0.0, 1.0], [1.0, 0.0]],  # x
            [[0.0, 1.0], [1.0, 0.0]],  # y
        ],
        [  # field 1
            [[0.0, 0.5], [0.5, 0.0]],  # x
            [[0.0, 0.5], [0.5, 0.0]],  # y
        ],
    ])
    j1_fields_logb = np.zeros((2, 2, 2))
    j2_fields_logb = np.zeros((2, 2, 2))
    fourds = openpifpaf.decoder.utils.normalize_paf(
        connection_intensity_fields,
        j1_fields, j2_fields,
        j1_fields_logb, j2_fields_logb)
    assert fourds.shape == (2, 2, 4, 2, 2)
    assert fourds[0, :, :, 0, 1].tolist() == [
        [1.0, 2.0, 1.0, 1.0], [1.0, 2.0, 1.0, 1.0]  # each: [intensity, x, y, b]
    ]
    assert fourds[0, :, :, 1, 0].tolist() == [
        [1.0, 1.0, 2.0, 1.0], [1.0, 1.0, 2.0, 1.0]
    ]
