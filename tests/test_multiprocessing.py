import multiprocessing


class Data:
    d = None


def get_data(_):
    return Data.d


def test_class_attr():
    Data.d = 0.5

    worker_pool = multiprocessing.Pool(2)
    result = worker_pool.starmap(get_data, [(0.0,), (1.0,)])
    assert result == [0.5, 0.5]
