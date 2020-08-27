"""Tests configuration system for multiprocessing.

Multiprocessing using 'fork' works in our configuration system but
has been shown to be problematic on Mac-based systems with threading in the
main process.
Multiprocessing with 'spawn' does not work for configuration
(new default for Python 3.8 on Mac).

Ref:
* https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
"""

import multiprocessing
import sys

import pytest


class Data:
    d = None


def get_data(_):
    return Data.d


def test_class_attr():
    if sys.platform.startswith('win'):
        pytest.skip('multiprocessing not supported on windows')

    Data.d = 0.5

    multiprocessing_context = multiprocessing.get_context('fork')
    worker_pool = multiprocessing_context.Pool(2)
    result = worker_pool.starmap(get_data, [(0.0,), (1.0,)])
    assert result == [0.5, 0.5]
