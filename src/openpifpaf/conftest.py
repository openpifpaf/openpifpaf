"""Context for doctests."""

import numpy
import pytest

import openpifpaf


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy


@pytest.fixture(autouse=True)
def add_openpifpaf(doctest_namespace):
    doctest_namespace['openpifpaf'] = openpifpaf
