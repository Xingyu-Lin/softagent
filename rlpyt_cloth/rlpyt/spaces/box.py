import numpy as np

from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.base import Space


class Box(Space):
    """A box in R^n, with specificiable bound and dtype."""

    def __init__(self, low, high, shape=None, dtype="float32", null_value=None):
        """
        low and high are scalars, applied across all dimensions of shape.
        """
        dtype = np.dtype(dtype)
        if dtype.kind == 'i' or dtype.kind == 'u':
            self.box = IntBox(low, high, shape=shape, dtype=dtype, null_value=None)
        elif dtype.kind == 'f':
            self.box = FloatBox(low, high, shape=shape, dtype=dtype, null_value=None)
        else:
            raise NotImplementedError(dtype)

    def sample(self):
        return self.box.sample()

    def null_value(self):
        return self.box.null_value()

    def __repr__(self):
        return f"Box({self.box.low}-{self.box.high - 1} shape={self.box.shape} dtype={self.box.dtype})"

    @property
    def shape(self):
        return self.box.shape

    @property
    def bounds(self):
        return self.box.bounds
