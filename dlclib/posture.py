#
# MIT License
#
# Copyright (c) 2020 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""a few utility classes for interpretation of 'pose' outputs."""

from collections import namedtuple as _namedtuple
from pathlib import Path as _Path
import numpy as _np

from . import load_config as _load_config

class EstimatedPosition(_namedtuple("_EstimatedPosition", ("x", "y", "prob"))):
    """a named tuple class for handling x-coord/y-coord/probability."""
    @property
    def xy(self):
        return _np.array([self.x, self.y])

class PostureMapper(_namedtuple("_PostureMapper", ("parts",))):
    """a base class for performing conversion from 'pose' to a dict."""
    @classmethod
    def from_config(cls, cfg):
        """generates a mapper from a DLC config file."""
        if isinstance(cfg, (str, _Path)):
            cfg = _load_config(cfg)
        return cls(cfg["bodyparts"])

    def map(self, pose, dictclass=dict):
        pose  = pose.flatten()
        out   = dictclass()
        for part, x, y, p in zip(self.parts, pose[:-2:3], pose[1:-1:3], pose[2::3]):
            out[part] = EstimatedPosition(x, y, p)
        return out
