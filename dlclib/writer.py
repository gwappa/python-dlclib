#
# MIT License
#
# Copyright (c) 2021 Keisuke Sehara
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

from pathlib import Path as _Path

class CSVWriter:
    """a simple interface for writing estimation outputs in CSV.

    >>> with CSVWriter("test.csv", tfsession) as out:
            for frame in frames:
                out.push(frame)

    """
    def __init__(self, path, tfsession, sep=","):
        """opens an output to `path` using the specified TFSession object."""
        self._session = tfsession
        self._batch   = tfsession.config["batch_size"]
        self._path    = _Path(path)
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True)
        self._out     = open(self._path, "w")
        self._buffer  = []
        self._offset  = 0
        self._sep     = sep
        self._write_headers()

    def _write_headers(self):
        headers = ["frame"]
        for part in self._session.config["all_joints_names"]:
            for attr in ("x", "y", "p"):
                headers.append(f"{part}_{attr}")
        print(self._sep.join(headers), file=self._out)

    def _write_row(self, values):
        print(f"{self._offset}{self._sep}", end="", file=self._out)
        print(self._sep.join(str(v) for v in values), file=self._out)
        self._offset += 1

    def push(self, frame):
        self._buffer.append(frame)
        if len(self._buffer) == self._batch:
            self.flush()
            self._buffer.clear()

    def flush(self):
        bufsiz = len(self._buffer)
        if bufsiz == 0:
            # do nothing
            return
        if self._batch == 1:
            self._write_row(self._session.get_pose(self._buffer[0]))
        else:
            for i in range(bufsiz, self._batch):
                self._buffer.append(self._buffer[-1])
            estimate = self._session.get_pose(self._buffer)
            for i in range(bufsiz):
                self._write_row(estimate[i].ravel(order='C'))

    def close(self):
        if self._out is not None:
            if len(self._buffer) > 0:
                self.flush()
            self._out.close()
            self._out = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
