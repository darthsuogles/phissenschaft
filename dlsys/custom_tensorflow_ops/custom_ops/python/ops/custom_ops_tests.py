# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for zero_out ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from .custom_ops_lib import zero_out, matrix_add


class ZeroOutTest(tf.test.TestCase):
    def test_zero_out(self):
        with self.test_session():
            self.assertAllClose(
                zero_out([[1, 2], [3, 4]]).eval(), np.array([[1, 0], [0, 0]]))


class MatrixAddTest(tf.test.TestCase):
    def _gen_test_forward(self, use_gpu=False, dtype=np.float32):
        _A = np.random.randn(1, 2, 3, 4).astype(dtype) * 10
        _B = np.random.randn(1, 2, 3, 4).astype(dtype) * 10
        bias = dtype(42.)

        expected = _A + _B + bias
        A = tf.convert_to_tensor(_A)
        B = tf.convert_to_tensor(_B)
        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:
            result_tensor = matrix_add(A, B, bias=bias)
            result = sess.run(result_tensor)

        self.assertShapeEqual(expected, result_tensor)
        self.assertAllClose(expected, result)

    def test_forward_uint32_cpu(self):
        self._gen_test_forward(use_gpu=False, dtype=np.float32)

    def test_forward_uint32_gpu(self):
        self._gen_test_forward(use_gpu=True, dtype=np.float32)

if __name__ == '__main__':
    test.main()
