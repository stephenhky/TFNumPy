
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from tfnumpy.tensor import kronecker_product


class test_KJ(unittest.TestCase):
    def setUp(self):
        self.x1 = np.array([[1, 1.4, 0], [2, 3, 3], [-1, -2, 1.3]], dtype=np.float32)
        self.x2 = np.array([[2, 3], [1, 5]], dtype=np.float32)

    def test_nopredefinedsession(self):
        result = kronecker_product(self.x1, self.x2)
        self.assertTrue(np.allclose(np.kron(self.x1, self.x2), result, rtol=1e-5))

    def test_predefinedsession(self):
        sess = tf.Session()
        result = kronecker_product(self.x1, self.x2, tfsess=sess)
        self.assertTrue(np.allclose(np.kron(self.x1, self.x2), result, rtol=1e-5))
