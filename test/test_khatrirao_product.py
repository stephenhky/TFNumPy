
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from tfnumpy.tensor import khatrirao_product


class test_KJ(unittest.TestCase):
    def setUp(self):
        x1 = np.array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]], dtype=np.float32)
        x2 = np.array([[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]], dtype=np.float32)
        x = np.zeros((3, 4, 2))
        x[:, :, 0] = x1
        x[:, :, 1] = x2
        self.x = x

        self.a1 = np.array([[2., 3., 3.], [1., 2., -1.]])
        self.a2 = np.array([[2., 3., 1.], [1., 2., 3.]])

        self.answer = np.array([[ 4.,  9.,  3.],
                                [ 2.,  6.,  9.],
                                [ 2.,  6., -1.],
                                [ 1.,  4., -3.]], dtype=np.float32)

    def test_nopredefinedsession(self):
        result = khatrirao_product(self.a1, self.a2)
        self.assertTrue(np.allclose(self.answer, result, rtol=1e-5))

    def test_predefinedsession(self):
        sess = tf.Session()
        result = khatrirao_product(self.a1, self.a2, tfsess=sess)
        self.assertTrue(np.allclose(self.answer, result, rtol=1e-5))
