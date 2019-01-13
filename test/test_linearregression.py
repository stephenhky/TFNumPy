
import unittest

import numpy as np
from tfnumpy.regression.linear import fit_linear_regression


class LinearRegressionTester(unittest.TestCase):
    def setUp(self):
        self.xtrain = np.array([[0.], [1.], [2.], [3.]])
        self.ytrain = np.array([[-1.199071], [1.098184], [3.399706], [5.700929]])

    def tearDown(self):
        pass

    def test_regression(self):
        regressed_results = fit_linear_regression(self.xtrain, self.ytrain, max_iter=2000)
        print(regressed_results)
        self.assertAlmostEqual(regressed_results['b'], -1.19, 0.01)
        self.assertAlmostEqual(regressed_results['theta'], 2.30, 0.01)
        self.assertEqual(regressed_results['nbfeatures'], 1)
        self.assertEqual(regressed_results['nbtrain'], 4)

    def test_ridge(self):
        pass

    def test_lasso(self):
        pass

    def test_both(self):
        pass

    def test_random(self):
        pass


if __name__ == '__main__':
    unittest.main()