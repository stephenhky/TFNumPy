
import unittest

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from tfnumpy.regression import fit_linear_regression, TFLinearRegression


class LinearRegressionTester(unittest.TestCase):
    def setUp(self):
        self.xtrain = np.array([[0.], [1.], [2.], [3.]])
        self.ytrain = np.array([[-1.199071], [1.098184], [3.399706], [5.700929]])

    def tearDown(self):
        pass

    def test_regression(self):
        # regressed_results, tfsess = fit_linear_regression(self.xtrain, self.ytrain, max_iter=2000)
        tflinreg = TFLinearRegression(max_iter=2000)
        tflinreg.train(self.xtrain, self.ytrain)

        # check regression coefficients
        self.assertEqual(tflinreg.fitted_param['nbfeatures'], 1)
        self.assertEqual(tflinreg.fitted_param['nbtrain'], 4)
        np.testing.assert_almost_equal(tflinreg.fitted_param['theta'][0], 2.30, 2)
        np.testing.assert_almost_equal(tflinreg.fitted_param['b'], -1.20, 2)

        # check prediction
        np.testing.assert_almost_equal(tflinreg.predict(np.array([[1], [2.5]])), np.array([[1.1006711], [4.549828400000001]]), 4)

    def test_ridge(self):
        ridge_reg = Ridge(alpha=0.1, solver='cholesky')
        ridge_reg.fit(self.xtrain, self.ytrain)

        # tfnp_reg, tfsess = fit_linear_regression(self.xtrain, self.ytrain, max_iter=2000, ridge_alpha=0.1)
        tflinreg = TFLinearRegression(max_iter=2000, ridge_alpha=0.1)
        tflinreg.train(self.xtrain, self.ytrain)

        # check sklearn results
        np.testing.assert_almost_equal(ridge_reg.predict([[1]])[0][0], 1.12241141, 2)
        np.testing.assert_almost_equal(ridge_reg.predict([[2.5]])[0][0], 4.50498818, 2)

        # check tensorflow prediction
        np.testing.assert_almost_equal(tflinreg.predict(np.array([[1], [2.5]])),
                                       np.array([[1.220062], [4.4537644]]), 2)

    def test_lasso(self):
        lasso_reg = Lasso(alpha=0.1)
        lasso_reg.fit(self.xtrain, self.ytrain)

        # check sklearn results
        np.testing.assert_almost_equal((lasso_reg.predict([[2.4]]))[0], 4.24807398)
        np.testing.assert_almost_equal((lasso_reg.predict([[1.34]]))[0], 1.89471265)

        # regressed_results, tfsess = fit_linear_regression(self.xtrain, self.ytrain, max_iter=2000, lasso_alpha=0.1)
        tflinreg = TFLinearRegression(max_iter=2000, lasso_alpha=0.1)
        tflinreg.train(self.xtrain, self.ytrain)

        # check tensorflow prediction
        np.testing.assert_almost_equal(tflinreg.predict(np.array([[2.4], [1.34]])),
                                       np.array([[4.279628], [1.89471265]]), 1)

    def test_elasticnet(self):
        elasticnet_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elasticnet_reg.fit(self.xtrain, self.ytrain)

        # check sklearn results
        np.testing.assert_almost_equal((elasticnet_reg.predict([[2.4]]))[0], 4.20583794)
        np.testing.assert_almost_equal((elasticnet_reg.predict([[1.23]]))[0], 1.66316672)

        regressed_results, tfsess = fit_linear_regression(self.xtrain, self.ytrain, max_iter=2000,
                                                          ridge_alpha=0.1*0.5, lasso_alpha=0.1*0.5)
        tflinreg = TFLinearRegression(max_iter=2000, ridge_alpha=0.1*0.5, lasso_alpha=0.1*0.5)
        tflinreg.train(self.xtrain, self.ytrain)

        # check tensorflow prediction
        np.testing.assert_almost_equal(tflinreg.predict(np.array([[2.4], [1.23]])),
                                       np.array([[4.20583794], [1.66316672]]), 1)



if __name__ == '__main__':
    unittest.main()