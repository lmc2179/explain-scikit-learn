import numpy as np
import unittest
from sklearn import linear_model
from savvy import regression_diagnostic
import random
from matplotlib import pyplot as plt

class RegressionDiagnosticTest(unittest.TestCase):
    def test_get_residuals(self):
        lr = linear_model.LinearRegression()
        X = [[0], [1]]
        y = [0, 1]
        lr.fit(X, y)
        diagnostic = regression_diagnostic.RegressionDiagnostic(lr, X, y)
        residuals = diagnostic.get_residuals()
        for r in residuals:
            self.assertAlmostEqual(r, 0) # Note that due to numeric issues, these may not be exactly zero

    def test_breusch_pagan_accept(self):
        diag = self._generate_diag_object(np.random.normal, {'loc': 0.0, 'scale': 1.0}, 100)
        test_statistic, p = diag.test_noise_breusch_pagan()
        print('Breusch-Pagan test P-value for homoskedastically distributed case: ', p)
        self.assertGreater(p, 0.05)

    def test_breusch_pagan_reject(self):
        sample_size = 100
        X_vector = np.linspace(0.0, 1.0, sample_size)
        X = X_vector.reshape(sample_size, 1)
        TRUE_W = random.randint(-10, 10)
        TRUE_BIAS = random.randint(-10, 10)
        noise = np.concatenate((np.random.normal(0.0, 0.5, sample_size/2),np.random.normal(0.0, 1.5, sample_size/2)))
        y = TRUE_W*X_vector + TRUE_BIAS + noise
        lr = linear_model.LinearRegression()
        lr.fit(X, y)
        diag = regression_diagnostic.RegressionDiagnostic(lr, X, y)
        test_statistic, p = diag.test_noise_breusch_pagan()
        print('Breusch-Pagan test P-value for heteroskedastically distributed case: ', p)
        self.assertLess(p, 0.05)

    def test_shapiro_accept(self):
        diag = self._generate_diag_object(np.random.normal, {'loc': 0.0, 'scale': 1.0}, 100)
        test_statistic, p = diag.test_residual_normality_shapiro_wilks()
        print('Shapiro-Wilks test P-value for normally distributed case: ', p)
        self.assertGreater(p, 0.05)

    def test_shapiro_reject_T(self):
        diag = self._generate_diag_object(np.random.standard_t, {'df': 3}, 100)
        test_statistic, p = diag.test_residual_normality_shapiro_wilks()
        print('Shapiro-Wilks test P-value for T distributed case: ', p)
        self.assertLess(p, 0.05)

    def test_dagostino_pearson_accept(self):
        diag = self._generate_diag_object(np.random.normal, {'loc': 0.0, 'scale': 1.0}, 100)
        test_statistic, p = diag.test_residual_normality_dagostino_pearson()
        print('D\'Agostino-Pearson test P-value for normally distributed case: ', p)
        self.assertGreater(p, 0.05)

    def test_dagostino_pearson_reject_T(self):
        diag = self._generate_diag_object(np.random.standard_t, {'df': 3}, 100)
        test_statistic, p = diag.test_residual_normality_dagostino_pearson()
        print('D\'Agostino-Pearson test P-value for T distributed case: ', p)
        self.assertLess(p, 0.05)

    def _generate_diag_object(self, sampler_fxn, sampler_kwargs, sample_size):
        # Create a dataset with random noise, train a linear regression model, and construct a diagnostic object from it
        X_vector = np.random.uniform(0.0, 1.0, sample_size)
        X = X_vector.reshape(sample_size, 1)
        TRUE_W = random.randint(-10, 10)
        TRUE_BIAS = random.randint(-10, 10)
        noise = sampler_fxn(size=sample_size, **sampler_kwargs)
        y = TRUE_W*X_vector + TRUE_BIAS + noise
        lr = linear_model.LinearRegression()
        lr.fit(X, y)
        diag = regression_diagnostic.RegressionDiagnostic(lr, X, y)
        return diag

    def test_VIF_no_correlation(self):
        SAMPLE_SIZE = 100
        X = np.random.uniform(0, 1.0, (SAMPLE_SIZE, 2))
        y = np.zeros(SAMPLE_SIZE) # The model is irrelevant in this test
        lr = linear_model.LinearRegression()
        lr.fit(X, y)
        diag = regression_diagnostic.RegressionDiagnostic(lr, X, y)
        vifs = diag.get_variance_inflation_factors()
        print('VIFs for uncorrelated case:', vifs)
        x1_vif, x2_vif = vifs
        self.assertLess(x1_vif, 1.5)
        self.assertLess(x2_vif, 1.5)

    def test_VIF_with_correlation(self):
        SAMPLE_SIZE = 100
        X_mu = np.array([0, 0])
        X_covariance = np.array([[30, 20], [20, 15]])
        X = np.random.multivariate_normal(X_mu, X_covariance, SAMPLE_SIZE)
        y = np.zeros(SAMPLE_SIZE) # The model is irrelevant in this test
        lr = linear_model.LinearRegression()
        lr.fit(X, y)
        diag = regression_diagnostic.RegressionDiagnostic(lr, X, y)
        vifs = diag.get_variance_inflation_factors()
        x1_vif, x2_vif = vifs
        self.assertGreater(x1_vif, 5.0)
        self.assertGreater(x2_vif, 5.0)
        print('VIFs for correlated case:', vifs)

# TODO: Additional tests
# 3) VIF calculation

if __name__ == '__main__':
    unittest.main()