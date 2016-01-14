import unittest
from sklearn import linear_model
from explain_sklearn import regression_diagnostic

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



if __name__ == '__main__':
    unittest.main()