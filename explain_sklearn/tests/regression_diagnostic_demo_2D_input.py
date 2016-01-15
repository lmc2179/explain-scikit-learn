import numpy as np
from explain_sklearn import regression_diagnostic
from matplotlib import pyplot as plt
from sklearn import linear_model
# This is a demo of the diagnostic tool being used where the assumptions of the linear regression model are met.

DATA_POINTS = 500
X1 = np.random.uniform(0, 10, DATA_POINTS).reshape(DATA_POINTS, 1)
X2 = np.random.uniform(0, 10, DATA_POINTS).reshape(DATA_POINTS, 1)
X = np.concatenate((X1, X2), axis=1)
TRUE_W = np.array([0.5, 1.2])
TRUE_BIAS = 2
y = np.dot(TRUE_W, X.T) + TRUE_BIAS + np.random.normal(size=DATA_POINTS)
lr = linear_model.LinearRegression()
lr.fit(X, y)
print(lr.coef_)

diag = regression_diagnostic.RegressionDiagnostic(lr, X, y)
# diag.get_residual_histogram()
# diag.get_predictor_vs_residual_plot()
diag.get_predictor_vs_squared_residual_plot()
plt.show()