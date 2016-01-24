import numpy as np
from savvy import regression_diagnostic
from matplotlib import pyplot as plt
from sklearn import linear_model
# This is a demo of the diagnostic tool being used where the assumptions of the linear regression model are met.

DATA_POINTS = 5000
X = np.linspace(0, 10, DATA_POINTS).reshape(DATA_POINTS, 1)
TRUE_SLOPE = 0.5
TRUE_BIAS = 2
noise = np.concatenate((np.random.normal(scale=1.0, size=DATA_POINTS/2), np.random.normal(scale=1.1, size=DATA_POINTS/2)))
y = TRUE_SLOPE*X.reshape(DATA_POINTS) + TRUE_BIAS + noise
lr = linear_model.LinearRegression()
lr.fit(X, y)

diag = regression_diagnostic.RegressionDiagnostic(lr, X, y)
# diag.get_residual_histogram()
# diag.get_predictor_vs_residual_plot()
# diag.plot_coordinates_vs_residual()
diag.plot_residual_vs_residual()
# plt.plot(X.reshape(DATA_POINTS), y, marker='.', linewidth=0.0)
plt.show()
print(diag.test_noise_breusch_pagan())