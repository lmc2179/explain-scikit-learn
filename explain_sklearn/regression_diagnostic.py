from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import math

class RegressionDiagnostic(object):
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.n = len(X)

    def get_residuals(self):
        return self.model.predict(self.X) - self.y

    def get_residual_histogram(self):
        r = self.get_residuals()
        mu = np.mean(r)
        variance = (1.0 / (self.n-1)) * np.sum((self.X.reshape(self.n) - mu)**2)
        bin_count = 20
        plt.hist(r, bins=bin_count, normed=True)
        curve_x = np.arange(min(r), max(r), 0.1)
        curve_y = np.array([norm.pdf(x, loc=mu, scale=variance) for x in curve_x])
        plt.plot(curve_x, curve_y)