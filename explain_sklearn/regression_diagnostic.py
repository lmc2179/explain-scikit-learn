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
        self.residuals = self.model.predict(self.X) - self.y

    def get_residuals(self):
        return self.residuals

    def get_residual_histogram(self):
        r = self.get_residuals()
        mu, variance = norm.fit(r)
        bin_count = 20
        plt.title('Distribution of residuals with Gaussian MLE')
        plt.hist(r, bins=bin_count, normed=True)
        curve_x = np.arange(min(r), max(r), 0.1)
        curve_y = np.array([norm.pdf(x, loc=mu, scale=variance) for x in curve_x])
        plt.plot(curve_x, curve_y)

    def show(self):
        plt.show()

    def get_predictor_vs_residual_plot(self):
        r = self.get_residuals()
        num_input_columns = len(self.X[0])
        X_components = (self.X[:,i] for i in range(num_input_columns))
        plt.figure(1)
        plot_base = 210
        for i, component in enumerate(X_components):
            plt.subplot(plot_base + i)
            plt.plot(component, r, linewidth=0.0, marker='.')

    def get_predictor_vs_squared_residual_plot(self):
        r = self.get_residuals()
        r_squared = r**2
        num_input_columns = len(self.X[0])
        X_components = (self.X[:,i] for i in range(num_input_columns))
        plt.figure(1)
        plot_base = 210
        for i, component in enumerate(X_components):
            plt.subplot(plot_base + i)
            plt.plot(component, r_squared, linewidth=0.0, marker='.')