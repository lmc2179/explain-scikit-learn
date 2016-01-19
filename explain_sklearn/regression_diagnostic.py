from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import chi2, norm, normaltest, shapiro
from scipy.integrate import quad
from sklearn import linear_model, metrics

class RegressionDiagnostic(object):
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.sample_size = len(X)
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

    def get_r_squared(self):
        return self._calculate_r_squared(self.model, self.X, self.y)

    def _calculate_r_squared(self, model, X, y):
        # This is a pure function, unlike get_r_squared, which is a getter
        residuals = model.predict(X) - y
        mean_y = np.mean(y)
        total_sum_of_squares = sum([(y_i - mean_y)**2 for y_i in y])
        squared_residuals = residuals**2
        residual_sum_of_squares = sum(squared_residuals)
        return 1.0 - (residual_sum_of_squares / total_sum_of_squares)


    def _get_chi_squared_p_value(self, test_statistic, degrees_of_freedom):
        p_complement, abserr = quad(chi2.pdf, 0, test_statistic, args=(degrees_of_freedom))
        return 1.0 - p_complement

    def test_noise_breusch_pagan(self):
        residual_model = linear_model.LinearRegression()
        residual_model.fit(self.X, self.residuals**2)
        residual_model_r_squared = self._calculate_r_squared(residual_model, self.X, self.residuals**2)
        lm_statistic = self.sample_size * residual_model_r_squared
        degrees_of_freedom = len(self.model.coef_)
        p = self._get_chi_squared_p_value(lm_statistic, degrees_of_freedom)
        return lm_statistic, p

    def _test_residual_normality(self, test_fxn):
        r = self.get_residuals()
        return test_fxn(r)

    def test_residual_normality_dagostino_pearson(self):
        return self._test_residual_normality(normaltest)

    def test_residual_normality_shapiro_wilks(self):
        return self._test_residual_normality(shapiro)

    def get_variance_inflation_factors(self):
        VIFs = []
        for i in range(len(self.X[0])):
            X = np.concatenate((self.X[0:,:i], self.X[0:,i+1:]), axis=1)
            y = self.X[:, i]
            model = linear_model.LinearRegression()
            model.fit(X, y)
            r_squared = self._calculate_r_squared(model, X, y)
            VIFs.append(1.0 / (1.0 - r_squared))
        return VIFs