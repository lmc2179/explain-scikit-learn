# regression_diagnostic.py


This module exposes the RegressionDiagnostic class, which helps the user understand whether or not the required assumptions for their model to work are met.

The theoretical foundation for some of the tests (particularly those based on classical linear regression) can be found in [Cosma Shalizi's regression notes](http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/07/lecture-07.pdf).

Like all modules in this package, this module works with regression models from scikit learn, such as sklearn.linear_model.LinearRegression.


&nbsp;&nbsp;&nbsp;&nbsp;**RegressionDiagnostic**(RegressionDiagnostic):


&nbsp;&nbsp;&nbsp;&nbsp;    This is the main class exposed by the module. It requires:

&nbsp;&nbsp;&nbsp;&nbsp;        * a fitted model

&nbsp;&nbsp;&nbsp;&nbsp;        * the training inputs (X)

&nbsp;&nbsp;&nbsp;&nbsp;        * the training outputs (y)

&nbsp;&nbsp;&nbsp;&nbsp;    Once these are provided to the __init__ method, other tests can be run by calling the methods of the instantiated object.
&nbsp;&nbsp;&nbsp;&nbsp;    

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**get_r_squared**(self):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Calculates the r-squared for the fitted model.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**get_residuals**(self):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Returns the distance between the predicted output and the training output for each training point.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**get_variance_inflation_factors**(self):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Calculates the variance inflation factor (VIF) for each component of the input.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**plot_coordinates_vs_residual**(self):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Plots the residuals in the order they were computed.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**plot_get_predictor_vs_residual**(self):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Plot each predictive variable (component of X) against the residuals. For under the assumptions of linear regression, we expect the residuals to be distributed in a fixed width band around a flat line at x=0. See Shalizi § 1.2.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**plot_predictor_vs_squared_residual**(self):

&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;        

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**plot_residual_histogram**(self):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Plots the distribution of the residual quantities. For OLS regression, this distribution is expected to be a normal distribution centered at zero. The MLE fit of the residuals to a normal distribution is also plotted for reference. For more info, see Shalizi § 1.5.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**plot_residual_vs_residual**(self):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Plots the residuals vs a randomly shuffled copy of the residuals.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**show**(self):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Calls pyplot.show(), provided for convenience.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**test_noise_breusch_pagan**(self):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Runs the Breusch-Pagan test for heteroskedasticity on the residuals.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**test_residual_normality_dagostino_pearson**(self):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Runs the D'Agostino-Pearson normality test on the residuals.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RegressionDiagnostic.**test_residual_normality_shapiro_wilks**(self):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Runs the Shapiro-Wilks normality test on the residuals.

