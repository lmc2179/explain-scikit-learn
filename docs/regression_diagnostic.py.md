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

&nbsp;&nbsp;&nbsp;&nbsp;**deepcopy**(x, memo=None, _nil=[]):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Deep copy operation on arbitrary Python objects.

&nbsp;&nbsp;&nbsp;&nbsp;    See the module's __doc__ string for more info.
&nbsp;&nbsp;&nbsp;&nbsp;    

&nbsp;&nbsp;&nbsp;&nbsp;**normaltest**(a, axis=0):


&nbsp;&nbsp;&nbsp;&nbsp;    Tests whether a sample differs from a normal distribution.

&nbsp;&nbsp;&nbsp;&nbsp;    This function tests the null hypothesis that a sample comes
&nbsp;&nbsp;&nbsp;&nbsp;    from a normal distribution.  It is based on D'Agostino and
&nbsp;&nbsp;&nbsp;&nbsp;    Pearson's [1]_, [2]_ test that combines skew and kurtosis to
&nbsp;&nbsp;&nbsp;&nbsp;    produce an omnibus test of normality.

&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;    Parameters
&nbsp;&nbsp;&nbsp;&nbsp;    ----------
&nbsp;&nbsp;&nbsp;&nbsp;    a : array_like
&nbsp;&nbsp;&nbsp;&nbsp;        The array containing the data to be tested.
&nbsp;&nbsp;&nbsp;&nbsp;    axis : int or None, optional
&nbsp;&nbsp;&nbsp;&nbsp;        Axis along which to compute test. Default is 0. If None,
&nbsp;&nbsp;&nbsp;&nbsp;        compute over the whole array `a`.

&nbsp;&nbsp;&nbsp;&nbsp;    Returns
&nbsp;&nbsp;&nbsp;&nbsp;    -------
&nbsp;&nbsp;&nbsp;&nbsp;    statistic : float or array
&nbsp;&nbsp;&nbsp;&nbsp;        `s^2 + k^2`, where `s` is the z-score returned by `skewtest` and
&nbsp;&nbsp;&nbsp;&nbsp;        `k` is the z-score returned by `kurtosistest`.
&nbsp;&nbsp;&nbsp;&nbsp;    pvalue : float or array
&nbsp;&nbsp;&nbsp;&nbsp;       A 2-sided chi squared probability for the hypothesis test.

&nbsp;&nbsp;&nbsp;&nbsp;    References
&nbsp;&nbsp;&nbsp;&nbsp;    ----------
&nbsp;&nbsp;&nbsp;&nbsp;    .. [1] D'Agostino, R. B. (1971), "An omnibus test of normality for
&nbsp;&nbsp;&nbsp;&nbsp;           moderate and large sample size," Biometrika, 58, 341-348

&nbsp;&nbsp;&nbsp;&nbsp;    .. [2] D'Agostino, R. and Pearson, E. S. (1973), "Testing for
&nbsp;&nbsp;&nbsp;&nbsp;           departures from normality," Biometrika, 60, 613-622

&nbsp;&nbsp;&nbsp;&nbsp;    

&nbsp;&nbsp;&nbsp;&nbsp;**quad**(func, a, b, args=(), full_output=0, epsabs=1.49e-08, epsrel=1.49e-08, limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50, limlst=50):


&nbsp;&nbsp;&nbsp;&nbsp;    Compute a definite integral.

&nbsp;&nbsp;&nbsp;&nbsp;    Integrate func from `a` to `b` (possibly infinite interval) using a
&nbsp;&nbsp;&nbsp;&nbsp;    technique from the Fortran library QUADPACK.

&nbsp;&nbsp;&nbsp;&nbsp;    Parameters
&nbsp;&nbsp;&nbsp;&nbsp;    ----------
&nbsp;&nbsp;&nbsp;&nbsp;    func : function
&nbsp;&nbsp;&nbsp;&nbsp;        A Python function or method to integrate.  If `func` takes many
&nbsp;&nbsp;&nbsp;&nbsp;        arguments, it is integrated along the axis corresponding to the
&nbsp;&nbsp;&nbsp;&nbsp;        first argument.
&nbsp;&nbsp;&nbsp;&nbsp;        If the user desires improved integration performance, then f may
&nbsp;&nbsp;&nbsp;&nbsp;        instead be a ``ctypes`` function of the form:

&nbsp;&nbsp;&nbsp;&nbsp;            f(int n, double args[n]),

&nbsp;&nbsp;&nbsp;&nbsp;        where ``args`` is an array of function arguments and ``n`` is the
&nbsp;&nbsp;&nbsp;&nbsp;        length of ``args``. ``f.argtypes`` should be set to
&nbsp;&nbsp;&nbsp;&nbsp;        ``(c_int, c_double)``, and ``f.restype`` should be ``(c_double,)``.
&nbsp;&nbsp;&nbsp;&nbsp;    a : float
&nbsp;&nbsp;&nbsp;&nbsp;        Lower limit of integration (use -numpy.inf for -infinity).
&nbsp;&nbsp;&nbsp;&nbsp;    b : float
&nbsp;&nbsp;&nbsp;&nbsp;        Upper limit of integration (use numpy.inf for +infinity).
&nbsp;&nbsp;&nbsp;&nbsp;    args : tuple, optional
&nbsp;&nbsp;&nbsp;&nbsp;        Extra arguments to pass to `func`.
&nbsp;&nbsp;&nbsp;&nbsp;    full_output : int, optional
&nbsp;&nbsp;&nbsp;&nbsp;        Non-zero to return a dictionary of integration information.
&nbsp;&nbsp;&nbsp;&nbsp;        If non-zero, warning messages are also suppressed and the
&nbsp;&nbsp;&nbsp;&nbsp;        message is appended to the output tuple.

&nbsp;&nbsp;&nbsp;&nbsp;    Returns
&nbsp;&nbsp;&nbsp;&nbsp;    -------
&nbsp;&nbsp;&nbsp;&nbsp;    y : float
&nbsp;&nbsp;&nbsp;&nbsp;        The integral of func from `a` to `b`.
&nbsp;&nbsp;&nbsp;&nbsp;    abserr : float
&nbsp;&nbsp;&nbsp;&nbsp;        An estimate of the absolute error in the result.
&nbsp;&nbsp;&nbsp;&nbsp;    infodict : dict
&nbsp;&nbsp;&nbsp;&nbsp;        A dictionary containing additional information.
&nbsp;&nbsp;&nbsp;&nbsp;        Run scipy.integrate.quad_explain() for more information.
&nbsp;&nbsp;&nbsp;&nbsp;    message :
&nbsp;&nbsp;&nbsp;&nbsp;        A convergence message.
&nbsp;&nbsp;&nbsp;&nbsp;    explain :
&nbsp;&nbsp;&nbsp;&nbsp;        Appended only with 'cos' or 'sin' weighting and infinite
&nbsp;&nbsp;&nbsp;&nbsp;        integration limits, it contains an explanation of the codes in
&nbsp;&nbsp;&nbsp;&nbsp;        infodict['ierlst']

&nbsp;&nbsp;&nbsp;&nbsp;    Other Parameters
&nbsp;&nbsp;&nbsp;&nbsp;    ----------------
&nbsp;&nbsp;&nbsp;&nbsp;    epsabs : float or int, optional
&nbsp;&nbsp;&nbsp;&nbsp;        Absolute error tolerance.
&nbsp;&nbsp;&nbsp;&nbsp;    epsrel : float or int, optional
&nbsp;&nbsp;&nbsp;&nbsp;        Relative error tolerance.
&nbsp;&nbsp;&nbsp;&nbsp;    limit : float or int, optional
&nbsp;&nbsp;&nbsp;&nbsp;        An upper bound on the number of subintervals used in the adaptive
&nbsp;&nbsp;&nbsp;&nbsp;        algorithm.
&nbsp;&nbsp;&nbsp;&nbsp;    points : (sequence of floats,ints), optional
&nbsp;&nbsp;&nbsp;&nbsp;        A sequence of break points in the bounded integration interval
&nbsp;&nbsp;&nbsp;&nbsp;        where local difficulties of the integrand may occur (e.g.,
&nbsp;&nbsp;&nbsp;&nbsp;        singularities, discontinuities). The sequence does not have
&nbsp;&nbsp;&nbsp;&nbsp;        to be sorted.
&nbsp;&nbsp;&nbsp;&nbsp;    weight : float or int, optional
&nbsp;&nbsp;&nbsp;&nbsp;        String indicating weighting function. Full explanation for this
&nbsp;&nbsp;&nbsp;&nbsp;        and the remaining arguments can be found below.
&nbsp;&nbsp;&nbsp;&nbsp;    wvar : optional
&nbsp;&nbsp;&nbsp;&nbsp;        Variables for use with weighting functions.
&nbsp;&nbsp;&nbsp;&nbsp;    wopts : optional
&nbsp;&nbsp;&nbsp;&nbsp;        Optional input for reusing Chebyshev moments.
&nbsp;&nbsp;&nbsp;&nbsp;    maxp1 : float or int, optional
&nbsp;&nbsp;&nbsp;&nbsp;        An upper bound on the number of Chebyshev moments.
&nbsp;&nbsp;&nbsp;&nbsp;    limlst : int, optional
&nbsp;&nbsp;&nbsp;&nbsp;        Upper bound on the number of cycles (>=3) for use with a sinusoidal
&nbsp;&nbsp;&nbsp;&nbsp;        weighting and an infinite end-point.

&nbsp;&nbsp;&nbsp;&nbsp;    See Also
&nbsp;&nbsp;&nbsp;&nbsp;    --------
&nbsp;&nbsp;&nbsp;&nbsp;    dblquad : double integral
&nbsp;&nbsp;&nbsp;&nbsp;    tplquad : triple integral
&nbsp;&nbsp;&nbsp;&nbsp;    nquad : n-dimensional integrals (uses `quad` recursively)
&nbsp;&nbsp;&nbsp;&nbsp;    fixed_quad : fixed-order Gaussian quadrature
&nbsp;&nbsp;&nbsp;&nbsp;    quadrature : adaptive Gaussian quadrature
&nbsp;&nbsp;&nbsp;&nbsp;    odeint : ODE integrator
&nbsp;&nbsp;&nbsp;&nbsp;    ode : ODE integrator
&nbsp;&nbsp;&nbsp;&nbsp;    simps : integrator for sampled data
&nbsp;&nbsp;&nbsp;&nbsp;    romb : integrator for sampled data
&nbsp;&nbsp;&nbsp;&nbsp;    scipy.special : for coefficients and roots of orthogonal polynomials

&nbsp;&nbsp;&nbsp;&nbsp;    Notes
&nbsp;&nbsp;&nbsp;&nbsp;    -----

&nbsp;&nbsp;&nbsp;&nbsp;    **Extra information for quad() inputs and outputs**

&nbsp;&nbsp;&nbsp;&nbsp;    If full_output is non-zero, then the third output argument
&nbsp;&nbsp;&nbsp;&nbsp;    (infodict) is a dictionary with entries as tabulated below.  For
&nbsp;&nbsp;&nbsp;&nbsp;    infinite limits, the range is transformed to (0,1) and the
&nbsp;&nbsp;&nbsp;&nbsp;    optional outputs are given with respect to this transformed range.
&nbsp;&nbsp;&nbsp;&nbsp;    Let M be the input argument limit and let K be infodict['last'].
&nbsp;&nbsp;&nbsp;&nbsp;    The entries are:

&nbsp;&nbsp;&nbsp;&nbsp;    'neval'
&nbsp;&nbsp;&nbsp;&nbsp;        The number of function evaluations.
&nbsp;&nbsp;&nbsp;&nbsp;    'last'
&nbsp;&nbsp;&nbsp;&nbsp;        The number, K, of subintervals produced in the subdivision process.
&nbsp;&nbsp;&nbsp;&nbsp;    'alist'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 array of length M, the first K elements of which are the
&nbsp;&nbsp;&nbsp;&nbsp;        left end points of the subintervals in the partition of the
&nbsp;&nbsp;&nbsp;&nbsp;        integration range.
&nbsp;&nbsp;&nbsp;&nbsp;    'blist'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 array of length M, the first K elements of which are the
&nbsp;&nbsp;&nbsp;&nbsp;        right end points of the subintervals.
&nbsp;&nbsp;&nbsp;&nbsp;    'rlist'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 array of length M, the first K elements of which are the
&nbsp;&nbsp;&nbsp;&nbsp;        integral approximations on the subintervals.
&nbsp;&nbsp;&nbsp;&nbsp;    'elist'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 array of length M, the first K elements of which are the
&nbsp;&nbsp;&nbsp;&nbsp;        moduli of the absolute error estimates on the subintervals.
&nbsp;&nbsp;&nbsp;&nbsp;    'iord'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 integer array of length M, the first L elements of
&nbsp;&nbsp;&nbsp;&nbsp;        which are pointers to the error estimates over the subintervals
&nbsp;&nbsp;&nbsp;&nbsp;        with ``L=K`` if ``K<=M/2+2`` or ``L=M+1-K`` otherwise. Let I be the
&nbsp;&nbsp;&nbsp;&nbsp;        sequence ``infodict['iord']`` and let E be the sequence
&nbsp;&nbsp;&nbsp;&nbsp;        ``infodict['elist']``.  Then ``E[I[1]], ..., E[I[L]]`` forms a
&nbsp;&nbsp;&nbsp;&nbsp;        decreasing sequence.

&nbsp;&nbsp;&nbsp;&nbsp;    If the input argument points is provided (i.e. it is not None),
&nbsp;&nbsp;&nbsp;&nbsp;    the following additional outputs are placed in the output
&nbsp;&nbsp;&nbsp;&nbsp;    dictionary.  Assume the points sequence is of length P.

&nbsp;&nbsp;&nbsp;&nbsp;    'pts'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 array of length P+2 containing the integration limits
&nbsp;&nbsp;&nbsp;&nbsp;        and the break points of the intervals in ascending order.
&nbsp;&nbsp;&nbsp;&nbsp;        This is an array giving the subintervals over which integration
&nbsp;&nbsp;&nbsp;&nbsp;        will occur.
&nbsp;&nbsp;&nbsp;&nbsp;    'level'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 integer array of length M (=limit), containing the
&nbsp;&nbsp;&nbsp;&nbsp;        subdivision levels of the subintervals, i.e., if (aa,bb) is a
&nbsp;&nbsp;&nbsp;&nbsp;        subinterval of ``(pts[1], pts[2])`` where ``pts[0]`` and ``pts[2]``
&nbsp;&nbsp;&nbsp;&nbsp;        are adjacent elements of ``infodict['pts']``, then (aa,bb) has level l
&nbsp;&nbsp;&nbsp;&nbsp;        if ``|bb-aa| = |pts[2]-pts[1]| * 2**(-l)``.
&nbsp;&nbsp;&nbsp;&nbsp;    'ndin'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 integer array of length P+2.  After the first integration
&nbsp;&nbsp;&nbsp;&nbsp;        over the intervals (pts[1], pts[2]), the error estimates over some
&nbsp;&nbsp;&nbsp;&nbsp;        of the intervals may have been increased artificially in order to
&nbsp;&nbsp;&nbsp;&nbsp;        put their subdivision forward.  This array has ones in slots
&nbsp;&nbsp;&nbsp;&nbsp;        corresponding to the subintervals for which this happens.

&nbsp;&nbsp;&nbsp;&nbsp;    **Weighting the integrand**

&nbsp;&nbsp;&nbsp;&nbsp;    The input variables, *weight* and *wvar*, are used to weight the
&nbsp;&nbsp;&nbsp;&nbsp;    integrand by a select list of functions.  Different integration
&nbsp;&nbsp;&nbsp;&nbsp;    methods are used to compute the integral with these weighting
&nbsp;&nbsp;&nbsp;&nbsp;    functions.  The possible values of weight and the corresponding
&nbsp;&nbsp;&nbsp;&nbsp;    weighting functions are.

&nbsp;&nbsp;&nbsp;&nbsp;    ==========  ===================================   =====================
&nbsp;&nbsp;&nbsp;&nbsp;    ``weight``  Weight function used                  ``wvar``
&nbsp;&nbsp;&nbsp;&nbsp;    ==========  ===================================   =====================
&nbsp;&nbsp;&nbsp;&nbsp;    'cos'       cos(w*x)                              wvar = w
&nbsp;&nbsp;&nbsp;&nbsp;    'sin'       sin(w*x)                              wvar = w
&nbsp;&nbsp;&nbsp;&nbsp;    'alg'       g(x) = ((x-a)**alpha)*((b-x)**beta)   wvar = (alpha, beta)
&nbsp;&nbsp;&nbsp;&nbsp;    'alg-loga'  g(x)*log(x-a)                         wvar = (alpha, beta)
&nbsp;&nbsp;&nbsp;&nbsp;    'alg-logb'  g(x)*log(b-x)                         wvar = (alpha, beta)
&nbsp;&nbsp;&nbsp;&nbsp;    'alg-log'   g(x)*log(x-a)*log(b-x)                wvar = (alpha, beta)
&nbsp;&nbsp;&nbsp;&nbsp;    'cauchy'    1/(x-c)                               wvar = c
&nbsp;&nbsp;&nbsp;&nbsp;    ==========  ===================================   =====================

&nbsp;&nbsp;&nbsp;&nbsp;    wvar holds the parameter w, (alpha, beta), or c depending on the weight
&nbsp;&nbsp;&nbsp;&nbsp;    selected.  In these expressions, a and b are the integration limits.

&nbsp;&nbsp;&nbsp;&nbsp;    For the 'cos' and 'sin' weighting, additional inputs and outputs are
&nbsp;&nbsp;&nbsp;&nbsp;    available.

&nbsp;&nbsp;&nbsp;&nbsp;    For finite integration limits, the integration is performed using a
&nbsp;&nbsp;&nbsp;&nbsp;    Clenshaw-Curtis method which uses Chebyshev moments.  For repeated
&nbsp;&nbsp;&nbsp;&nbsp;    calculations, these moments are saved in the output dictionary:

&nbsp;&nbsp;&nbsp;&nbsp;    'momcom'
&nbsp;&nbsp;&nbsp;&nbsp;        The maximum level of Chebyshev moments that have been computed,
&nbsp;&nbsp;&nbsp;&nbsp;        i.e., if ``M_c`` is ``infodict['momcom']`` then the moments have been
&nbsp;&nbsp;&nbsp;&nbsp;        computed for intervals of length ``|b-a| * 2**(-l)``,
&nbsp;&nbsp;&nbsp;&nbsp;        ``l=0,1,...,M_c``.
&nbsp;&nbsp;&nbsp;&nbsp;    'nnlog'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 integer array of length M(=limit), containing the
&nbsp;&nbsp;&nbsp;&nbsp;        subdivision levels of the subintervals, i.e., an element of this
&nbsp;&nbsp;&nbsp;&nbsp;        array is equal to l if the corresponding subinterval is
&nbsp;&nbsp;&nbsp;&nbsp;        ``|b-a|* 2**(-l)``.
&nbsp;&nbsp;&nbsp;&nbsp;    'chebmo'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-2 array of shape (25, maxp1) containing the computed
&nbsp;&nbsp;&nbsp;&nbsp;        Chebyshev moments.  These can be passed on to an integration
&nbsp;&nbsp;&nbsp;&nbsp;        over the same interval by passing this array as the second
&nbsp;&nbsp;&nbsp;&nbsp;        element of the sequence wopts and passing infodict['momcom'] as
&nbsp;&nbsp;&nbsp;&nbsp;        the first element.

&nbsp;&nbsp;&nbsp;&nbsp;    If one of the integration limits is infinite, then a Fourier integral is
&nbsp;&nbsp;&nbsp;&nbsp;    computed (assuming w neq 0).  If full_output is 1 and a numerical error
&nbsp;&nbsp;&nbsp;&nbsp;    is encountered, besides the error message attached to the output tuple,
&nbsp;&nbsp;&nbsp;&nbsp;    a dictionary is also appended to the output tuple which translates the
&nbsp;&nbsp;&nbsp;&nbsp;    error codes in the array ``info['ierlst']`` to English messages.  The
&nbsp;&nbsp;&nbsp;&nbsp;    output information dictionary contains the following entries instead of
&nbsp;&nbsp;&nbsp;&nbsp;    'last', 'alist', 'blist', 'rlist', and 'elist':

&nbsp;&nbsp;&nbsp;&nbsp;    'lst'
&nbsp;&nbsp;&nbsp;&nbsp;        The number of subintervals needed for the integration (call it ``K_f``).
&nbsp;&nbsp;&nbsp;&nbsp;    'rslst'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 array of length M_f=limlst, whose first ``K_f`` elements
&nbsp;&nbsp;&nbsp;&nbsp;        contain the integral contribution over the interval
&nbsp;&nbsp;&nbsp;&nbsp;        ``(a+(k-1)c, a+kc)`` where ``c = (2*floor(|w|) + 1) * pi / |w|``
&nbsp;&nbsp;&nbsp;&nbsp;        and ``k=1,2,...,K_f``.
&nbsp;&nbsp;&nbsp;&nbsp;    'erlst'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 array of length ``M_f`` containing the error estimate
&nbsp;&nbsp;&nbsp;&nbsp;        corresponding to the interval in the same position in
&nbsp;&nbsp;&nbsp;&nbsp;        ``infodict['rslist']``.
&nbsp;&nbsp;&nbsp;&nbsp;    'ierlst'
&nbsp;&nbsp;&nbsp;&nbsp;        A rank-1 integer array of length ``M_f`` containing an error flag
&nbsp;&nbsp;&nbsp;&nbsp;        corresponding to the interval in the same position in
&nbsp;&nbsp;&nbsp;&nbsp;        ``infodict['rslist']``.  See the explanation dictionary (last entry
&nbsp;&nbsp;&nbsp;&nbsp;        in the output tuple) for the meaning of the codes.

&nbsp;&nbsp;&nbsp;&nbsp;    Examples
&nbsp;&nbsp;&nbsp;&nbsp;    --------
&nbsp;&nbsp;&nbsp;&nbsp;    Calculate :math:`\int^4_0 x^2 dx` and compare with an analytic result

&nbsp;&nbsp;&nbsp;&nbsp;    >>> from scipy import integrate
&nbsp;&nbsp;&nbsp;&nbsp;    >>> x2 = lambda x: x**2
&nbsp;&nbsp;&nbsp;&nbsp;    >>> integrate.quad(x2, 0, 4)
&nbsp;&nbsp;&nbsp;&nbsp;    (21.333333333333332, 2.3684757858670003e-13)
&nbsp;&nbsp;&nbsp;&nbsp;    >>> print(4**3 / 3.)  # analytical result
&nbsp;&nbsp;&nbsp;&nbsp;    21.3333333333

&nbsp;&nbsp;&nbsp;&nbsp;    Calculate :math:`\int^\infty_0 e^{-x} dx`

&nbsp;&nbsp;&nbsp;&nbsp;    >>> invexp = lambda x: np.exp(-x)
&nbsp;&nbsp;&nbsp;&nbsp;    >>> integrate.quad(invexp, 0, np.inf)
&nbsp;&nbsp;&nbsp;&nbsp;    (1.0, 5.842605999138044e-11)

&nbsp;&nbsp;&nbsp;&nbsp;    >>> f = lambda x,a : a*x
&nbsp;&nbsp;&nbsp;&nbsp;    >>> y, err = integrate.quad(f, 0, 1, args=(1,))
&nbsp;&nbsp;&nbsp;&nbsp;    >>> y
&nbsp;&nbsp;&nbsp;&nbsp;    0.5
&nbsp;&nbsp;&nbsp;&nbsp;    >>> y, err = integrate.quad(f, 0, 1, args=(3,))
&nbsp;&nbsp;&nbsp;&nbsp;    >>> y
&nbsp;&nbsp;&nbsp;&nbsp;    1.5

&nbsp;&nbsp;&nbsp;&nbsp;    Calculate :math:`\int^1_0 x^2 + y^2 dx` with ctypes, holding
&nbsp;&nbsp;&nbsp;&nbsp;    y parameter as 1::

&nbsp;&nbsp;&nbsp;&nbsp;        testlib.c =>
&nbsp;&nbsp;&nbsp;&nbsp;            double func(int n, double args[n]){
&nbsp;&nbsp;&nbsp;&nbsp;                return args[0]*args[0] + args[1]*args[1];}
&nbsp;&nbsp;&nbsp;&nbsp;        compile to library testlib.*

&nbsp;&nbsp;&nbsp;&nbsp;    >>> from scipy import integrate
&nbsp;&nbsp;&nbsp;&nbsp;    >>> import ctypes
&nbsp;&nbsp;&nbsp;&nbsp;    >>> lib = ctypes.CDLL('/home/.../testlib.*') #use absolute path
&nbsp;&nbsp;&nbsp;&nbsp;    >>> lib.func.restype = ctypes.c_double
&nbsp;&nbsp;&nbsp;&nbsp;    >>> lib.func.argtypes = (ctypes.c_int,ctypes.c_double)
&nbsp;&nbsp;&nbsp;&nbsp;    >>> integrate.quad(lib.func,0,1,(1))
&nbsp;&nbsp;&nbsp;&nbsp;    (1.3333333333333333, 1.4802973661668752e-14)
&nbsp;&nbsp;&nbsp;&nbsp;    >>> print((1.0**3/3.0 + 1.0) - (0.0**3/3.0 + 0.0)) #Analytic result
&nbsp;&nbsp;&nbsp;&nbsp;    1.3333333333333333

&nbsp;&nbsp;&nbsp;&nbsp;    

&nbsp;&nbsp;&nbsp;&nbsp;**shapiro**(x, a=None, reta=False):


&nbsp;&nbsp;&nbsp;&nbsp;    Perform the Shapiro-Wilk test for normality.

&nbsp;&nbsp;&nbsp;&nbsp;    The Shapiro-Wilk test tests the null hypothesis that the
&nbsp;&nbsp;&nbsp;&nbsp;    data was drawn from a normal distribution.

&nbsp;&nbsp;&nbsp;&nbsp;    Parameters
&nbsp;&nbsp;&nbsp;&nbsp;    ----------
&nbsp;&nbsp;&nbsp;&nbsp;    x : array_like
&nbsp;&nbsp;&nbsp;&nbsp;        Array of sample data.
&nbsp;&nbsp;&nbsp;&nbsp;    a : array_like, optional
&nbsp;&nbsp;&nbsp;&nbsp;        Array of internal parameters used in the calculation.  If these
&nbsp;&nbsp;&nbsp;&nbsp;        are not given, they will be computed internally.  If x has length
&nbsp;&nbsp;&nbsp;&nbsp;        n, then a must have length n/2.
&nbsp;&nbsp;&nbsp;&nbsp;    reta : bool, optional
&nbsp;&nbsp;&nbsp;&nbsp;        Whether or not to return the internally computed a values.  The
&nbsp;&nbsp;&nbsp;&nbsp;        default is False.

&nbsp;&nbsp;&nbsp;&nbsp;    Returns
&nbsp;&nbsp;&nbsp;&nbsp;    -------
&nbsp;&nbsp;&nbsp;&nbsp;    W : float
&nbsp;&nbsp;&nbsp;&nbsp;        The test statistic.
&nbsp;&nbsp;&nbsp;&nbsp;    p-value : float
&nbsp;&nbsp;&nbsp;&nbsp;        The p-value for the hypothesis test.
&nbsp;&nbsp;&nbsp;&nbsp;    a : array_like, optional
&nbsp;&nbsp;&nbsp;&nbsp;        If `reta` is True, then these are the internally computed "a"
&nbsp;&nbsp;&nbsp;&nbsp;        values that may be passed into this function on future calls.

&nbsp;&nbsp;&nbsp;&nbsp;    See Also
&nbsp;&nbsp;&nbsp;&nbsp;    --------
&nbsp;&nbsp;&nbsp;&nbsp;    anderson : The Anderson-Darling test for normality

&nbsp;&nbsp;&nbsp;&nbsp;    References
&nbsp;&nbsp;&nbsp;&nbsp;    ----------
&nbsp;&nbsp;&nbsp;&nbsp;    .. [1] http://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm

&nbsp;&nbsp;&nbsp;&nbsp;    

