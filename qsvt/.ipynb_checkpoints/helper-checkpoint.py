# Helper functions for verifying whether the result is correct.
# Loosely defined functions.
# May be replaced by numpy directly.

import numpy as np
import matplotlib.pyplot as plt


# Credit: "qsppack" (https://github.com/qsppack/QSPPACK)
def cvx_poly_coef(func, deg, opts=None):
    import cvxpy as cp
    """
    Find a polynomial approximation for a given function, using convex optimization.

    Input:
        func: The target function for which the polynomial approximation is sought.
        deg: The degree of the polynomial approximation.
        opts: A dictionary containing optional parameters for the function.

    Output:
        coef_full: The Chebyshev coefficients of the best polynomial approximation.
    """

    # Default options
    if opts is None:
        opts = {
            'npts': 200,
            'epsil': 0.01,
            'fscale': 1 - 0.01,
            'intervals': [0, 1],
            'isplot': False,
            'objnorm': np.inf
        }

    # Check variables and assign local variables
    assert len(opts['intervals']) % 2 == 0
    parity = deg % 2
    epsil = opts['epsil']
    npts = opts['npts']

    xpts = np.union1d(np.polynomial.chebyshev.chebpts1(2 * npts), opts['intervals'])
    xpts = xpts[xpts >= 0]
    npts = len(xpts)

    n_interval = len(opts['intervals']) // 2
    ind_union = []
    ind_set = {}
    for i in range(n_interval):
        ind_set[i] = np.where((xpts >= opts['intervals'][2 * i]) & (xpts <= opts['intervals'][2 * i + 1]))[0]
        ind_union = np.union1d(ind_union, ind_set[i])

    # Evaluate the target function
    fx = np.zeros(npts)


    ind_union = ind_union.astype(int)


    # print(f'ind_union: {ind_union}')
    fx[ind_union] = opts['fscale'] * func(xpts[ind_union])

    # Prepare the Chebyshev polynomials
    if parity == 0:
        n_coef = deg // 2 + 1
    else:
        n_coef = (deg + 1) // 2

    Ax = np.zeros((npts, n_coef))
    for k in range(1, n_coef + 1):
        if parity == 0:
            coef = [0] * (2 * (k - 1)) + [1]
            # Ax[:, k - 1] = cheby_calc(xpts, 2 * (k - 1))
            # Tcheb = np.polynomial.chebyshev.cheb2poly(2 * (k - 1))
        else:
            coef = [0] * (2 * k - 1) + [1]
            # Ax[:, k - 1] = cheby_calc(xpts, 2 * k - 1)
            # Tcheb = np.polynomial.chebyshev.cheb2poly(2 * k - 1)
        # print(f'Tcheb: {Tcheb}')
        # print(f'xpts: {xpts}')
        # Ax[:, k - 1] = Tcheb[xpts]
        Ax[:, k - 1] = np.polynomial.chebyshev.chebval(xpts, coef)

    # Use CVXPY to optimize the Chebyshev coefficients
    coef = cp.Variable(n_coef)
    y = cp.Variable(npts)

    # print(ind_union.shape)
    # print(ind_union)
    # print(ind_union)
    # y = cp.Variable()

    # print('I am half way!')

    # print(f'y: {y}')
    # print(f'fx: {fx}')
    # print(y, fx[ind_union], opts['objnorm'])
    # print(cp.norm(y - fx[ind_union], opts['objnorm']))


    objective = cp.Minimize(cp.norm(y[ind_union] - fx[ind_union], opts['objnorm']))
    constraints = [
        y == Ax @ coef,
        y >= -(1 - epsil),
        y <= (1 - epsil)
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    err_inf = np.linalg.norm(y[ind_union].value - fx[ind_union], opts['objnorm'])
    print(f'norm error = {err_inf}')

    # Use numpy.poly1d to make sure the maximum is less than 1
    coef_full = np.zeros(deg + 1)
    if parity == 0:
        coef_full[::2] = coef.value
    else:
        coef_full[1::2] = coef.value
    sol_cheb = np.poly1d(coef_full)
    max_sol = np.max(np.abs(sol_cheb(xpts)))
    print(f'max of solution = {max_sol}')
    if max_sol > 1.0 - 1e-10:
        raise ValueError('Solution is not bounded by 1. Increase npts')


    # Plot target polynomial
    if opts['isplot']:
        plt.figure(1)
        plt.clf()
        plt.plot(xpts, y.value, 'ro', linewidth=1.5)
        for i in range(n_interval):
            plt.plot(xpts[ind_set[i]], y.value[ind_set[i]], 'b-', linewidth=2)
        plt.xlabel('$x$', fontsize=15)
        plt.ylabel('$f(x)$', fontsize=15)
        plt.legend({'polynomial', 'target'}, fontsize=15)

        plt.figure(2)
        plt.clf()
        for i in range(n_interval):
            plt.plot(xpts[ind_set[i]], np.abs(y.value[ind_set[i]] - fx[ind_set[i]]), 'k-', linewidth=1.5)
        plt.xlabel('$x$', fontsize=15)
        plt.ylabel('$|f_{poly}(x) - f(x)|$', fontsize=15)
        plt.show()

    return coef_full


def total_variation(P: np.ndarray, Q: np.ndarray):
    return np.sum(np.abs(P - Q)) / 2


############# Warning! The functions covered runs extremely slow! ########################
def cheby(n: int) -> list:
    """
        Description & Return:
            Return the coefficients of the Chebyshev polynomial of the first kind (Tn(x)).
            (Descending order: [x_n, x_{n-1}, ..., x_0])
        Args:
            n: The order of T (Chebyshev polynomial) requested.
    """
    if n == 0: return [1]
    if n == 1: return [1, 0]

    Tn_1 = cheby(n - 1)
    Tn_2 = cheby(n - 2)

    Tn_1.append(0)

    Tn_2.insert(0, 0)
    Tn_2.insert(0, 0)

    # Tn = []
    # for i in range(n + 1):
    #     Tn.append(2 * Tn_1[i] - Tn_2[i])
    Tn = [2 * Tn_1[i] - Tn_2[i] for i in range(n + 1)]
    
    return Tn


def cheby_calc(S: list, n: int) -> list:
    """
        Description & Return:
            Given a list "S" of (complex) numbers, return a list 
            that evaluate "S" by "Tn(x)" (Chebyshev polynomial), element-wise.
        Args:
            S: Input list, consisting of numbers.
            n: The order of T (Chebyshev polynomial) requested.
    """
    coeff = cheby(n)
    # fS = []
    # for x in S:
    #     fS.append(np.polyval(coeff, x))
    fS = [np.polyval(coeff, x) for x in S]
    # fS = np.array([np.polyval(coeff, x) for x in S])
    return fS


def coskx(S, k):
    """
        Description & Return:
            Given a list "S" of (complex) numbers, return a list 
            that evaluate "S" by "cos(kx)", element-wise.
        Args:
            S: Input list, consisting of numbers.
            k: The parameter for "cos(kx)".
    """
    k = float(k)
    # fS = []
    # for x in S:
    #     fS.append(np.cos(k * x))
    fS = np.array([np.cos(k * x) for x in S])
    return fS


def sinkx(S, k):
    """
        Description & Return:
            Given a list "S" of (complex) numbers, return a list 
            that evaluate "S" by "sin(kx)", element-wise.
        Args:
            S: Input list, consisting of numbers.
            k: The parameter for "sin(kx)".
    """
    k = float(k)
    # fS = []
    # for x in S:
    #     fS.append(np.sin(k * x))
    fS = np.array([np.sin(k * x) for x in S])
    return fS

##############################################################################