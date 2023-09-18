# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement, import-outside-toplevel

#This file covers the case of \min |S|_\gamma

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar, root

def sc(x):
    #Standard semicircular density
    if np.abs(x) >= 2:
        return 0.
    else:
        return np.sqrt(4. - x**2) / (2.*np.pi)

def mu(x, gamma_, h, q, lambdam):
    ax = np.abs(x)
    return (2.*gamma_*(gamma_-1)*ax**(gamma_-2) / np.sqrt(q) + h*np.sqrt(q))*sc(2*np.sign(x)*gamma_*ax**(gamma_-1)/np.sqrt(q) + h*np.sqrt(q)*x + 2*lambdam/np.sqrt(q))

#The quantities in the case gamma = 2, that can be computed analytically
def h_gamma_2(alpha_):
    return (1./4)*(1.-2*alpha_)**2

def q_gamma_2(alpha_):
    return 32.*alpha_ / (1.-2*alpha_)**3

def lambdam_gamma_2(alpha_):
    return - 2./(1.-2*alpha_)

#Given (gamma, h, q, lambdam) find the boundary of the support of mu.
def bounds(gamma_, h, q, lambdam):
    def to_zero_bound_max(x):
        ax = np.abs(x)
        return 2*np.sign(x)*gamma_*ax**(gamma_-1)/(np.sqrt(q)) + h*np.sqrt(q)*x + 2*lambdam/np.sqrt(q) - 2
    def to_zero_bound_min(x):
        ax = np.abs(x)
        return 2*np.sign(x)*gamma_*ax**(gamma_-1)/(np.sqrt(q)) + h*np.sqrt(q)*x + 2*lambdam/np.sqrt(q) + 2

    sol_max = root_scalar(to_zero_bound_max, x0 = 0)
    sol_min = root_scalar(to_zero_bound_min, x0 = 0)
    return sol_min.root, sol_max.root

def moment(gamma_, h, q, lambdam, k, abs_moment = True):
    #The (absolute) k-th moment of mu: EE X^k / EE|X|^k
    BOUND_MIN, BOUND_MAX = bounds(gamma_, h, q, lambdam) #The boundary
    total = 0
    if BOUND_MAX > 0:
        value_pos, _ = quad(lambda x: x**k * mu(x, gamma_, h, q, lambdam), max(0, BOUND_MIN), BOUND_MAX)
        total += value_pos
    if BOUND_MIN < 0:
        value_neg = 0
        if abs_moment:
            value_neg, _ = quad(lambda x: (-x)**k * mu(x, gamma_, h, q, lambdam), BOUND_MIN, min(BOUND_MAX, 0))
        else:
            value_neg, _ = quad(lambda x: x**k * mu(x, gamma_, h, q, lambdam), BOUND_MIN, min(BOUND_MAX, 0))
        total += value_neg
    return total


def to_zero(x, alpha_, gamma_):
    h, q, lambdam = x[0], x[1], x[2]
    moment_1 = moment(gamma_, h, q, lambdam, 1, abs_moment = False)
    moment_2 = moment(gamma_, h, q, lambdam, 2, abs_moment = True)
    moment_gamma = moment(gamma_, h, q, lambdam, gamma_, abs_moment = True)

    return [(1./q) - (h**2/(2*alpha_))*moment_2, moment_1 - 1., lambdam + gamma_*moment_gamma]

def run(alpha_, gamma_, starting_ = None):
    if starting_ is None:
        starting_ = {'h':h_gamma_2(alpha_), 'q':q_gamma_2(alpha_), 'lambdam':lambdam_gamma_2(alpha_)}

    q_s, h_s, lambdam_s = starting_['q'], starting_['h'], starting_['lambdam']
    sol = root(to_zero, [h_s, q_s, lambdam_s], args=(alpha_, gamma_,), method = 'lm')
    lmin, lmax = bounds(gamma_, sol.x[0], sol.x[1], sol.x[2])

    return {'h':sol.x[0], 'q':sol.x[1], 'lambdam':sol.x[2], 'lmin':lmin, 'lmax':lmax}
