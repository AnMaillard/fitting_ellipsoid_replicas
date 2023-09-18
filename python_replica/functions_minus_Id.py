# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement, import-outside-toplevel

#This file covers the case of \min |S - Id|_\gamma

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar, root

def sc(x):
    #Standard semicircular density
    if np.abs(x) >= 2:
        return 0.
    else:
        return np.sqrt(4. - x**2) / (2.*np.pi)

#Only for x > 0, for x < 0 we get it via symmetry
def nu(x, gamma_, h, q):
    return (2.*gamma_*(gamma_-1)*x**(gamma_-2) / np.sqrt(q) + h*np.sqrt(q))*sc(2*gamma_*x**(gamma_-1)/np.sqrt(q) + h*np.sqrt(q)*x)

#The quantities in the case gamma = 2, that can be computed analytically
def h_gamma_2(alpha_):
    return (1./4) - alpha_*(1.-alpha_)

def q_gamma_2(alpha_):
    return 32.*alpha_ / (1.-2*alpha_)**3

#Given gamma, h and q, find the boundary of the support of nu.
#Recall that the support of nu is symmetric.
def bound(gamma_, h, q):
    def to_zero_bound(x):
        if x <= 0:
            return -2 + np.sqrt(q)*h*x #Just to push the gradient so that there is no negative solution. When gamma > 2 the derivative is continuous in x = 0
        else: #Then x > 0
            return 2*gamma_*x**(gamma_-1)/np.sqrt(q) + h*np.sqrt(q)*x - 2

    sol = root_scalar(to_zero_bound, x0 = 0)
    return sol.root

def moment(gamma_, h, q, k):
    #The absolute k-th moment of nu: EE|X|^k
    BOUND = bound(gamma_, h, q) #The boundary
    value, _ = quad(lambda x: x**k * nu(x, gamma_, h, q), 0, BOUND)
    return 2*value #Factor 2 because of the symmetry

def to_zero(x, alpha_, gamma_):
    h, q = x[0], x[1]
    moment_2 = moment(gamma_, h, q, 2)
    moment_gamma = moment(gamma_, h, q, gamma_)

    return [h - alpha_ / (gamma_*moment_gamma*(1+moment_2)), (1./q) - alpha_ / (2*gamma_**2*(1+moment_2)*moment_gamma**2)]
    #I find by experience that this is the best way to solve these equations (invert the equation in q)

def run(alpha_, gamma_, starting_ = None):
    if starting_ is None:
        starting_ = {'h':h_gamma_2(alpha_), 'q':q_gamma_2(alpha_)}

    q_s, h_s = starting_['q'], starting_['h']
    sol = root(to_zero, [h_s, q_s], args=(alpha_, gamma_,), method = 'lm')

    return {'h':sol.x[0], 'q':sol.x[1], 'lmax':bound(gamma_, sol.x[0], sol.x[1])}
