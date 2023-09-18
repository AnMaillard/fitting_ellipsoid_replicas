# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement, import-outside-toplevel

#This file computes the solution to the problem min |S - Id|_op.

import numpy as np
import cvxpy as cv

np.random.seed(1)

alphas = [0.15, 0.1892, 0.2]
trials = 50
d = 100

for alpha_ix, alpha in enumerate(alphas):
    n = int(alpha * d ** 2)
    alpha_str = str(alpha).replace('.', '_')
    
    X_var = cv.Variable((d, d), symmetric=True)
    V_par = cv.Parameter((d, n))

    obj = cv.Minimize(cv.norm(X_var - np.eye(d), 2))
    constraints = [cv.quad_form(V_par[:, i], X_var) == 1 for i in range(n)]

    prob = cv.Problem(obj, constraints)
    
    for ix in range(trials):
        print()
        print('***', alpha, ix, '***')
        print()
        V = np.random.normal(size=(d, n)) / np.sqrt(d)
        
        V_par.value = V
        value = prob.solve(verbose=True)
        X = X_var.value
        evals = np.linalg.eigvalsh(X)
        
        filename = '../Data/finite_size/evals__dist_op' + '__d' + str(d) + '__a' + alpha_str + '__t' + str(ix) + '.csv'
        print(filename)
        
        np.savetxt(filename, evals, delimiter=',')
