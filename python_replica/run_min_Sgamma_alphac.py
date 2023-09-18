# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement, import-outside-toplevel

import argparse, time, pickle
from tqdm import tqdm
import numpy as np
from scipy.optimize import root_scalar
from functions import run

"""
Solve the replica equations for the minimal Schatten-gamma norm solution in the ellipsoid fitting problem. 
Here we use a root finding solver to find alphac

This is for the case min_{S in Gamma}||S||_gamma.

We parametrize the measure mu by the three parameters h, q, lambdam (q is denoted hat{q}_0 in the draft, lambdam is denoted lambda1 in the draft)
F_mu(x) = F_{sc}[2*gamma*|x|^(gamma)/(x*Sqrt[q]) + h*Sqrt[q]*x + 2 lambda_m / sqrt(q)] where F_{sc} is the CDF of the standard semicircle. 
This file makes a solver to find the value of alpha_c(gamma), starting from gamma = 2 for which alpha_c = 1/10
"""

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Iterator for the min - Schatten gamma Norm solution for ellipsoid fitting')
    parser.add_argument("--gammas", nargs="+", type = float)#gamma_min, gamma_max, and the NB of points
    parser.add_argument("--save", type = int, default = 0) #0 or 1
    parser.add_argument("--verbosity", type = int, default = 1)
    args = parser.parse_args()

    global_seed = int(time.time())
    np.random.seed(global_seed)
    verbosity = args.verbosity

    assert np.size(args.gammas) in [1,3]
    if np.size(args.gammas) == 3:
        gammas = np.linspace(args.gammas[0], args.gammas[1], num = int(args.gammas[2]), endpoint=True)
    else:
        gammas = np.array([args.gammas[0]])

    #Now we do an annealing in gammas, and look for alpha =  alphac
    alphacs, qs, hs, lambdams, lmins, lmaxs = np.zeros_like(gammas), np.zeros_like(gammas), np.zeros_like(gammas), np.zeros_like(gammas), np.zeros_like(gammas), np.zeros_like(gammas)
    starting = None
    alpha_starting = 0.1
    for i_g in tqdm(range(np.size(gammas)),leave=False):
        gamma = gammas[i_g]
        def to_zero_alpha(alpha):
            return run(alpha, gamma, starting)['lmin']

        sol = root_scalar(to_zero_alpha, x0 = alpha_starting, method = 'secant')
        alphacs[i_g] = sol.root
        results = run(alphacs[i_g], gamma, starting)
        qs[i_g] = results['q']
        hs[i_g] = results['h']
        lambdams[i_g] = results['lambdam']
        lmaxs[i_g] = results['lmax']
        lmins[i_g] = results['lmin']
        starting = {'q': qs[i_g], 'h':hs[i_g], 'lambdam':lambdams[i_g]}
        alpha_starting = alphacs[i_g]

        if verbosity >= 2:
            tqdm.write(f"For gamma = {gamma}, I found alphac = {alphacs[i_g]}")

    if bool(args.save):
        output_ = {'alphacs':alphacs, 'gammas':gammas, 'qs':qs, 'hs':hs, 'lambdams':lambdams, 'lmins':lmins, 'lmaxs':lmaxs}
        filename_ = f"Data/Min_Sgamma_alphac_gamma0_{gammas[0]}_gamma1_{gammas[-1]}.pkl"
        outfile_ = open(filename_,'wb')
        pickle.dump(output_,outfile_)
        outfile_.close()
