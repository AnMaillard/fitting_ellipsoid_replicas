# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement, import-outside-toplevel

import argparse, time, pickle
from tqdm import tqdm
import numpy as np
from scipy.optimize import root_scalar
from functions_minus_Id import run

"""
Solve the replica equations for the minimal Schatten-gamma norm solution in the ellipsoid fitting problem. 
Here we use a root finding solver to find alphac

This is for the case min_{S in Gamma}||S-Id||_gamma

Here nu = \tilde{\nu} in the paper notations is the limit ESD of S - Id, and it is symmetric around 0

We parametrize the measure nu by the two parameters h and q (q is denoted hat{q}_0 in the latex file)
F_nu(x) = F_{sc}[2*gamma*|x|^(gamma)/(x*Sqrt[q]) + h*Sqrt[q]*x] where F_{sc} is the CDF of the standard semicircle. 
We have A = 2 gamma / sqrt(qh), B = h sqrt(qh) and C = 2 lambda_m / sqrt(qh), so A, B >= 0

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
        if args.gammas[1] < 2:
            gammas = 1.+np.exp(np.linspace(np.log(args.gammas[0]-1), np.log(args.gammas[1]-1.), num = int(args.gammas[2]), endpoint=True)) #A log scale for gamma - 1
        else:
            gammas = np.linspace(args.gammas[0], args.gammas[1], num = int(args.gammas[2]), endpoint=True) 
    else:
        gammas = np.array([args.gammas[0]])

    #Now we do an annealing in gammas, and look for alphac
    alphacs, qcs, hcs, lmaxcs = np.zeros_like(gammas), np.zeros_like(gammas), np.zeros_like(gammas), np.zeros_like(gammas)
    starting = None
    alpha_starting = 0.1
    for i_g in tqdm(range(np.size(gammas)),leave=False):
        gamma = gammas[i_g]
        def to_zero_alpha(alpha):
            return run(alpha, gamma, starting)['lmax'] - 1

        sol = root_scalar(to_zero_alpha, x0 = alpha_starting, method = 'secant')
        alphacs[i_g] = sol.root
        results = run(alphacs[i_g], gamma, starting)
        qcs[i_g] = results['q']
        hcs[i_g] = results['h']
        lmaxcs[i_g] = results['lmax']
        starting = {'q': qcs[i_g], 'h':hcs[i_g]}
        alpha_starting = alphacs[i_g]

        if verbosity >= 2:
            tqdm.write(f"For gamma = {gamma}, I found alphac = {alphacs[i_g]}")

    if bool(args.save):
        output_ = {'alphacs':alphacs, 'gammas':gammas, 'qcs':qcs, 'hcs':hcs, 'lmaxcs':lmaxcs}
        filename_ = f"Data/Min_Sgamma_minus_Id_alphac_gamma0_{gammas[0]}_gamma1_{gammas[-1]}.pkl"
        outfile_ = open(filename_,'wb')
        pickle.dump(output_,outfile_)
        outfile_.close()
