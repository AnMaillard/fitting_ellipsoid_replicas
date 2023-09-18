# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement, import-outside-toplevel

import argparse, time, pickle
from tqdm import tqdm
import numpy as np
from functions_minus_Id import run, bound

"""
Solve the replica equations for the minimal Schatten-gamma norm solution in the ellipsoid fitting problem. 
This gives the spectra for a fixed value of (alpha, gamma)

This is for the case min_{S in Gamma}||S-Id||_gamma

Here nu is the limit ESD of S - Id, and it is symmetric around 0

We parametrize the measure nu by the two parameters h and q (q is denoted hat{q}_0 in the latex file)
F_nu(x) = F_{sc}[2*gamma*|x|^(gamma)/(x*Sqrt[q]) + h*Sqrt[q]*x] where F_{sc} is the CDF of the standard semicircle. 
We have A = 2 gamma / sqrt(qh), B = h sqrt(qh) and C = 2 lambda_m / sqrt(qh), so A, B >= 0

We return the values of q, h, and the boundary lambdamax of the spectrum of nu
"""

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Iterator for the min - Schatten gamma Norm solution for ellipsoid fitting')
    parser.add_argument("--alphas", nargs="+", type = float)#alpha_min, alpha_max, and the NB of points
    parser.add_argument("--gammas", nargs="+", type = float)#gamma_min, gamma_max, and the NB of points
    parser.add_argument("--save", type = int, default = 0) #0 or 1
    parser.add_argument("--verbosity", type = int, default = 1)
    args = parser.parse_args()

    global_seed = int(time.time())
    np.random.seed(global_seed)
    verbosity = args.verbosity

    assert np.size(args.alphas) in [1,3]
    if np.size(args.alphas) == 3:
        alphas = np.linspace(args.alphas[0], args.alphas[1], num = int(args.alphas[2]), endpoint=True)
    else:
        alphas = np.array([args.alphas[0]])

    assert np.size(args.gammas) in [1,3]
    if np.size(args.gammas) == 3:
        gammas = np.linspace(args.gammas[0], args.gammas[1], num = int(args.gammas[2]), endpoint=True)
    else:
        gammas = np.array([args.gammas[0]])
    assert np.amin(gammas) > 1, f"ERROR: I can only handle gamma > 1, here gammas[0] = {gammas[0]}"

    #Now we do an annealing in gammas, alphas
    qs, hs, lmaxs = np.zeros((np.size(gammas), np.size(alphas))), np.zeros((np.size(gammas), np.size(alphas))), np.zeros((np.size(gammas), np.size(alphas)))
    for i_g in tqdm(range(np.size(gammas)),leave=False):
        gamma = gammas[i_g]
        starting = None
        if i_g > 0:
            starting = {'q': qs[i_g-1][0], 'h':hs[i_g-1][0]} #We use the value for the previous gamma in this case
        for i_a in tqdm(range(np.size(alphas)),leave=False):
            alpha = alphas[i_a]
            results = run(alpha, gamma, starting)
            qs[i_g][i_a] = results['q']
            hs[i_g][i_a] = results['h']
            lmaxs[i_g][i_a] = bound(gamma, results['h'], results['q'])
            starting = {'q': qs[i_g][i_a], 'h':hs[i_g][i_a]}

    if verbosity >= 2:
        for (i_g, gamma) in enumerate(gammas):
            for (i_a, alpha) in enumerate(alphas):
                print(f"For gamma = {gamma} and alpha = {alpha}, I found q = {qs[i_g][i_a]}, h = {hs[i_g][i_a]}, lmax = {lmaxs[i_g][i_a]}")

    if bool(args.save):
        output_ = {'alphas':alphas, 'gammas':gammas, 'qs': qs, 'hs': hs, 'lmaxs': lmaxs}
        filename_ = f"Data/Min_Sgamma_minus_Id_gamma0_{gammas[0]}_gamma1_{gammas[-1]}.pkl"
        outfile_ = open(filename_,'wb')
        pickle.dump(output_,outfile_)
        outfile_.close()
