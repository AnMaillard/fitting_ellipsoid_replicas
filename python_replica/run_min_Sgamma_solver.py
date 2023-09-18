# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, line-too-long, missing-module-docstring, multiple-imports, pointless-string-statement, import-outside-toplevel

import argparse, time, pickle
from tqdm import tqdm
import numpy as np
from functions import run

"""
Solve the replica equations for the minimal Schatten-gamma norm solution in the ellipsoid fitting problem. 
This gives the spectra for a fixed value of (alpha, gamma)

This is for the case min_{S in Gamma}||S||_gamma, with gamma > 1.  The NN case is directly tackled in the Jupyer notebook. 

We parametrize the measure mu by the three parameters h, q, lambdam (q is denoted hat{q}_0 in the latex file, lambdam is lambda1)
F_mu(x) = F_{sc}[2*gamma*|x|^(gamma)/(x*Sqrt[q]) + h*Sqrt[q]*x + 2 lambda_m / sqrt(q)] where F_{sc} is the CDF of the standard semicircle. 
We return the values of q, h, lambdam, and the boundaries (lambdamin, lambdamax) of the spectrum of nu
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
    qs, hs, lambdams, lmaxs, lmins = np.zeros((np.size(gammas), np.size(alphas))), np.zeros((np.size(gammas), np.size(alphas))), np.zeros((np.size(gammas), np.size(alphas))), np.zeros((np.size(gammas), np.size(alphas))), np.zeros((np.size(gammas), np.size(alphas)))
    for i_g in tqdm(range(np.size(gammas)),leave=False):
        gamma = gammas[i_g]
        starting = None
        if i_g > 0:
            starting = {'q': qs[i_g-1][0], 'h':hs[i_g-1][0], 'lambdam':lambdams[i_g-1][0]} #We use the value for the previous gamma in this case
        for i_a in tqdm(range(np.size(alphas)),leave=False):
            alpha = alphas[i_a]
            results = run(alpha, gamma, starting)
            qs[i_g][i_a] = results['q']
            hs[i_g][i_a] = results['h']
            lambdams[i_g][i_a] = results['lambdam']
            lmins[i_g][i_a], lmaxs[i_g][i_a] = results['lmin'], results['lmax']
            starting = {'q': qs[i_g][i_a], 'h':hs[i_g][i_a], 'lambdam':lambdams[i_g][i_a]}

    if verbosity >= 2:
        for (i_g, gamma) in enumerate(gammas):
            for (i_a, alpha) in enumerate(alphas):
                print(f"For gamma = {gamma} and alpha = {alpha}, I found q = {qs[i_g][i_a]}, h = {hs[i_g][i_a]}, lambdam = {lambdams[i_g][i_a]}. Bounds of the spectrum: ({lmins[i_g][i_a]}, {lmaxs[i_g][i_a]}).")

    if bool(args.save):
        output_ = {'alphas':alphas, 'gammas':gammas, 'qs': qs, 'hs': hs, 'lambdams': lambdams, 'lmaxs': lmaxs, 'lmins':lmins}
        filename_ = f"Data/Min_Sgamma_gamma0_{gammas[0]}_gamma1_{gammas[-1]}.pkl"
        outfile_ = open(filename_,'wb')
        pickle.dump(output_,outfile_)
        outfile_.close()
