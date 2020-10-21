import numpy as np
from scipy.stats import norm, poisson

def runEpiFilterSm(Rgrid, nPts, eta, nday, pr0, Lam, Iday):
    '''

    % Assumptions and notes
    % - compatible version of runEpiFilter for smoothing
    % - discrete filter on grid from Rmin to Rmax
    % - provides conditional posterior and mean
    % - expects a renewal model incidence input curve
    '''

    # Prob vector for R and prior
    pR = np.zeros((int(nday), int(nPts)))
    pRup = np.zeros_like(pR)
    pR[0, :] = pr0
    pRup[0, :] = pr0

    # Mean, median, and confidence on R
    Rm = np.zeros(nday)
    Rmed = np.zeros_like(Rm)
    Rlow = np.zeros_like(Rm)
    Rhigh = np.zeros_like(Rm)

    # Initial stats
    Rm[0] = np.matmul(pR[0, :], Rgrid.reshape(-1, 1))
    ids = np.zeros(3, dtype=np.int)
    Rcdf0 = np.cumsum(pR[0, :])
    ids[0] = np.argwhere(Rcdf0 > 0.5)[0][0]
    ids[1] = np.argwhere(Rcdf0 > 0.025)[0][0]
    ids[2] = np.argwhere(Rcdf0 > 0.975)[0][0]
    Rmed[0] = Rgrid[ids[0]]
    Rlow[0] = Rgrid[ids[1]]
    Rhigh[0] = Rgrid[ids[0]] # really 0?

    # Precompute state distributions
    pstate = np.zeros((nPts, nPts))
    for j in range(nPts):
        pstate[j, :] = norm.pdf(Rgrid[j], loc=Rgrid, scale=np.sqrt(Rgrid)*eta)

    # Update prior to posterior sequentially
    for i in range(1, nday):
        # Compute rate from Poisson renewal
        rate = Lam[i]*Rgrid
        # Probabilities of observations
        pI = poisson.pmf(Iday[i], rate)

        # State equations for R
        pRup[i, :] = np.matmul(pR[i-1, :], pstate)

        # Posterior over R (update)
        pR[i, :] = pRup[i, :]*pI  # by element
        pR[i, :] = pR[i, :] / np.sum(pR[i, :])

        # Posterior mean and CDF
        Rm[i] = np.matmul(pR[i, :], Rgrid.reshape(-1, 1))
        Rcdf = np.cumsum(pR[i, :])

        # Quantiles from estimates
        ids[0] = np.argwhere(Rcdf > 0.5)[0][0]
        ids[1] = np.argwhere(Rcdf > 0.025)[0][0]
        ids[2] = np.argwhere(Rcdf > 0.975)[0][0]
        Rmed[i] = Rgrid[ids[0]]
        Rlow[i] = Rgrid[ids[1]]
        Rhigh[i] = Rgrid[ids[2]]

    return Rmed, Rlow, Rhigh, Rm, pR, pRup, pstate