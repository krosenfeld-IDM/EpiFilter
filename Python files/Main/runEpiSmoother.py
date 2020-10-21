import numpy as np


def runEpiSmoother(Rgrid, nPts, nday, pR, pRup, pstate):
    '''
    % Bayesian recursive smoother for renewal models
    % Assumptions and notes
    % - discrete smoother on grid from Rmin to Rmax
    % - assumes runEpiFilter outputs available
    % - forward-back algorithm for smoothing applied
    '''

    # Last smoothed distribution same as filtered
    qR = np.zeros((nday, nPts))
    qR[-1, :] = pR[-1, :]

    # Main smooth equation iteratively computed
    for i in range(nday-2, -1, -1):
        # Remove zeros
        pRup[i+1, pRup[i+1, :] == 0] = 1e-8

        # Integral term in smoother
        integ = qR[i+1, :] / pRup[i+1, :]
        integ = np.matmul(integ, pstate)

        # Smoothed posterior over Rgrid
        qR[i, :] = pR[i, :].ravel()*integ.ravel()
        # Force normalization
        qR[i, :] = qR[i, :] / np.sum(qR[i, :])

    # Mean, median, and confidence on R
    Rm = np.zeros(nday)
    Rmed = np.zeros_like(Rm)
    Rlow = np.zeros_like(Rm)
    Rhigh = np.zeros_like(Rm)
    for i in range(nday):
        # Posterior mean and CDF
        Rm[i] = np.matmul(qR[i, :], Rgrid.reshape(-1, 1))
        Rcdf = np.cumsum(qR[i, :])

        # Quantiles from estimates
        Rmed[i] = Rgrid[np.argwhere(Rcdf > 0.5)[0][0]]
        Rlow[i] = Rgrid[np.argwhere(Rcdf > 0.025)[0][0]]
        Rhigh[i] = Rgrid[np.argwhere(Rcdf > 0.975)[0][0]]

    return Rmed, Rlow, Rhigh, Rm, qR