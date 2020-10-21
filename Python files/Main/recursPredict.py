from tqdm import tqdm
import numpy as np
from scipy.stats import poisson

def show_progress(it, progress):
    if progress:
        return tqdm(it)
    else:
        return it

def recursPredict(Rgrid, pR, Lam, Rmean, progress=False):
    '''
    % Given discrete R distribution get Poiss predictions

    % Assumptions and notes
    % - uses posterior over R from recursive filter
    % - computes APE score for predictions
    '''

    # Number of points, days, and range
    (nday, m) = np.shape(pR)
    ir = np.arange(nday)

    if (np.size(Rgrid) != m) or (np.size(Lam) != nday):
        errormsg = 'Input vectors of incorrect dimension'
        raise ValueError(errormsg)

    # Discrete space of possible predictions
    Igrid = np.arange(801)
    lenI = np.size(Igrid)
    # Check if close to upper bound
    pred0 = Lam*Rmean
    pred0 = pred0[1:]
    if np.any(pred0 > 0.9*np.max(Igrid)):
        errormsg = 'Epidemic size too large'
        raise RuntimeError(errormsg)

    # Prediction cdf and quantiles
    Fpred = np.zeros((nday-1, lenI))
    predInt = np.zeros((nday-1, 2))
    pred = np.zeros(nday-1)

    # At every timem construct CDF of predictions
    for i in show_progress(range(nday-1), progress):
        # Compute rate from Poisson renewal
        rate = Lam[i]*Rgrid

        # Probability of any I marginalized over Rgrid
        pI = np.zeros(lenI)

        # Probability of observations 1 day ahead
        for j in range(lenI):
            # Raw probabilities of Igrid
            pIset = poisson.pmf(Igrid[j], rate)
            # Normalized by probs of R
            pI[j] = np.sum(pIset*pR[i, :])

        # Quantile predictions and CDF at i+1
        Fpred[i, :] = np.cumsum(pI)/np.sum(pI)
        idlow = np.argwhere(Fpred[i, :] >= 0.025)[0][0]
        idhigh = np.argwhere(Fpred[i, :] >= 0.975)[0][0]

        # Assign prediction results
        predInt[i, 0] = Igrid[idlow]
        predInt[i, 1] = Igrid[idhigh]
        # Mean prediction by integrating over grid
        if i > 0:
            pred[i] = np.matmul(Igrid, pI.reshape(-1, 1)/np.sum(pI))
        else:
            # Initialize with renewal mean
            pred[i] = pred0[0]

    return pred, predInt
