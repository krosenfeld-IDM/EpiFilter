import numpy as np
from scipy.special import gamma

def serialDistrTypes(tmax, distvals):
    '''

    % Assumptions and notes
    % - Chooses between discrete distributions on days
    % - Insert max days and calculate serial probabilities
    % - p must be a parameter, tday an array of integers
    % - p is 1/mean of each distribution
    '''

    if distvals['type'] == 1:
        raise ValueError('Not yet implemented')
    elif distvals['type'] == 2:
        # Gamma distribution with integer shape (Erlang)
        pdistr = lambda p: gammaDistr(p, np.arange(1, tmax+1), distvals['pm'])
    else:
        raise ValueError('Not yet implemented')

    return pdistr

def gammaDistr(p, x, shapePm):
    scalePm = 1/(p*shapePm)
    ratePm = 1/scalePm
    pr = -np.log(gamma(shapePm)) + shapePm*np.log(ratePm) \
         + (shapePm-1)*np.log(x) - ratePm*x
    pr = np.exp(pr)
    return pr