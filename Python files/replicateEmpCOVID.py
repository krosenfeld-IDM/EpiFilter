'''
% Assumptions and notes
% - all predictions are causal
% - input of empirical data from WHO on COVID-19
% - includes solutions to smoothing and filtering problems
% - no comparison to APE or EpiEstim and naively uses cases
% - assumes serial interval from Ferguson et al
'''

import pickle
import sys
import os
import glob
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sys.path.append('Main')
from serialDistrTypes import serialDistrTypes
from runEpiFilterSm import runEpiFilterSm
from recursPredict import recursPredict
from runEpiSmoother import runEpiSmoother

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Directory and if saving
thisDir = os.path.dirname(__file__)
saveTrue = 1
pickleTrue = 1

# Folder for saving and loading
saveFol = os.path.join('Results', 'COVID')
loadFol = os.path.join('Data', 'COVID/')
# Possible countries of interest
countries = ('New Zealand', 'Croatia', 'Greece')
# Dates of lockdown and release for each
lockdowns = ('26-03-20', '18-03-20', '23-03-20');
releases = ('14-05-20', '19-04-20', '04-05-20');

# Decide country to investigate
scenNo = 0; scenNam = countries[scenNo];
print('Examining data from ' + scenNam);
# % Identifier for saving
namstr = '_' + scenNam

# Get specific lockdown/release for country
lock = lockdowns[scenNo]; relax = releases[scenNo];
lock = datetime.strptime(lock, '%d-%m-%y').date();
relax = datetime.strptime(relax, '%d-%m-%y').date();

# Read in epidemic curves
file = glob.glob(os.path.join(loadFol, 'WHO*'))[0]
# Data for all countries
data = pd.read_csv(file)
# Select parts with country of interest
id = (data[' Country'] == scenNam)
if np.sum(id) == 0:
    raise ValueError('Incorrect or unavailable country')

# Incidence and death time-series
Iday = data[id][' New_cases'].values
nday = len(Iday)
# Dates and deaths
dateRep = data[id]['Date_reported']
Dday = data[id][' New_deaths'].values
del data

# Reorder dates to be in commin format
dateRep = dateRep.apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date()).values
# Get index of lockdown and release
idLock = np.argwhere(lock == dateRep)[0]
idRelax = np.argwhere(relax == dateRep)[0]
if (len(idLock) == 0) or (len(idRelax) == 0):
    raise ValueError('Lockdown and release dates no in data')

# Numeric value associate with date
xval = np.arange(1, nday+1)

# Gamma serial interval distribution
distvals = {}
distvals['type'] = 2
distvals['pm'] = (1/0.65)**2
distvals['omega'] = 6.5
# Serial distribution over all days
serial = serialDistrTypes(nday, distvals)
Pomega = serial(1/distvals['omega']);

# times
tday = np.arange(1, nday+1)

# Total infectiousness
Lam = np.zeros(Iday.shape)
for i in range(1, nday):
    Pomegat = Pomega[0:i]
    # Total infectiousness
    Lam[i] = np.sum(Iday[i-1::-1]*Pomegat)

# Recursive filter and predictions
# Grid limits and noise level
Rmin = 0.01
Rmax = 10
eta = 0.1
print(f'[Eta, Rmax, Rmin] = [{Rmin}, {Rmax}, {eta}]')

# Uniform prior over grid of size m
m = 2000
p0 = 1/m*np.ones(m)
# Delimited grid definite space of R
Rgrid = np.linspace(Rmin, Rmax, m)

# EpiFilter estimates for for single trajectory
Rmed, Rlow, Rhigh, Rmean, pR, pRup, pstate = runEpiFilterSm(Rgrid, m, eta, nday, p0, Lam, Iday);

# EpiFilter one-step-ahead prediction (takes time so pickle the results)
if False:
    predF, predIntF = recursPredict(Rgrid, pR, Lam, Rmean, progress=True)
    if pickleTrue:
        pickle.dump({'predF': predF, 'predIntF': predIntF}, open(namstr+'_recursPredict.pkl', 'wb'))
else:
    warnings.warn('Loading pickled recursPredict results')
    tmp = pickle.load(open(namstr+'_recursPredict.pkl', 'rb'))
    predF = tmp['predF']
    predIntF = tmp['predIntF']

# For probabilities above or below 1
id1 = np.argwhere(Rgrid <= 1)[-1][0]
prL1 = np.zeros(nday)
# Update prior to posterior sequentially
for i in range(1, nday):
    # Posterior CDF and prob R <= 1
    Rcdf = np.cumsum(pR[i, :])
    prL1[i] = Rcdf[id1]

## Recursive smoother and predictions
# EpiSmoother estimates for single trajectory
[RmedS, RlowS, RhighS, RmeanS, qR] = runEpiSmoother(Rgrid, m, nday, pR, pRup, pstate)

# EpiSmoother one-step-ahead predictions
if False:
    predS, predIntS = recursPredict(Rgrid, qR, Lam, RmeanS, progress=True);
    if pickleTrue:
        pickle.dump({'predS': predS, 'predIntS': predIntS}, open(namstr+'S_recursPredict.pkl', 'wb'))
else:
    warnings.warn('Loading pickled S_recursPredict results')
    tmp = pickle.load(open(namstr+'S_recursPredict.pkl', 'rb'))
    predS = tmp['predS']
    predIntS = tmp['predIntS']

# For probabilities above or below 1
id1 = np.argwhere(Rgrid <= 1)[-1][0]
prL1S = np.zeros(nday)
# Update prior to posterior sequentially
for i in range(1, nday):
    # Posterior CDF and prob R <= 1
    Rcdf = np.cumsum(pR[i, :])
    prL1S[i] = Rcdf[id1]

## Figure
fig, axes = plt.subplots(2, 1, figsize=(8, 8))
# top figure
ax = axes[0]
ax.fill_between(xval, Rlow, Rhigh, color=colors[0], alpha=0.3)
ax.plot(xval, Rmean, color=colors[0])
ax.fill_between(xval, RlowS, RhighS, color=colors[1], alpha=0.3)
ax.plot(xval, RmeanS, color=colors[1])
ax.set_ylim([0, 5])
ylim = ax.get_ylim()
xlim = ax.get_xlim()
ax.hlines(1.0, xlim[0], xlim[1], colors='k', linestyles='--')
ax.vlines([xval[idLock], xval[idRelax]], ylim[0], ylim[1], color=3*[0.6], linestyles='--')
ax.set_ylabel('$\hat{{R}}_s$, \, $\\tilde{{R}}_s$')
ax.set_xlim(xlim)
# bottom figure
ax = axes[1]
ax.scatter(xval[1:], Iday[1:], s=40, c=[3*[0.7]], alpha=0.9, edgecolors='none')
# ax.fill_between(xval[1:], predIntF[:, 0], predIntF[:, 1], color=colors[0], alpha=0.3)
# ax.plot(xval[1:], predF, color=colors[0])
ax.fill_between(xval[1:], predIntS[:, 0], predIntS[:, 1], color=colors[1], alpha=0.3)
ax.plot(xval[1:], predS, color=colors[1])
ax.set_ylabel(f'$\hat{{I}}_s | \eta = ${eta}')
ax.set_xlabel('Days')
ax.set_ylim(0, None)
plt.savefig(os.path.join(saveFol, f'RIsmoothfil_{namstr}.png'))
