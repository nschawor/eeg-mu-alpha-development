"""Make tiny plots of waveform features to explain parameterization of cycles
with bycycle (Figure 2B).
"""
# %% Imports
import os
import matplotlib.pyplot as plt
import numpy as np
import neurodsp.sim
from core.params import *

# Set parameters
nr_seconds = 0.1
fs = 1000
time = np.linspace(-nr_seconds/2, nr_seconds, int(fs*nr_seconds))
freq = 10
sig = np.sin(2*np.pi*freq*time) + 0.25*np.sin(2*np.pi*2*freq*(time+0.01))
sig -= np.mean(sig)

# Find peaks and zero-crossings
idx_min = np.where(np.diff(np.sign(np.diff(sig)))>1)
idx_max = np.where(np.diff(np.sign(np.diff(sig)))<-1)
idx_zx1 = np.where(np.diff(np.sign(sig)) > 1)
idx_zx = np.where(np.diff(np.sign(sig)) < -1)

# Initialize plot
fig, ax = plt.subplots(2, 2)

# Plot frequency
ax[0, 0].axvspan(
    xmin=time[idx_min[0][0]], xmax=time[idx_min[0][1]], alpha=0.3, color='gray')
ax[0, 0].axvline(time[idx_min[0][0]], linestyle='dotted', c='k')
ax[0, 0].axvline(time[idx_min[0][1]], linestyle='dotted', c='k')

# Plot rise-decay asymmetry
ax[0, 1].axvspan(xmin=time[idx_min[0][0]], xmax=time[idx_max], alpha=0.3)
ax[0, 1].axvspan(
    xmin=time[idx_max], xmax=time[idx_min[0][1]], alpha=0.3, color='r')

# Plot peak-trough asymmetry
ax[1, 0].plot(time[idx_zx1], sig[idx_zx1], 'k.', markersize=10, zorder=2)
ax[1, 0].plot(time[idx_zx], sig[idx_zx], 'k.', markersize=10, zorder=2)
ax[1, 0].axvspan(
    xmin=time[idx_zx[0][0]], xmax=time[idx_zx1[0][0]], alpha=0.3,
    color='goldenrod')
ax[1, 0].axvspan(
    xmin=time[idx_zx1[0][0]], xmax=time[idx_zx[0][1]], alpha=0.3,
    color='lightseagreen')

# Plot zero line and make window tight
for i in range(3):
    ax.flat[i].axis('off')
    ax.flat[i].plot(time, sig, 'k', zorder=0)
    ax.flat[i].plot(time, 0*time, 'k', alpha=0.2, zorder=-3)
    ax.flat[i].plot(time[idx_min], sig[idx_min], 'k.', markersize=10)
    ax.flat[i].plot(time[idx_max], sig[idx_max], 'k.', markersize=10)
    ax.flat[i].set(ylim=(-1.3, 1.3))

# Create bursting signal to demonstrate normalized amplitude feature
np.random.seed(22)
nr_seconds = 0.35
fs = 5000
time = np.linspace(-nr_seconds/2, nr_seconds, int(fs*nr_seconds))
freq = 10
sig = np.sin(2*np.pi*freq*time) + 0.25*np.sin(2*np.pi*2*freq*(time+0.01))
idx_start = 450
sig[:idx_start] = 0.1*np.random.randn(len(sig[:idx_start]))
idx_end = int(len(sig)/2)+370
sig[idx_end:] = 0.1*np.random.randn(len(sig[idx_end:]))
ap = neurodsp.sim.aperiodic.sim_powerlaw(nr_seconds, fs, exponent=-2)
sig -= np.mean(sig)

# Plot normalized amplitude
ax[1, 1].plot(time, sig, 'k', zorder=0)
idx_zx1 = np.where(np.diff(np.sign(sig)) > 1)
idx_zx = np.where(np.diff(np.sign(sig)) < -1)
ax[1, 1].axvspan(xmin=time[0], xmax=time[idx_start], color='grey', alpha=0.3)
ax[1, 1].axvspan(xmin=time[idx_end], xmax=time[-1], color='grey', alpha=0.3)
ax[1, 1].axvspan(
    xmin=time[idx_start], xmax=time[idx_end], color='orange', alpha=0.3)
ax[1, 1].axis('off')

# Save figure
fig.set_size_inches(4, 3)
fig.tight_layout()
os.makedirs(FIG_FOLDER, exist_ok=True)
fig.savefig(f'{FIG_FOLDER}/fig2_tiny_plots.pdf', dpi=200, transparent=True)
# %%
