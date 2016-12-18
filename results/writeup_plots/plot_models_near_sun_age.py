'''
For writeup:
``quantity vs time'' plot, focused on the age of the Sun. 
Quantities:
    * Lstar, Rstar, Teffstar, Itotstar
Lines:
    * MIST solar model (not calibrated, but what's in their grids),
    * Our MESA-computed solar model (calibrated, what's in our grids)
        (i.e. using grid_production_0)
    * vline: sun age
    * hline (or point): CD96 solar model moment of inertia
Set xlim to 4Gyr to 5Gyr to see that indeed their values are pretty far off.
'''

import matplotlib as mpl
mpl.use("pgf")
pgf_with_custom_preamble = {
    'pgf.texsystem': 'pdflatex', # xelatex is default; i don't have it
    'font.family': 'serif', # use serif/main font for text elements
    'text.usetex': True,    # use inline math for ticks
    'pgf.rcfonts': False,   # don't setup fonts from rc parameters
    }
mpl.rcParams.update(pgf_with_custom_preamble)

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from math import pi as π
import os
import read_mist_models

sun_age = 4.57e9 # C+2016
sun_teff = 5772. # C+2016
fs = 14

# Get Choi et al published grid solar model
eep_dir = 'MIST_v1.0_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS/'
track_name = '00100M.track.eep'
eep = read_mist_models.EEP(eep_dir+track_name)
print('available columns: {}'.format(eep.hdr_list))

plt.close('all')
f, ax = plt.subplots(figsize=(7,5))
star_age = eep.eeps['star_age']
star_radius = 10**(eep.eeps['log_R'])
star_teff = 10**(eep.eeps['log_Teff'])
star_L = 10**(eep.eeps['log_L'])

ax.plot(star_age/1e9, star_radius-1., c='gold', ls='--')
ax.plot(star_age/1e9, star_L-1., c='forestgreen', ls='--')
ax.plot(star_age/1e9, star_teff/sun_teff-1., 
    c='steelblue', ls='--')

ax.text(0.05,0.9,'Dashed: MIST C+16 (not fine-tuned, '+track_name+\
        ' 1$M_\odot$, $Z_\odot$)'+
        '\nSolid: MESA (solar fine-tuned, our grid)', 
        fontsize=fs*0.7,
        transform=ax.transAxes, horizontalalignment='left',
        verticalalignment='center',
        bbox=dict(facecolor='white', edgecolor='white', 
        boxstyle='square,pad=0.5'), zorder=11)

# Sun age vertical line
xmin, xmax = 4, 5
ymin, ymax = -0.1, 0.2

ax.vlines(sun_age/1e9, ymin, ymax, zorder=10)
ax.hlines(0., xmin, xmax, zorder=10)
ax.text(sun_age/1e9, -0.07, '4.57Gyr (age of sun)', fontsize=fs*0.7,
        horizontalalignment='center', verticalalignment='center',
        bbox=dict(facecolor='white', edgecolor='white', 
        boxstyle='square,pad=0.5'), zorder=11)

ax.set(xlabel='Model age [Gyr]', ylabel='Model - predicted [units in legend]',
       xlim=[xmin, xmax], ylim=[ymin, ymax])

# MIST fine-tuned solar model
table_dir = '../grid_production_0/tables/'
table_path =  [table_dir+p for p in np.array(os.listdir(table_dir)) if
    'M1.0_Z0.015' in p][0]
tab = pd.read_csv(table_path, index_col='age')

ax.plot(tab.index/1e9, tab['R_star']-1., label='$R_\star\ [R_\odot]$', 
        c='gold', ls='-')
ax.plot(tab.index/1e9, tab['L_star']-1., label='$L_\star\ [L_\odot]$', 
        c='forestgreen', ls='-')
R_sun, M_sun, L_sun = 6.957e10, 1.988e33, 3.928e33  # Choi+ 2016 values.
σ_SB = 5.6704e-5
tab_teff = ((tab['L_star']*L_sun) / \
        (4*π*(tab['R_star']*R_sun)**2 * σ_SB))**(1/4.)
ax.plot(tab.index/1e9, tab_teff/sun_teff-1., 
    label='$T_\mathrm{eff,\star}\ [T_\mathrm{eff,\odot}]$',
    c='steelblue', ls='-')
# total moment of inertia
I_sun = M_sun * R_sun * R_sun
    # ignores order unity prefactor.
I_sun_num = 7.07990201e53 / I_sun
    # from numerical integration of Christensen-Dalsgaard 1996
tab_I_tot = tab['I_conv']+tab['I_rad']
ax.plot(tab.index/1e9, tab_I_tot - I_sun_num, 
        label='$I_{\star,\mathrm{tot}}\ [M_\odot R_\odot^2]$',
        c='firebrick', ls='-')


leg = ax.legend(loc=3, fontsize=fs*0.7)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_edgecolor('white')

f.tight_layout()
f.savefig('models_near_sun_age.pdf')
