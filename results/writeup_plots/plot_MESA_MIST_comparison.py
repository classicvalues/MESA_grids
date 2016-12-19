'''
For writeup:
``quantity vs time'' plot, over all time (~1e5yr-1e10yr)for
Quantities:
    * Lstar, Rstar
Lines:
    * MIST solar model (not calibrated, but what's in their grids),
    * Our MESA-computed solar model (calibrated, what's in our grids)
        (i.e. using grid_production_0)
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

sun_teff = 5772. # C+2016
fs = 14

masses = np.arange(0.4,1.3,0.1)
mass_list = ['00040','00050','00060','00070','00080','00090','00100','00110','00120']

for ix, mass in enumerate(masses):
    # Get Choi et al published grid solar model
    eep_dir = 'MIST_v1.0_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS/'
    track_name = mass_list[ix]+'M.track.eep'
    eep = read_mist_models.EEP(eep_dir+track_name)
    print('initial mass: {}'.format(eep.minit))
    #print('available columns: {}'.format(eep.hdr_list))

    plt.close('all')
    f, ax = plt.subplots(figsize=(7,5))
    star_age = eep.eeps['star_age']
    star_radius = 10**(eep.eeps['log_R'])
    star_L = 10**(eep.eeps['log_L'])

    ax.plot(np.log10(star_age), np.log10(star_radius), c='gold', ls='-', 
        label='C+16 $R_\star\ [R_\odot]$')
    ax.plot(np.log10(star_age), np.log10(star_L), c='forestgreen', ls='-',
        label='C+16 $L_\star\ [L_\odot]$')

    # Get MESA grid_production_0 ("main to be used" grid)
    table_dir = '../grid_production_0/tables/'
    table_path =  [table_dir+p for p in np.array(os.listdir(table_dir)) if
        'M'+str(mass)+'_Z0.015' in p][0]
    tab = pd.read_csv(table_path, index_col='age')

    ax.plot(np.log10(tab.index), np.log10(tab['R_star']), 
        c='gold', ls='--',
        label=r'MESA (production_0) $R_\star\ [R_\odot]$')
    ax.plot(np.log10(tab.index), np.log10(tab['L_star']), 
        c='forestgreen', ls='--',
        label=r'MESA (production_0) $L_\star\ [L_\odot]$')

    # MESA grid_Choi_repl uses the correct [Fe/H] normalization (I think)
    table_dir = '../grid_Choi_repl/tables/'
    table_path =  [table_dir+p for p in np.array(os.listdir(table_dir)) if
        'M'+str(mass)+'_Z0.0142' in p][0]
    tab = pd.read_csv(table_path, index_col='age')

    ax.plot(np.log10(tab.index), np.log10(tab['R_star']), c='gold', ls=':',
        label=r'MESA (Choi repl) $R_\star\ [R_\odot]$')
    ax.plot(np.log10(tab.index), np.log10(tab['L_star']), c='forestgreen', ls=':',
        label=r'MESA (Choi repl) $L_\star\ [L_\odot]$')

    # Sun age vertical line
    xmin, xmax = 5, 12

    ax.set(xlabel='$\log_{10}$(age [yr])', 
        ylabel='$\log_{10}$(quantity [unit in legend])',
        xlim=[xmin, xmax],
        title=str(mass)+'M$_\odot$. [Fe/H]=0 (n.b. MIST \& MESA have diff calibrations)')

    leg = ax.legend(loc='best', fontsize=fs*0.7)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('white')

    f.tight_layout()
    f.savefig('MESA_MIST_comparison/'+str(mass)+'Msun_MESA_MIST_comparison.pdf')
