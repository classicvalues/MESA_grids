'''
From /results/grid_*/tables, give this program a *.csv file. It will give
you a saved HR diagram.

Usage:
    python UTIL_make_2d_plots.py grid_id mass mass_frac_to_2sf
Args:
    grid_id: substring identifying the MESA grid data. (e.g., "0') for 
        "grids_0" in data.
    mass: of star, e.g., "0.35"
    mass_frac_to_2sf: Z to 2 signifacant figures, e.g., 0.047 for 0.0474341649
Options:
    None
'''

import numpy as np, pandas as pd, units as u
from math import pi as π
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import re, os
import argparse
plt.style.use('classic')

global R_sun, M_sun, L_sun, T_eff_sun, I_sun
R_sun, M_sun, L_sun, T_eff_sun = 6.957e10, 1.988e33, 3.928e33, 5772 
    # Choi+ (2016) cgs values.
I_sun = 7.07990201e53 
    # from numerical integration of Christensen-Dalsgaard 1996
    # see email chain with Dappen & Rhodes. Purely model-determined.

def plot_HR(df, grid_dir, star_name):
    '''
    Plots HR diagram. Saves to /results/grid_*/plots.

    Input: 
    df: pandas DataFrame with data from /results/tables/*csv loaded in
    grid_dir: string to be parsed for save
    star_name: string of csv file name of star
    '''
    assert 'L_star' in df.columns and 'R_star' in df.columns

    df['T_eff'] = (df.L_star * u.Lsun/\
            (4 * π * (df.R_star*u.Rsun)**2 * u.sigma_SB))**(1/4.)

    plt.close('all')
    f, ax = plt.subplots()
    ax.plot(np.log10(df['T_eff']), np.log10(df['L_star']), 'k-o')
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set(xlabel='log10(Teff [K])', ylabel='log10(Lstar [Lsun])')

    save_dir = grid_dir[:-7]+'plots/'
    f.savefig(save_dir+'HR_full_'+star_name[:-4]+'.pdf')

    ax.set(xlim=[3.465,3.45], ylim=[-2.2,-2.0])
    f.savefig(save_dir+'HR_small_'+star_name[:-4]+'.pdf')

def plot_R_Mconv(df, grid_dir, star_name):
    '''
    Replicates row5, col3 of FigA1, Choi+ 2016. Saves to /results/grid_*/plots.

    Input: 
    df: pandas DataFrame with data from /results/tables/*csv loaded in
    grid_dir: string to be parsed for save
    star_name: string of csv file name of star
    '''

    plt.close('all')
    f, ax = plt.subplots()
    ax.plot(df['age'], df['M_conv'], c='steelblue', label=
        'M_conv (no addition, in Msun units)')
    ax.plot(df['age'], df['R_star'], c='salmon', label='R_star [R_sun]')
    ax.set(xlabel='Age [yr]', ylabel='', xlim=[8e9, 2e10], ylim=[0.2,0.4])
    ax.legend(loc='best')

    save_dir = grid_dir[:-7]+'plots/'
    save_path = save_dir+'Mconv_and_Rstar_vs_t_'+star_name[:-4]+'.pdf' 
    f.savefig(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('args', nargs='+')
    ao = parser.parse_args()
    assert len(ao.args) == 3, 'See header for allowed args.'

    base_data = '/home/luke/Dropbox/software/mesa/results/grid_'
    global data_dir
    grid_dir = base_data + ao.args[0] + '/tables/' 
    star_names = np.sort([f for f in os.listdir(grid_dir) if ('M' in f) and 
        ('_Z' in f)])

    M, Z_2sf = ao.args[1], ao.args[2]
    star_name = [sn for sn in star_names if ('M'+M in sn) and ('Z'+Z_2sf in sn)]
    assert len(star_name) == 1, 'There should be only one matching file'
    data_path = grid_dir + star_name[0]

    df = pd.read_csv(data_path)
    df = df.sort_values('age')

    plot_HR(df, grid_dir, star_name[0])
    plot_R_Mconv(df, grid_dir, star_name[0])

    print('made plots.')

if __name__ == '__main__':
    main()
