'''
Take tables in /results/grid_?/tables and plot QUANTITY vs time, for QUANTITY=
    {R_star, L_star, M_star, R_tachocline, M_ini, M_conv, M_rad, I_conv,
    I_rad}. By default, also creates a plot for I_tot.
Usage:
    >> python plot_stellar_property_vs_age grid_id Z
Arguments:
    grid_id: a number or string specifying the grid from /mesa/data/grid_*
    Z: metal mass fraction, e.g., 0.015 for solar.

todo:
1. smarter mass fractions (not just what's passed as an argument)
2. also appplies in "mass_step" lines
'''

import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages as pdf_pages
import re, os, argparse
plt.style.use('classic')

def get_table_paths(grid_base, Z):
    table_dir = grid_base+'/tables/'
    return [table_dir+f for f in os.listdir(table_dir) if str(Z) in f]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arglist', nargs=2)
    a = parser.parse_args()
    grid_base = '/home/luke/Dropbox/software/mesa/results/grid_'+a.arglist[0]
    Z = a.arglist[1]

    param_list = ['R_star', 'L_star', 'R_tachocline', 'I_conv', 'I_rad',
        'I_tot']

    for stellar_param in param_list:
        table_paths = get_table_paths(grid_base, Z)
        star_names = [tp.split('/')[-1][:-4] for tp in table_paths]
        masses, Zs = [float(s.split('_')[0][1:]) for s in star_names], \
                     [float(s.split('_')[1][1:]) for s in star_names]
        mass_step = float(np.diff(np.sort(masses))[0])

        logy = True
        plt.ioff()
        f, ax = plt.subplots(figsize=(16/2.,9/2.))
        c_norm = mpl.colors.Normalize(vmin=min(masses), vmax=max(masses))
        scalar_map = mpl.cm.ScalarMappable(norm=c_norm, cmap='viridis')

        for ind, star in enumerate(star_names):
            table_path = [p for p in table_paths if star in p]
            assert len(table_path)==1
            dat = pd.read_csv(table_path[0])
            dat['I_tot'] = dat['I_conv'] + dat['I_rad']

            color_val = scalar_map.to_rgba(masses[ind])
            x_val = np.log10(dat.sort('age')['age'])
            y_val = np.log10(dat.sort('age')[stellar_param]) if logy \
                else dat.sort('age')[stellar_param]
            ax.plot(x_val, y_val, c=color_val, label=star)
        
        handles,labels = ax.get_legend_handles_labels()
        desired_order = np.sort(labels)
        inds, s_inds = [], []
        for l in np.array(labels):
            inds.append(np.where(desired_order == l)[0][0])
        for i in range(len(inds)):
            s_inds.append(np.where(np.array(inds) == i)[0][0])
        lgd = ax.legend(np.array(handles)[s_inds], np.array(labels)[s_inds], 
            loc='best', fontsize='xx-small')

        ylab = 'log10('+stellar_param+')' if logy else stellar_param
        ax.set(xlabel='log10(age[yr])', ylabel=ylab)

        plot_dir = grid_base + '/plots/'
        pdf_name = plot_dir+stellar_param+'_vs_t_varM_Z'+Z+'.pdf'
        f.tight_layout()
        f.savefig(pdf_name)
        plt.close()

if __name__ == '__main__':
    main()
