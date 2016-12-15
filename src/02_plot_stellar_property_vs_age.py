'''
Take tables in /results/grid_?/tables and plot QUANTITY vs time, for QUANTITY=
    {R_star, L_star, M_star, R_tachocline, M_ini, M_conv, M_rad, I_conv,
    I_rad}. By default, also creates a plot for I_tot.
    Also does it for all metallicities.
Usage:
    >> python 02_plot_stellar_property_vs_age grid_id
Arguments:
    grid_id: a number or string specifying the grid from /mesa/data/grid_*
'''

import numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages as pdf_pages
import re, os, argparse
plt.style.use('classic')

def get_table_paths(grid_base):
    table_dir = grid_base+'/tables/'
    return [table_dir+f for f in os.listdir(table_dir)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arglist', nargs=1)
    a = parser.parse_args()
    grid_base = '/home/luke/Dropbox/software/mesa/results/grid_'+a.arglist[0]

    table_paths = get_table_paths(grid_base)
    star_names = [tp.split('/')[-1][:-4] for tp in table_paths]
    masses, Zs = [float(s.split('_')[0][1:]) for s in star_names], \
                 np.unique([float(s.split('_')[1][1:]) for s in star_names])
    mass_step = float(np.diff(np.sort(masses))[0])

    param_list = ['R_star', 'L_star', 'R_tachocline', 'I_conv', 'I_rad',
        'I_tot']
    ylim_list = [[-0.6,0.6],[-2,1],[-2,0.5],[-10,0.5],[-7,0],[-2.5,0.5]]

    for ix, stellar_param in enumerate(param_list):
        for Z in Zs:
            logy = True
            plt.ioff()
            f, ax = plt.subplots(figsize=(16/2.,9/2.))
            c_norm = mpl.colors.Normalize(vmin=min(masses), vmax=max(masses))
            scalar_map = mpl.cm.ScalarMappable(norm=c_norm, cmap='viridis')

            allowed_stars = [s for s in star_names if 'Z'+str(Z) in s]
            for ind, star in enumerate(np.sort(allowed_stars)):
                mass = star.split('_')[0][1:]
                table_path = [p for p in table_paths if star in p]
                assert len(table_path)==1
                dat = pd.read_csv(table_path[0])
                dat['I_tot'] = dat['I_conv'] + dat['I_rad']

                color_val = scalar_map.to_rgba(mass)
                x_val = np.array(np.log10(dat.sort('age')['age']))
                y_val = np.array(np.log10(dat.sort('age')[stellar_param])) if logy \
                    else np.array(dat.sort('age')[stellar_param])
                if len(x_val) > 1:
                    ax.plot(x_val, y_val, c=color_val, label=star)
                if len(x_val) <= 1:
                    print('Bad data: {:s} {:s}'.format(stellar_param, star))
            
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
            agelim = [5,12]
            ax.set(xlabel='log10(age[yr])', ylabel=ylab, xlim=agelim,
                    ylim=ylim_list[ix])

            plot_dir = grid_base + '/plots/'
            pdf_name = plot_dir+stellar_param+'_vs_t_varM_Z'+str(Z)+'.pdf'
            f.tight_layout()
            f.savefig(pdf_name)
            plt.close()

if __name__ == '__main__':
    main()
