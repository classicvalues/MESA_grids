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
#plt.style.use('fivethirtyeight') # nope nope nope
import os

table_dir = '../grid_production_0/tables/'
table_paths =  [table_dir+p for p in np.array(os.listdir('../grid_production_0/tables/'))]
bold_Z = 0.015
bold_Z_paths = np.sort(np.array([p for p in table_paths if str(bold_Z) in p]))
other_paths = np.sort(np.array([p for p in table_paths if str(bold_Z) not in p]))

bold_Z_tabs, other_tabs = {}, {}
for path in bold_Z_paths:
    bold_Z_tabs[path] = pd.read_csv(path, index_col='age')
for path in other_paths:
    other_tabs[path] = pd.read_csv(path, index_col='age')

plt.close('all')
f, axs = plt.subplots(3, 2, sharex='col', figsize=(7,9.5))

for bk in list(bold_Z_tabs.keys()):
    this_table = bold_Z_tabs[bk]
    this_table = this_table.sort_index()
    this_table['I_tot'] = np.array(this_table['I_rad']) + np.array(this_table['I_conv'])
    y_order = ['R_star', 'I_conv',  'L_star', 
               'I_rad', 'R_tachocline', 'I_tot']
    y_labels = ['$\log_{10}(R_\star\ [R_\odot])$', 
                '$\log_{10}(I_\mathrm{conv}\ [M_\odot R_\odot^2])$',
                '$\log_{10}(L_\star\ [L_\odot])$', 
                '$\log_{10}(I_\mathrm{rad}\ [M_\odot R_\odot^2])$',
                '$\log_{10}(R_\mathrm{tachocline}\ [R_\odot])$', 
                '$\log_{10}(I_\mathrm{tot}\ [M_\odot R_\odot^2])$']
    for i, ax in enumerate(axs.flatten()):
        this_y, this_y_label = y_order[i], y_labels[i]
        ax.plot(np.log10(this_table.index.values), np.log10(this_table[this_y]), c='k', lw=1)
        ax.set_ylabel(this_y_label)
        if i == 4 or i == 5:
            ax.set_xlabel('$\log_{10}(\mathrm{age\ [yr]})$')

mass_fracs = [0.0015, 0.00266741911506, 0.00474341649025, 0.00843511987786, 
              0.0266741911506, 0.0474341649025]
mass_frac_alphas = np.arange(0.1,0.1+.14*6,0.14)
l = np.array([el for el in zip(mass_fracs, mass_frac_alphas)])

for bk in list(other_tabs.keys()):
    if 'M0.5' in bk or 'M1.1' in bk:
        this_Z = float(bk.split('_Z')[1][:-4])
        this_α = float(l[np.where(np.isclose(l, this_Z))[0], 1])
        
        this_table = other_tabs[bk]
        this_table = this_table.sort_index()
        this_table['I_tot'] = np.array(this_table['I_rad']) + np.array(this_table['I_conv'])
        y_order = ['R_star', 'I_conv',  'L_star', 'I_rad', 'R_tachocline', 'I_tot']
        for i, ax in enumerate(axs.flatten()):
            this_y = y_order[i]
            ax.plot(np.log10(this_table.index.values), np.log10(this_table[this_y]), c='k', 
                    lw=0.5, alpha=this_α)
            
f.tight_layout()
f.savefig('grids_2by3.pdf')
