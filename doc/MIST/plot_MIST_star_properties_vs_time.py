'''
Make plot of any MIST star (track.eeps), showing radius, luminosity and Teff 
all vs star age.
Set broad xlims to have diagnostic.
'''
import read_mist_models
import numpy as np, matplotlib.pyplot as plt

M_stars = \
['0.30','0.31','0.32','0.33','0.34','0.35','0.36','0.37','0.38','0.39']

for M_star in M_stars:
    Mstar_str = ''.join(M_star.split('.')).zfill(5)
    sun_teff = 5772.
    fs = 12

    eep_dir = 'MIST_v1.0_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS/'
    track_name = Mstar_str+'M.track.eep'
    eep = read_mist_models.EEP(eep_dir+track_name)
    print('version: {}'.format(eep.version))
    print('abundances: {}'.format(eep.abun))
    print('rotation: {}'.format(eep.rot))
    print('initial mass: {}'.format(eep.minit))
    print('available columns: {}'.format(eep.hdr_list))

    plt.close('all')
    f, ax = plt.subplots()
    star_age = eep.eeps['star_age']
    star_radius = 10**(eep.eeps['log_R'])
    star_teff = 10**(eep.eeps['log_Teff'])
    star_L = 10**(eep.eeps['log_L'])

    ax.plot(star_age, star_radius, label='Radius [Rsun]')
    ax.plot(star_age, star_L, label='Luminosity [Lsun]')
    ax.plot(star_age, star_teff/sun_teff, label='Teff [Teffsun,5772K]')
    ax.text(0.1,0.85,track_name+'\n'+M_star+'Msun, solar metallicity', 
            fontsize=fs, transform=ax.transAxes, horizontalalignment='left',
            verticalalignment='center')
    xmin, xmax = 1e6, 1.5e10
    ymin, ymax = 0.01, 10

    ax.set(xlabel='Star Age', ylabel='Quantity',
           xlim=[xmin, xmax], ylim=[ymin, ymax],
           xscale='log', yscale='log')
    leg = ax.legend(loc=3, fontsize=fs)
    leg.draw_frame(False)   

    plt.savefig('mist_'+M_star+'Msun_props_vs_time.pdf')
