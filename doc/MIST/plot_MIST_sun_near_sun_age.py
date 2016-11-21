'''
Make plot of MIST's published sun, showing radius, luminosity and Teff all vs
star age.
Set xlim to 4Gyr to 5Gyr to see that indeed their values are pretty far off.
'''
import read_mist_models
import numpy as np, matplotlib.pyplot as plt

sun_age = 4.57e9
sun_teff = 5772.
fs = 14

eep_dir = 'MIST_v1.0_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS/'
track_name = '00100M.track.eep'
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
ax.text(0.1,0.85,track_name+',\n1Msun, solar metallicity', fontsize=fs,
        transform=ax.transAxes, horizontalalignment='left',
        verticalalignment='center')
xmin, xmax = 4e9, 5e9
ymin, ymax = 0.5, 1.5

ax.set(xlabel='Star Age', ylabel='Quantity',
       xlim=[xmin, xmax], ylim=[ymin, ymax])
ax.vlines(sun_age, ymin, ymax, label='4.57Gyr (age of sun)')
leg = ax.legend(loc=3, fontsize=fs)
leg.draw_frame(False)   
plt.savefig('mist_sun_vs_time_near_sun_age.pdf')

