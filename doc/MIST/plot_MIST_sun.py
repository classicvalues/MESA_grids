'''
Make plot of MIST's published sun, showing radius, luminosity and Teff all vs
star age.
Keep broad xlims as a diagnostic.
'''
import read_mist_models
import numpy as np, matplotlib.pyplot as plt

sun_age = 4.57e9
sun_teff = 5772.

eep_dir = 'MIST_v1.0_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS/'
eep = read_mist_models.EEP(eep_dir+'00100M.track.eep')
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
xmin, xmax = 1e7, 1.5e10
ymin, ymax = 0.1, 10.

ax.set(xlabel='Star Age', ylabel='Quantity', xscale='log', yscale='log',
       xlim=[xmin, xmax], ylim=[ymin, ymax])
ax.vlines(sun_age, ymin, ymax, label='4.57Gyr (age of sun)')
leg = ax.legend(loc=3, fontsize=16)
leg.draw_frame(False)   
plt.savefig('mist_sun_vs_time.pdf')

