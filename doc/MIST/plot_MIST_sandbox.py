import read_mist_models
import numpy as np, matplotlib.pyplot as plt

eep_dir = 'MIST_v1.0_feh_p0.00_afe_p0.0_vvcrit0.4_EEPS/'
eep = read_mist_models.EEP(eep_dir+'00100M.track.eep')
print('version: {}'.format(eep.version))
print('abundances: {}'.format(eep.abun))
print('rotation: {}'.format(eep.rot))
print('initial mass: {}'.format(eep.minit))
print('available columns: {}'.format(eep.hdr_list))

####################
# Plot an HR diagram:
# Following the FSPS notation, PMS:-1 ; MS:0 ; SGB+RGB:2 ; CHeB:3 ; EAGB:4 ;
# TPAGB:5 ; post-AGB:6 ; WR:9
eep.plot_HR(color='Black', phases=[0, 6], phasecolor=['Red', 'Blue'])
plt.savefig('mist_solar_HR.pdf')


####################
# Plot center mass frac (H1, He4, C12) vs age:
plt.close('all')
star_age = eep.eeps['star_age']
center_h1 = eep.eeps['center_h1']
center_he4 = eep.eeps['center_he4']
center_c12 = eep.eeps['center_c12']
plt.plot(star_age, center_h1, label='H1')
plt.plot(star_age, center_he4, label='He4')
plt.plot(star_age, center_c12, label='C12')
plt.xlabel('Star Age')
plt.ylabel('Mass Fraction')
plt.axis([1e7, 1.5e10, 1e-6, 3])
plt.xscale('log')
plt.yscale('log')
leg = plt.legend(loc=3, fontsize=16)
leg.draw_frame(False)   
plt.savefig('mist_coremass_vs_time.pdf')

####################
# Plot radius vs age:
sun_teff = 5772.
plt.close('all')
f, ax = plt.subplots()
star_age = eep.eeps['star_age']
star_radius = 10**(eep.eeps['log_R'])
star_teff = 10**(eep.eeps['log_Teff'])
star_L = 10**(eep.eeps['log_L'])
ax.plot(star_age, star_radius, label='radius [Rsun]')
ax.plot(star_age, star_L, label='luminosity [Lsun]')
ax.plot(star_age, star_teff/sun_teff, label='teff [Teffsun,5772K]')
xmin, xmax = 1e7, 1.5e10
ymin, ymax = 0.1, 10.
ax.set(xlabel='Star Age', ylabel='Quantity', xscale='log', yscale='log',
       xlim=[xmin, xmax], ylim=[ymin, ymax])
sun_age = 4.57e9
ax.vlines(sun_age, ymin, ymax, label='4.57Gyr, age of sun')
leg = ax.legend(loc=3, fontsize=16)
leg.draw_frame(False)   
plt.savefig('mist_sun_vs_time.pdf')

