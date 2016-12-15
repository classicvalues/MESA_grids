'''
Description:
    From MESA output files, create a table with:
        age, Rstar, Lstar, Mstar, Rtachocline, Mini, Mconv, Mrad, Iconv, Irad. 
    Optionally, create:
        * a pdf report of ρ(r), where each page is a saved time step.
        * an HR diagram (L vs Teff), where the dot color is age (see evolution)
Usage:
    python wrangle_output.py grid_id -spec -tests -we -dp
Args:
    grid_id: substring identifying the MESA grid data. (e.g., "0') for "grids_0" in data.
Options:
    -spec: make ρ(r) plot for specific model (for which user is prompted for
    name)
    -tests: make plots of ρ(r) tests for solar case (default false)
    -dp: make ρ(r) plots (density profiles) for all models (warning: SLOW)
    -we: write evolution tracks (TODO: implement)
'''

import mesa_reader as mr, numpy as np, pandas as pd
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages as pdf_pages
import re, os
import argparse
plt.style.use('classic')

global R_sun, M_sun, L_sun, T_eff_sun, I_sun
R_sun, M_sun, L_sun, T_eff_sun = 6.957e10, 1.988e33, 3.928e33, 5772 
    # Choi+ (2016) cgs values.
I_sun = M_sun * R_sun * R_sun
    # ignores order unity prefactor.
I_sun_num = 7.07990201e53 
    # from numerical integration of Christensen-Dalsgaard 1996
    # see email chain with Dappen & Rhodes. Purely model-determined.

def plot_evolution_tracks(masses, age_lims, strs, logs):
    age_min, age_max = age_lims[0], age_lims[1]

    plt.close('all')
    f, ax = plt.subplots()
    plt.ion()

    for i, mass in enumerate(masses):
        # make MesaData object from hist file
        h = mr.MesaData(logs+'/solar_'+str(mass)+'_history.data') 
        # extract the star_age column of the data
        ages = h.data('star_age') 
        if i == 0:
            print('Available params:\n', [name for name in h.bulk_names])
        cax = ax.scatter(h.log_Teff[(h.star_age>age_min)&(h.star_age<age_max)], 
                         h.log_L[(h.star_age>age_min)&(h.star_age<age_max)], 
                         c=h.star_age[(h.star_age>age_min)&(h.star_age<age_max)], 
                         lw=0, s=1.5, cmap='viridis', label=str(mass)+' Msun')
       
    ax.set_xlim(ax.get_xlim()[::-1])
    #ax.legend(bbox)
    cbar = f.colorbar(cax, label='Time [yr]')
    ax.set(ylabel=r'log10 L', xlabel=r'log10 Teff',
           title=strs[0])
    f.savefig(strs[1]);

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in "human order"
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def get_convective_radiative_bndry(dat, p, Z):
    '''
    Returns number of boundaries detected, and radius coordinate of 
    tacholocline's middle, up to grid's precision, in units of Msun.    

    dat: pandas DataFrame containing the stellar properties at a given 
        timestep
    p: mesa profile header (has star mass, age, etc. as attributes)
    Z: metal mass fraction (float)
    '''
    # Fine tune the upper limit for radiative-convective boundary search.
    # "Necessary" b/c at Mstar>1.1Msun, get very thin layer.
    upper_ignore_boundary = 0.99
    if p.star_mass >= 1.1 and Z < 0.003:
        upper_ignore_boundary = 0.998
    if p.star_mass == 1.2 and p.header('star_age')>1.5e9 and Z < 0.003:
        upper_ignore_boundary = 0.9999
    if p.star_mass == 1.2 and Z < 0.003 \
            and p.header('star_age')>9.5e6 and p.header('star_age')<9.9e6:
        upper_ignore_boundary = 0.9999 # could fine tune; won't

    grad = np.gradient(dat.mixing_type)
    n_boundaries = len(grad[grad != 0])/2
    # Ignore convection near the surface (in Mstar!=1.2Msun cases)
    boundary_radii = np.array(dat.radius[(grad != 0)&\
        (dat.radius/max(dat.radius) < upper_ignore_boundary)])
    if (boundary_radii.size < 4):
        return 0, np.nan
    else:
        tachocline_bounds = np.array(boundary_radii)[:4] # outer radius 1st, inner radius last
        tachocline_ideal_middle = (tachocline_bounds[0] + tachocline_bounds[3])/2.
        tachocline_grid_middle = find_nearest(np.array(dat.radius), tachocline_ideal_middle)
        return n_boundaries, tachocline_grid_middle, upper_ignore_boundary

def make_profile_report(mass, Z, star_path, profile_names, mainsub, \
        make_plot=False, make_table=True, log_x=False):
    '''
    Make table that saves relevant info (if make_table).
    Optionally, also make pdf with ρ(r) over all timesteps (if make_plot).
    '''
    minimum_age = 5e5 # yr. Only care about near ZAMS evoln and onward.

    profile_names.sort(key=natural_keys)
    out = {'age': [], 'R_star': [], 'L_star': [], 'M_star': [], 
           'R_tachocline': [], 'M_ini': [], 'M_conv': [], 'M_rad': [], 
           'I_conv': [], 'I_rad': []}

    base = '/home/luke/Dropbox/software/mesa/results/'
    grid_sub = grid_dir.split('/')[-2]
    if not os.path.exists(base+grid_sub):
        os.makedirs(base+grid_sub)
        os.makedirs(base+grid_sub+'/tables/')
        os.makedirs(base+grid_sub+'/plots/')

    if make_plot:
        sub = 'log_r' if log_x else 'linear_r'
        pdf_name = '/plots/'+mainsub+'_density_profiles_'+sub+'.pdf'
        pdf = pdf_pages(base+grid_sub+pdf_name)

    for profile in profile_names:
        p = mr.MesaData(star_path+profile) # load profile into a MesaData instance
        if p.header('star_age') > minimum_age:
            bulk = pd.DataFrame(p.bulk_data)
            wanted = ['zone', 'logRho', 'logR', 'radius', 'mixing_type',
                'dm', 'mass', 'rmid']
            dat = bulk[wanted]
            del bulk

            n_bndry, bndry, upper_ignore_boundary= \
                    get_convective_radiative_bndry(dat, p, float(Z))

            out['age'].append(p.header('star_age')) # stellar age, units: yr
            out['R_star'].append(p.photosphere_r) # stellar radius, units: Rsun
            out['L_star'].append(p.photosphere_L) # luminosity at photosphere, units: Lsun
            out['M_star'].append(p.star_mass) # stellar mass at this time, units: Msun
            out['M_ini'].append(p.initial_mass) # initial (ZAMS?) stellar mass, units: Msun
            if not (np.isnan(bndry)):
                out['R_tachocline'].append(bndry) # radius of convective/radiative bndry, accurate 
                    # to grid precision of profile. Units: Rsun
                out['M_rad'].append(max(np.array(dat.mass)[np.array(dat.radius)<bndry])) 
                    # mass in the radiative core. [Msun]
                out['M_conv'].append(max(np.array(dat.mass)[np.array(dat.radius)>=bndry])-\
                                     min(np.array(dat.mass)[np.array(dat.radius)>=bndry])) 
                    # mass in the convective envelope. [Msun]

                # Computing and writing moments of inertia:
                # I = \int r^2 dm \approx \sum_i r_i^2 m_i, where i indexes over each cell.
                # MESA moments of inertia:
                dm_conv, r_conv = np.array(dat.dm[np.array(dat.radius)>=bndry]), \
                                  np.array(dat.rmid[np.array(dat.radius)>=bndry])*R_sun
                I_conv = 2/3.*np.sum(r_conv**2 * dm_conv)
                dm_rad, r_rad = np.array(dat.dm[np.array(dat.radius)<bndry]), \
                                np.array(dat.rmid[np.array(dat.radius)<bndry])*R_sun
                I_rad = 2/3.*np.sum(r_rad**2 * dm_rad)
                out['I_conv'].append(I_conv/I_sun)
                out['I_rad'].append(I_rad/I_sun)

            if np.isnan(bndry):
                dm_conv, r_conv = np.array(dat.dm), \
                                  np.array(dat.rmid)*R_sun
                I_conv = 2/3.*np.sum(r_conv**2 * dm_conv)
                M_conv = max(np.array(dat.mass))
                I_rad, M_rad = 0, 0
                out['R_tachocline'].append(0)
                out['M_rad'].append(M_rad)
                out['I_rad'].append(I_rad/I_sun)
                out['M_conv'].append(M_conv)
                out['I_conv'].append(I_conv/I_sun)

            if make_plot:
                plt.close('all')
                f, ax = plt.subplots()
                norm = mpl.colors.BoundaryNorm(np.arange(-1,5,1)+0.5, plt.cm.RdYlGn.N)

                ymax, ymin = 4, -8
                if log_x:
                    cax = ax.scatter(dat.logR, dat.logRho, c=dat.mixing_type, cmap='RdYlGn', 
                        lw=0, norm=norm)
                    xmax, xmin = max(dat.logR), 0
                if not log_x:
                    cax = ax.scatter(10**dat.logR, dat.logRho, c=dat.mixing_type, cmap='RdYlGn', 
                        lw=0, norm=norm)
                    xmax, xmin = max(10**dat.logR), 0
                if bndry != np.nan:
                    if log_x:
                        ax.vlines(np.log10(bndry), min(dat.logRho), max(dat.logRho), \
                            colors='k', linestyles='dotted')
                    if not log_x:
                        ax.vlines(bndry, min(dat.logRho), max(dat.logRho), \
                            colors='k', linestyles='dotted')

                ax.text(0.05, 0.05, \
                    'no_mixing=0\nconvective_mixing=1\nsoftened_convective_mixing=2\n'+\
                    'overshoot_mixing=3\nsemiconvective_mixing=4\nthermohaline_mixing=5\n\n'+\
                    'number of <{:f}R/Rstar boundaries detected: {:d}'.\
                    format(upper_ignore_boundary, int(n_bndry)), \
                    ha='left', va='bottom', transform=ax.transAxes, fontsize=6, zorder=11,
                    bbox=dict(facecolor='white', edgecolor='white'))
                cbar = f.colorbar(cax, label='mixing_type', ticks=np.arange(-1,5,1))
                xlab = 'log10 (R/Rsun)' if log_x else 'R/Rsun'
                ax.text(0.05, 0.5, 'R_star/R_sun: {:.3g}'.format(p.photosphere_r)+\
                    '\nR_tachocline/R_sun: {:.3g}'.format(bndry)+\
                    '\nMESA moments of inertia:'+\
                    '\nI_rad/I_sun: {:.3g}'.format(I_rad/I_sun)+\
                    '\nI_conv/I_sun: {:.3g}'.format(I_conv/I_sun)+\
                    '\nI_rad/MR^2: {:.3g}'.format(I_rad/(M_sun*R_sun**2))+\
                    '\nI_conv/MR^2: {:.3g}'.format(I_conv/(M_sun*R_sun**2)),
                    ha='left', va='bottom', transform=ax.transAxes,
                    bbox=dict(facecolor='white', edgecolor='white'),
                    fontsize=6, zorder=10)
                ax.set(xlabel=xlab, ylabel=r'log10 rho',
                    ylim=[ymin,ymax], xlim=[xmin,xmax],
                    title='{:.2g}Msun, at age {:.4g}yr'.format(\
                    p.header('initial_mass'), p.header('star_age')))

                pdf.savefig()
                plt.close()
    pdf.close()

    if make_table:
        out = pd.DataFrame.from_dict(out)
        tab_name = '/tables/'+mainsub+'.csv'

        out.round({'age': 1, 'R_star': 4, 'L_star': 4, 'M_star': 4, 
           'R_tachocline': 4, 'M_ini': 4, 'M_conv': 4, 'M_rad': 4, 
           'I_conv': 4, 'I_rad': 4}).to_csv(\
           base+grid_sub+tab_name, 
           index=False, 
           na_rep='NaN',
           columns=['age', 'R_star', 'L_star', 'M_star', 'R_tachocline', 
               'M_ini', 'M_conv', 'M_rad', 'I_conv', 'I_rad'])


def test_profile_reports(mass, Z, star_path, profile_names, mainsub, log_x=False):
    '''
    Test plots of ρ(r) over enough timesteps to trust your results
    Input: float mass, float metallicity, length 4 list of plot limits,
    profile names list from write_profile_report, logs string, log_x bool.
    '''
    assert 0, 'fix lims calls'
    minimum_age, maximum_age = 4e9, 5.5e9 # yr. Only care about near ZAMS evoln and onward.

    profile_names.sort(key=natural_keys)

    sub = 'log_r' if log_x else 'linear_r'
    pdf_name = 'plots/test_'+mainsub+'_density_profiles_'+sub+'.pdf'
    plt.ioff()
    with pdf_pages(pdf_name) as pdf:
        for profile in profile_names:
            p = mr.MesaData(star_path+profile) # load profile into a MesaData instance
            if (p.header('star_age') > minimum_age) and (p.header('star_age') < maximum_age):
                bulk = pd.DataFrame(p.bulk_data)
                wanted = ['zone', 'logRho', 'logR', 'radius', 'mixing_type',
                        'dm', 'mass', 'rmid', 'i_rot']
                dat = bulk[wanted]
                del bulk

                f, ax = plt.subplots()
                norm = mpl.colors.BoundaryNorm(np.arange(-1,5,1)+0.5, plt.cm.RdYlGn.N)
                xvals = dat.logR if log_x else 10**dat.logR
                cax = ax.scatter(xvals, dat.logRho, c=dat.mixing_type, cmap='RdYlGn', 
                        lw=0, norm=norm, s=1, label='MESA result', zorder=50)

                n_bndry, bndry = get_convective_radiative_bndry(dat, p)
                if bndry != np.nan:
                    bdnry_val = np.log10(bndry) if log_x else bndry
                    ax.vlines(bdnry_val, min(dat.logRho), max(dat.logRho), \
                        colors='k', linestyles='dotted', zorder=8)

                ax.text(0.05, 0.05, 'no_mixing=0\nconvective_mixing=1\nsoftened_convective_mixing=2\n'+\
                        'overshoot_mixing=3\nsemiconvective_mixing=4\nthermohaline_mixing=5\n\n'+\
                        'number of <0.99R/Rstar boundaries detected: {:d}'.format(int(n_bndry)), \
                        ha='left', va='bottom', transform=ax.transAxes, fontsize=6, zorder=11,
                        bbox=dict(facecolor='white', edgecolor='white'))
                cbar = f.colorbar(cax, label='mixing_type', ticks=np.arange(-1,5,1))
                xlab = 'log10 (R/Rsun)' if log_x else 'R/Rsun'

                # Comparison data
                df = pd.read_csv('src_data/christensen_dalsgaard_1996_science_data.txt', 
                    names=['r', 'c', 'rho', 'p', 'gamma_1', 'T'], skiprows=5, delimiter=' ', 
                    index_col=False)
                ax.plot(df.r, np.log10(df.rho), color='green', label='CD96', lw=1, zorder=30)
                leg = ax.legend(fontsize=6, bbox_transform=ax.transAxes, 
                        bbox_to_anchor=(0.25, 0.35), scatterpoints=1)
                leg.get_frame().set_linewidth(0.0)

                ax.set(xlabel=xlab, ylabel=r'log10 rho (cgs)',
                        ylim=[lims[0],lims[1]], xlim=[lims[2],lims[3]],
                        title='{:.1g}Msun, at age {:.2g}yr'.format(\
                        p.header('initial_mass'), p.header('star_age')))
                #ax.minorticks_on()
                #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
                #ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
                #ax.grid(b=True, which='major', color='k', linestyle='-', lw=0.5, alpha=0.5,
                #        zorder=0)
                #ax.grid(b=True, which='minor', color='gray', linestyle='--', linewidth=0.25,
                #        alpha=0.5, zorder=-1)

                # radius of convective/radiative bndry, accurate to grid precision of profile. units: Rsun
                if not (np.isnan(bndry)):
                    # dm ! cell mass (grams)
                    # I = \int r^2 dm \approx \sum_i r_i^2 m_i, where i indexes over each cell.
                    # MESA moments of inertia:
                    dm_conv, r_conv = np.array(dat.dm[np.array(dat.radius)>=bndry]), \
                                      np.array(dat.rmid[np.array(dat.radius)>=bndry])*R_sun
                    I_conv = 2/3. * np.sum(r_conv**2 * dm_conv)
                    dm_rad, r_rad = np.array(dat.dm[np.array(dat.radius)<bndry]), \
                                    np.array(dat.rmid[np.array(dat.radius)<bndry])*R_sun
                    I_rad = 2/3. * np.sum(r_rad**2 * dm_rad)
                    # Moments of inertia if I divide all radius coords in mesa by 1.02:
                    I_conv_fudge = 2/3. * np.sum((r_conv/1.02)**2 * dm_conv)
                    I_rad_fudge = 2/3. * np.sum((r_rad/1.02)**2 * dm_rad)
                    # Compare w/ Christensen-Dalsgaard 96. The minor grid difference at edge
                    # should not matter (the difference matrix has 1 less element than grid)
                    dr_CD96 = -np.diff(df.r)*R_sun
                    r_CD96 = np.array(df.r)[:-1]*R_sun
                    rho_CD96 = np.array(df.rho)[:-1]
                    dm_CD96 = rho_CD96 * 4 * np.pi * r_CD96**2 * dr_CD96
                    I_CD96 = 2/3. * np.sum(dm_CD96 * r_CD96**2)

                    dr_CD96_fudge = -np.diff(df.r*1.02)
                    dm_CD96_fudge = rho_CD96 * 4 * np.pi * (r_CD96*1.02)**2 * dr_CD96
                    I_CD96_fudge = 2/3. * np.sum(dm_CD96_fudge * (r_CD96*1.02)**2)

                    ax.text(0.05, 0.45, 
                            'R_star/R_sun: {:.3g}'.format(p.photosphere_r)+\
                            '\nL_star/L_sun: {:.3g}'.format(p.photosphere_L)+\
                            '\nTeff [K] (solar: 5772): {:.4g}'.format(p.Teff)+\
                            '\nR_tachocline/R_sun: {:.3g} (lit: 0.71-0.73Rsun)'.format(bndry)+\
                            '\nMoments of inertia via (2/3)*int(dm r^2), my calcn:'+\
                            '\nMESA:'+\
                            '\n  I_rad/I_sun: {:.3g}'.format(I_rad/I_sun)+\
                            '\n  I_conv/I_sun: {:.3g}'.format(I_conv/I_sun)+\
                            '\n  I_tot/I_sun: {:.3g}'.format((I_conv+I_rad)/I_sun)+\
                            '\n  I_rad/MR^2: {:.3g} (CD96: sun\'s total is 0.074)'.format(I_rad/\
                                (M_sun*R_sun**2))+\
                            '\n  I_conv/MR^2: {:.3g}'.format(I_conv/(M_sun*R_sun**2))+\
                            '\nMoments of inertia CD96:'+\
                            '\n  I_tot: {:.3g}'.format(I_CD96/I_sun),
                            ha='left', va='bottom', transform=ax.transAxes,
                            bbox=dict(facecolor='white', edgecolor='white'),
                            fontsize=6, zorder=5)

                pdf.savefig()
                plt.close()
                
    
def write_evolution_tracks():
    masses = [0.6, 0.7, 0.8, 0.9, 1.0]
    fig_title = '0.6Msun to 1.0Msun, solar abundances,\n 1Myr to 11Gyr'
    save_path = 'plots/evolution_tracks_0.6_to_1.0Msun.pdf'
    age_min, age_max = 1e6, 1.1e10
    age_lims = [age_min, age_max]
    strs = [fig_title, save_path]

    plot_evolution_tracks(masses, age_lims, strs, logs)
    print('wrote evolution tracks')    

def write_profile_report(mass, Z, star, mainsub, run_tests, make_ρ_profile, 
    log_x=False):
    '''
    mass: of star, float in units of [Msun]
    Z: metal mass fraction (float, think relative to Zsolar=0.015)
    star: path identifying star in grid output directory
    run_tests, make_ρ_profile: boolean
    '''
    star_path = grid_dir+star+'/LOGS/'
    profile_names = [f for f in os.listdir(star_path) if ('profile' in f) & \
            ('.data' in f) & ('M'+str(mass) in f) & (str(Z) in f)]

    if run_tests:
        test_profile_reports(mass, Z, star_path, profile_names, mainsub, log_x)
        print('Ran test profiles')    
    else:
        if make_ρ_profile:
            print('Making specific density profile...')
            make_profile_report(mass, Z, star_path, profile_names, mainsub,\
                    make_plot=make_ρ_profile, make_table=False, log_x=False)
            print('Done.')
        else:
            make_profile_report(mass, Z, star_path, profile_names, mainsub,\
                    make_plot=make_ρ_profile, make_table=True, log_x=False)
            print('Wrote table and(or) plots for M={:.2g}Msun, Z={:.2g}'.format(mass, Z))    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('args', nargs='+')
    parser.add_argument('-tests', action='store_true', default=False,
        dest='run_tests_flag')
    parser.add_argument('-dp', action='store_true', default=False,
        dest='make_all_ρ_profiles')
    parser.add_argument('-spec', action='store_true', default=False,
        dest='make_specific_ρ_profile')
    parser.add_argument('-we', action='store_true', default=False,
        dest='write_evoln_tracks_flag')
    ao = parser.parse_args()
    assert len(ao.args) == 1, 'First arg is grid ID of relevant data file'

    base_data = '/home/luke/Dropbox/software/mesa/data/grid_'
    global grid_dir
    grid_dir = base_data + ao.args[0] + '/'
    star_names = np.sort([f for f in os.listdir(grid_dir) if ('_M' in f) and 
        ('_Z' in f)])

    if ao.make_specific_ρ_profile:
        mass = input('Enter star mass: [e.g., 1.2]: ')         
        metal_mass_frac = input('Enter Z: [e.g., 0.0015]: ')         
        star = [s for s in star_names if mass in s and metal_mass_frac in s][0]
        print(star)
        mainsub = 'M'+str(mass)+'_Z'+str(metal_mass_frac)
        write_profile_report(mass, metal_mass_frac, star, mainsub, ao.run_tests_flag, 
                ao.make_specific_ρ_profile)
    else:
        for star in star_names:
            mass = float(star.split('_')[1][1:])
            metal_mass_frac = float(star.split('_')[2][1:])

            mainsub = 'M'+str(mass)+'_Z'+str(metal_mass_frac)
            write_profile_report(mass, metal_mass_frac, star, mainsub, ao.run_tests_flag, 
                    ao.make_all_ρ_profiles)

    print('all done.')

if __name__ == '__main__':
    main()
