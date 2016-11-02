#!/usr/bin/python

'''
This script:
    * makes MIST inlists directories with different stellar masses and metallicities.
'''
import re, subprocess, os, sys
import numpy as np

global python_ver, task_id
python_ver = sys.version_info[0]
#task_id = os.environ['SLURM_ARRAY_TASK_ID']

def write_inlist(desired_mass, desired_Z, desired_abund, dir_to_write):
    assert type(desired_mass) == float
    assert type(desired_Z) == float
    assert (desired_abund.size == 4) & (type(desired_abund)==np.ndarray)
    abund_str = ['!INITIAL_H1!', '!INITIAL_H2!', '!INITIAL_HE3!',
            '!INITIAL_HE4!']

    fr = open('inlist_variable_params', 'r')
    lines = fr.readlines()
    fr.close()
    for ix, l in enumerate(lines):
        if ('!MASS!' in l):
            existing_mass = re.findall('\d+\.\d*', l)
            lines[ix] = l.replace(existing_mass[0], str(desired_mass))
    for ix, l in enumerate(lines):
        if ('!MASS!' in l and ('\'M' in l or '_M' in l)):
            existing_mass = re.findall('M\d+\.\d*', l)
            lines[ix] = l.replace(existing_mass[0], 'M'+str(desired_mass))
    for ix, l in enumerate(lines):
        if ('INITIAL_Z' in l):
            existing_Z = re.findall('_Z0\.\d*', l)
            lines[ix] = l.replace(existing_Z[0], '_Z'+str(desired_Z))
    for ix, l in enumerate(lines):
        if ('for initial metal mass fraction' in l):
            existing_Z = re.findall('of 0\.\d*', l)
            lines[ix] = l.replace(existing_Z[0], 'of _'+str(desired_Z)+'_end')
    for ix, l in enumerate(lines):
        for j, a_str in enumerate(abund_str):
            if (a_str in l):
                existing_abund = re.findall(a_str.lower().split('!')[1]+ ' = 0\.\d*', l)
                lines[ix] = l.replace(existing_abund[0],
                        a_str.lower().split('!')[1]+ ' = '+str(desired_abund[j]))

    with open(dir_to_write+'inlist_cluster', 'w+') as f:
        f.writelines(lines)

    if python_ver==3:
        print('wrote mass: {:.2g}Msun,\t Metal mass fraction (Z): {:.2g}.'.format(\
            mass, Z))
    if python_ver==2:
        print('wrote mass: %.2g Msun,\t Metal mass fraction (Z): %.2g.' % \
            (mass, Z))


def get_ini_abundances(Z, solar_special=False):
    '''
    Follow exact prescription from Sec3.1 of Choi+ 2016.
    Sec4 : change abundance prescription to match sun if you want to replicate
    sun. These are initial abundances for protostellar core.
    '''
    Y_p = 0.249 # Planck Collaboration 2015
    if not solar_special:
        Y_sun_protosolar = 0.2703 # Asplund 2009
        Z_sun_protosolar = 0.0142 # Asplund 2009
    if solar_special:
        Y_sun_protosolar = 0.2612 # Choi 2016, table 2
        Z_sun_protosolar = 0.0150 # Choi 2016, table 2

    Y = Y_p + (Y_sun_protosolar - Y_p)/Z_sun_protosolar * Z
    X = 1-Y-Z

    # Use isotope abundances from Asplund (2009), Table 3:
    initial_h1 = (1 - 2e-5) * X
    initial_h2 = 2e-5 * X
    initial_he3 = 1.66e-4 * Y
    initial_he4 = (1 - 1.66e-4) * Y

    ini_abunds = np.array([initial_h1, initial_h2, initial_he3, initial_he4])
    return ini_abunds 

def make_sym_link(mass, Z, ini_abunds, dir_to_write):
    MESA_BASE='/home/lbouma/software/mesa/base'
    MESA_RUN='/home/lbouma/software/mesa/run-grids'
    os.symlink(MESA_BASE+'/inlist', MESA_RUN+'/'+dir_to_write+'inlist')

if __name__ == '__main__':
    Z_solar = 0.0150 # special solar calibration base solar metallicity
    solar_flag = True # required for special solar calibration.

    min_mass, max_mass, mass_step = 0.4, 1.4, 0.1
    # e.g. -1,1,5 gives (0.1, 0.316, 1, 3.16, 10)*solar metal mass frac
    min_Z, max_Z, N_metallicities = -1, 1, 5
    mass_grid = np.arange(min_mass, max_mass+mass_step, mass_step)
    Z_grid = np.logspace(min_Z, max_Z, N_metallicities)*Z_solar

    k = 1
    for mass in mass_grid:
        for Z in Z_grid:
            ini_abunds = get_ini_abundances(Z, solar_flag)
            dir_to_write = str(k)+'_M'+str(mass)+'_Z'+str(Z)+'/'
            if not os.path.exists(dir_to_write[:-1]):
                os.makedirs(dir_to_write[:-1])

            write_inlist(float(mass), float(Z), ini_abunds, dir_to_write)
            make_sym_link(float(mass), float(Z), ini_abunds, dir_to_write)
            k += 1

