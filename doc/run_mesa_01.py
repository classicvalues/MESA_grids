#!/usr/bin/python

'''
This script:
    * make MIST inlists with different stellar masses and metallicities.
    * run mesa jobs
'''
import re, subprocess, os, sys
import numpy as np

global python_ver
python_ver = sys.version_info[0]

def run_script(script, stdin=None):
    """Returns (stdout, stderr), raises error on non-zero return code"""
    # Note: by using a list here (['bash', ...]) you avoid quoting issues, as the 
    # arguments are passed in exactly this order (spaces, quotes, and newlines won't
    # cause problems):
    proc = subprocess.Popen(['bash', '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise ScriptException(proc.returncode, stdout, stderr, script)
    return stdout, stderr

class ScriptException(Exception):
    def __init__(self, returncode, stdout, stderr, script):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        Exception.__init__('Error in script')

def set_inlist_params(desired_mass, desired_Z, desired_abund, log_path):
    assert type(desired_mass) == float
    assert type(desired_Z) == float
    assert (desired_abund.size == 4) & (type(desired_abund)==np.ndarray)
    abund_str = ['!INITIAL_H1!', '!INITIAL_H2!', '!INITIAL_HE3!',
            '!INITIAL_HE4!']

    fr = open('inlist_default', 'r')
    lines = fr.readlines()
    fr.close()
    for ix, l in enumerate(lines):
        if ('!MASS!' in l):
            existing_mass = re.findall('\d+\.\d+', l)
            lines[ix] = l.replace(existing_mass[0], str(desired_mass))
    for ix, l in enumerate(lines):
        if ('!MASS!' in l and ('\'M' in l or '_M' in l)):
            existing_mass = re.findall('M\d+\.\d+', l)
            lines[ix] = l.replace(existing_mass[0], 'M'+str(desired_mass))
    for ix, l in enumerate(lines):
        if ('INITIAL_Z' in l):
            existing_Z = re.findall('_Z0\.\d*', l)
            lines[ix] = l.replace(existing_Z[0], '_Z'+str(desired_Z))
    for ix, l in enumerate(lines):
        if ('for initial Z of' in l):
            existing_Z = re.findall('of _0\.\d*_end', l)
            lines[ix] = l.replace(existing_Z[0], 'of _'+str(desired_Z)+'_end')
    for ix, l in enumerate(lines):
        for j, a_str in enumerate(abund_str):
            if (a_str in l):
                existing_abund = re.findall(a_str.lower().split('!')[1]+ ' = 0\.\d*', l)
                lines[ix] = l.replace(existing_abund[0],
                        a_str.lower().split('!')[1]+ ' = '+str(desired_abund[j]))

    with open('INLISTS/inlist_M'+str(desired_mass)+'_Z'+str(desired_Z), 'w+') as f:
        f.writelines(lines)
    with open('inlist_to_run', 'w+') as f:
        f.writelines(lines)

    if python_ver==3:
        print('Rewrote inlists with desired mass of {:.1g} Msun'.format(desired_mass))
    if python_ver==2:
        print('Rewrote inlists with desired mass of %.1f Msun' % desired_mass)

def get_initial_abundances_given_Z(Z, solar_special=False):
    '''
    Follow exact prescription from Sec3.1 of Choi+ 2016.
    Sec4 : change abundance prescription to match sun if you want to replicate
    sun.
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
    return initial_h1, initial_h2, initial_he3, initial_he4


if __name__ == '__main__':
    Z = 0.0150 # special solar calibration
    solar_special = True # flag for special solar calibration.

    prompt_str = 'Desired stellar masses (comma-separated, e.g. 0.8,1.0) [Msun]: '
    #l = input(prompt_str) if python_ver==3 else raw_input(prompt_str)
    l = '0.7'
    desired_masses = np.sort(list(map(float, l.split(','))))

    ini_h1, ini_h2, ini_he3, ini_he4 = get_initial_abundances_given_Z(Z, solar_special)
    ini_abunds = np.array([ini_h1, ini_h2, ini_he3, ini_he4])

    single_run_flag = True if desired_masses.size==1 else False
    desired_masses = np.append(desired_masses, np.nan) if single_run_flag \
            else desired_masses

    for ind, desired_mass in enumerate(desired_masses):
        if (ind==1) and (single_run_flag):
            continue
        log_path = 'LOGS/M'+str(desired_mass)+'_Z'+str(Z)+'/'
        if not os.path.exists(log_path[:-1]):
            os.makedirs(log_path[:-1])

        set_inlist_params(float(desired_mass), Z, ini_abunds, log_path)

        if python_ver==3:
            print('Running mass: {:.2g}Msun,\t Metal mass fraction (Z): {:.2g}.'.format(\
                desired_mass, Z))
        if python_ver==2:
            print('Running mass: %.2g Msun,\t Metal mass fraction (Z): %.2g.' % \
                (desired_mass, Z))

        run_script('./rn > '+log_path+'M'+str(desired_mass)+'_Z'+str(Z)+'_out.txt')
        print('./rn > '+log_path+'M'+str(desired_mass)+'_Z'+str(Z)+'_out.txt')
