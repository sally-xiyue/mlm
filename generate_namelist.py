import argparse
import json
import pprint
from sys import exit
import uuid

def main():
    parser = argparse.ArgumentParser(prog='Namelist Generator')
    parser.add_argument('case_name')
    args = parser.parse_args()

    case_name = args.case_name

    if case_name == 'Isdac':
        namelist = Isdac()
    else:
        print('Not a valid case name')
        exit()

    write_file(namelist)


def Isdac():

    namelist = {}

    namelist['initial'] = {}
    namelist['initial']['SST'] = 265.0 #initial surface temperature
    namelist['initial']['dTi'] = 5.0 #temperature jump at the inversion
    namelist['initial']['rh0'] = 0.8 #Surface relative humidity
    namelist['initial']['gamma'] = 5.0/1000. #free tropospheric lapse rate
    namelist['initial']['rh'] = 0.6 #free tropospheric relative humidity
    namelist['initial']['z_top'] = 820.0 #top of mixed layer
    namelist['initial']['dSST'] = 0.0 #SST change (climate change)
    namelist['initial']['divergence'] = 5.0e-6 # LS divergence
    namelist['initial']['div_frac'] = 1.0 # fractional divergence rate

    namelist['grid'] = {}
    namelist['grid']['dz'] = 5.0

    namelist['entrainment'] = {}
    namelist['entrainment']['efficiency'] = 0.7

    namelist['radiation'] = {}
    namelist['radiation']['frequency'] = 300.0
    namelist['radiation']['n_buffer'] = 15 # adjust according to dz
    namelist['radiation']['stretch_factor'] = 1.5 # adjust according to dz

    namelist['time_stepping'] = {}
    namelist['time_stepping']['t'] = 0.0
    namelist['time_stepping']['dt_initial'] = 300.0
    # namelist['time_stepping']['dt_max'] = 60.0
    namelist['time_stepping']['t_max'] = 3600.0 * 24.0

    namelist['stats_io'] = {}
    namelist['stats_io']['frequency'] = 300.0
    # namelist['stats_io']['output_root'] = './output/'
    namelist['stats_io']['output_root'] = '/Users/xiyue/Clouds/mlm/output/data/'

    namelist['meta'] = {}
    namelist['meta']['simname'] = 'IsdacMLM_final'
    namelist['meta']['casename'] = 'IsdacMLM'

    return namelist


def write_file(namelist):

    try:
        type(namelist['meta']['simname'])
    except:
        print('Casename not specified in namelist dictionary!')
        print('FatalError')
        exit()

    namelist['meta']['uuid'] = str(uuid.uuid4())

    fh = open(namelist['meta']['simname'] + '.in', 'w')
    pprint.pprint(namelist)
    json.dump(namelist, fh, sort_keys=True, indent=4)
    fh.close()

    return


if __name__ == '__main__':
    main()
