import argparse
import yaml
import numpy as np
import openmc
import json
import sys
import os


def search_for_match(top_dict, search_str):
    '''
    Takes a nested top dictionary and searches all sub-dictionaries for matches with a provided string.
    '''
    matches = []
    for key, value in top_dict.items():

        if key.startswith(search_str):
            matches.append(value)

        elif isinstance(value, dict):
            match_res = search_for_match(value, search_str)
            matches.extend(match_res)

    return matches


def calc_time_params_from_out(spp, run_dicts):
    '''
    Calculates an irradiation time array using parameters from ALARA output files.
    It is assumed that the active burn time is constant over all runs.
    input: 
        run_dicts : dictionary containing sub-dictionaries organized by duty cycle, whose values are
        dictionaries with keys = run labels organized by number of pulses, and values = paths to the
        corresponding output file.
    outputs:
        active_burn_time: total amount of active irradiation time in seconds (float)
        t_irr_arr : array of total irradiation times of shape (# duty cycles x # pulses)
    '''
    t_irr_arr = []
    for run_dict in run_dicts.values():
        for out_path in run_dict.values():
            lines = spp.read_out(out_path)
            pulse_dict = spp.read_pulse_histories(lines)
            sch_dict = spp.make_nested_dict(lines)
            num_pulses = eval(
                search_for_match(pulse_dict,
                                 'num_pulses_all_levels')[0].strip(" []"))
            pulse_length = search_for_match(sch_dict, 'pe_dur')[0]
            abs_dwell_time = eval(
                search_for_match(pulse_dict,
                                 'delay_seconds_all_levels')[0].strip(" []"))
            active_burn_time = num_pulses * pulse_length
            t_irr_arr.append(active_burn_time + abs_dwell_time *
                             (num_pulses - 1))
    t_irr_arr = np.array(t_irr_arr).reshape(len(run_dicts.values()),
                                            len(run_dict.values()))
    return active_burn_time, t_irr_arr


def open_flux_file(flux_file):
    with open(flux_file, 'r') as flux_data:
        flux_lines = flux_data.readlines()
    return flux_lines


def parse_flux_lines(flux_lines):
    '''
    Uses provided list of flux lines and group structure applied to the run to create an array of flux entries, with:
    # rows = # of intervals = total # flux entries / # group structure bins
    # columns = # group structure bins
    input : flux_lines (list of lines from ALARA flux file)
    output : flux_array (numpy array of shape # intervals x number of energy groups)
    '''
    energy_bins = openmc.mgxs.GROUP_STRUCTURES['VITAMIN-J-175']
    all_entries = np.array(' '.join(flux_lines).split(), dtype=float)
    if len(all_entries) == 0:
        raise Exception("The chosen flux file is empty.")
    num_groups = len(energy_bins) - 1
    num_intervals = len(all_entries) // num_groups
    if len(all_entries) % num_groups != 0:
        raise Exception("The number of intervals must be an integer.")
    flux_array = all_entries.reshape(num_intervals, num_groups)
    return flux_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_yaml',
                        default="iter_dt_out.yaml",
                        help="Path (str) to YAML containing inputs")
    parser.add_argument(
        '--spp_path',
        '-sp',
        required=True,
        help="Path (str) to folder containing sched_post_processor.py")
    args = parser.parse_args()
    return args


def read_yaml(yaml_arg):
    '''
    input:
        yaml_arg : output of parse_args() corresponding to args.db_yaml
    '''
    with open(yaml_arg, 'r') as yaml_file:
        inputs = yaml.safe_load(yaml_file)
    return inputs


def main():
    args = parse_args()
    inputs = read_yaml(args.db_yaml)

    spp_path = args.spp_path
    sys.path.insert(0, spp_path)
    import sched_post_processor as spp

    flux_file = inputs['flux_file']
    run_dicts = inputs['run_dicts']
    flux_lines = open_flux_file(flux_file)
    flux_array = parse_flux_lines(flux_lines)

    active_burn_time, t_irr_arr = calc_time_params_from_out(spp, run_dicts)

    total_flux = np.sum(flux_array,
                        axis=1)  #sum over the bin widths of flux array
    # normalize flux spectrum by the total flux in each interval
    norm_flux_arr = flux_array / total_flux.reshape(
        len(total_flux), 1)  # 2D array of shape num_intervals x num_groups
    # for each interval, calculate flux averaged over time elapsed between the start of the 1st pulse and the end of the last
    avg_flux_arr = np.multiply.outer(
        total_flux, active_burn_time / t_irr_arr
    )  # array of shape num_intervals x len(duty_cycles) x len(num_pulses)


if __name__ == "__main__":
    main()
