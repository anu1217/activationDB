import argparse
import yaml
import numpy as np
import alara_output_processing as aop
import pandas as pd
import script_template as script_temp

'''
Runs and tests the methods in script_template.py using a series of run dictionaries
with various pulse numbers and duty cycles.
'''

def calc_time_params(active_burn_time, duty_cycle_list, num_pulses):
    '''
    Uses provided pulsing information to determine dwell time and total irradiation time.
    Assumes that the active irradiation time per pulse and dwell time between pulses both remain constant in any given simulation.
    Iterates over the number of pulses, and for each number, calculates dwell time.
    The duty cycle is defined as the pulse length / (pulse length + dwell time).
    inputs:
        active_burn_time : total active irradiation time (float) in any chosen unit
        duty_cycle_list : list of chosen duty cycles (float)
        num_pulses : list of number of pulses (int) that the active irradiation period is divided into
    '''
    pulse_lengths = active_burn_time / num_pulses
    rel_dwell_times = (1 - duty_cycle_list) / duty_cycle_list
    abs_dwell_times = np.outer(rel_dwell_times, pulse_lengths)
    t_irr_arr = active_burn_time + abs_dwell_times * (num_pulses - 1)
    return t_irr_arr

def write_to_adf(run_dicts):
    adf_data = []
    for run_dict in run_dicts:
        lib = aop.DataLibrary()
        adf = lib.make_entries(run_dicts[run_dict])
        adf_data.append(adf)
    adf = pd.concat(adf_data)  
    return adf

def assign_adf_tirr_arr_mod(adf, t_irr_arr, num_pulses, duty_cycles):
    '''
    Maps each of the number of pulses and duty cycle values from the run label to an iterator,
    from which the corresponding value of the irradiation time is identified. The irradiation
    time array is added to a new column in the adf by running the map_adf_flux_tirr() method from
    script_template.py.
    '''
    pulse_num_dc = adf["run_lbl"].str.extract(r"_(\d+)p_(\d+)_").astype(int)
    # Map num_pulses and duty_cycles to an index
    pulse_idx = pulse_num_dc[0].map(
        {pulse_num: i
         for i, pulse_num in enumerate(num_pulses)})
    duty_cycle_idx = (pulse_num_dc[1] / 100).map(
        {duty_cycle: i
         for i, duty_cycle in enumerate(duty_cycles)})
    t_irr_arr_mod = t_irr_arr.T[pulse_idx.to_numpy(),
                                duty_cycle_idx.to_numpy()]
    adf["t_irr"] = t_irr_arr_mod
    return adf