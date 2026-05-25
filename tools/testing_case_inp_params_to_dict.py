import numpy as np
from itertools import product

'''
The following dictionary (child_dict) is the representative data structure used to
store and access pulse history-related information for ALARA simulations. This script
populates this dictionary using user-specified information about the pulse history.

Because this script is intended to store information for the testing cases,
flux variation is introduced through duty cycles, and the flux normalization
factor is taken to be 1.

  {'type': 'pulse_entry',
      'pulse_length': (float),
      'pulse_length_unit': (str),
      'flux_filepath' : (str),
      'flux_norm' : (float),
      'pulse_history': (iterable of (int, float, str)),
      'delay_dur' : (float),
      'delay_dur_unit': (str)
  }
'''
def calc_testing_simple_ph_time_params(fluences, duty_cycles, nums_pulses):
    '''
    Calculates the pulse lengths and pulse dwell times given a set of fluences/on-times,
    duty cycles, and pulse numbers. These parameters are calculated for a single-level
    pulse history.
    input:
    :param: fluences: iterable of times (float) where the system experiences nonzero flux
    :param: duty_cycles: iterable of duty cycles (float), defined as pulse length / (pulse length + dwell time)
    For each value, 0 < duty cycle <= 1
    :param: nums_pulses: iterable of the number (int) of pulses the on-time is divided into

    output:
    pulse_lengths: 2D numpy array of shape (len(fluences), len(nums_pulses))
    abs_dwell_times: 3D numpy array of shape (len(duty_cycles), len(fluences), len(nums_pulses))
    '''
    pulse_lengths = np.outer(fluences, 1/nums_pulses)
    rel_dwell_times = (1 - duty_cycles) / duty_cycles
    abs_dwell_times = np.multiply.outer(rel_dwell_times, pulse_lengths)
    return pulse_lengths, abs_dwell_times

def populate_simple_child_dict(flux_filepaths, pulse_lengths, nums_pulses, abs_dwell_times, time_unit):
    '''
    Populates the child dictionary structure using the pulse lengths and dwell times calculated in
    calc_testing_simple_ph_time_params().
    input:
    flux_filepaths: iterable of paths (str) to files containing flux data
    time_unit: unit (str) of fluence time

    output:
    testing_pe_array: 4D numpy array of shape (len(flux_filepaths), len(duty_cycles), len(fluences), len(nums_pulses))
    '''
    testing_pe_array = np.empty((len(flux_filepaths),) + abs_dwell_times.shape, dtype=object)
    for (flux_idx, flux_filepath), ((duty_cycle_idx, fluence_idx, num_pulse_idx), abs_dwell_time) in product(
        enumerate(flux_filepaths), np.ndenumerate(abs_dwell_times)
        ):
        testing_pe_array[flux_idx, duty_cycle_idx, fluence_idx, num_pulse_idx] = {
        'type': 'pulse_entry',
        'pulse_length': pulse_lengths[fluence_idx, num_pulse_idx],
        'pulse_length_unit': time_unit,
        'flux_filepath' : flux_filepath,
        'flux_norm': 1.0,
        'pulse_history': [nums_pulses[num_pulse_idx], abs_dwell_time, time_unit],
        'delay_dur' : 0.0,
        'delay_dur_unit': time_unit
        }
    return testing_pe_array    