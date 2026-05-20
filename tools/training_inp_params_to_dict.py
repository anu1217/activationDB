import numpy as np
from itertools import product

def write_training_params_dict(on_times, flux_norm_factors, flux_files, time_unit):
  """
  This function takes a series of input parameters and converts them into a dictionary of the form below.
  This dictionary can be used to build a single-line schedule with a single-line pulse history.
  [
  {'type': 'pulse_entry',
      'pulse_length': (float),
      'pulse_length_unit': (str),
      'flux_filepath' : (str),
      'flux_norm' : (float),
      'pulse_history': (iterable of (int, float, str)),
      'delay_dur' : (float),
      'delay_dur_unit': (str)
  }
  ]
  :param: on_times (iterable of times (float) during which the flux is nonzero)
  :param: flux_norm_factors (iterable of factors (float) that scale the maximum flux magnitude)
  :param: flux_files (iterable of paths (str) to flux files)
  :param: time_unit (unit (str) of pulse length) 
  """
  training_child_dicts = np.ndarray((len(on_times), len(flux_norm_factors), len(flux_files)), dtype=object)
  for (on_time_idx, on_time), (flux_norm_factor_idx, flux_norm_factor), (flux_file_idx, flux_file) in product(
      enumerate(on_times), 
      enumerate(flux_norm_factors), 
      enumerate(flux_files)
      ):
      training_child_dicts[on_time_idx, flux_norm_factor_idx, flux_file_idx] = {'type': 'pulse_entry',
      'pulse_length' : on_time,
      'pulse_length_unit': time_unit,
      'flux_filepath' : flux_file,
      'flux_norm' : flux_norm_factor,
      'pulse_history': [1, 0, 's'],
      'delay_dur' : 0.0,
      'delay_dur_unit' : 's'}
  return training_child_dicts    