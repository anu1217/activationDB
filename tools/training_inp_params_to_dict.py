import numpy as np
from itertools import product


def write_training_params_dict(min_on_time, rel_on_time_factors,
                               flux_norm_factors, flux_files, time_unit):
    """
    This function takes a series of input parameters and converts them into a dictionary of the form below.
    This dictionary can be used to build a single-line schedule with a single-line pulse history. A separate
    dictionary for each viable combination of input parameters is constructed, and each dictionary is stored
    in a numpy array of shape len(rel_on_time_factors) x len(flux_norm_factors) x len(flux_files).

    The total amount of fluence is defined by the product of the minimum on-time
    and the maximum flux magnitude. If a combination of on-time and flux scaling factors results in
    the total fluence being exceeded, the corresponding entry in the numpy array of dictionaries is
    set to None.
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
    :param: min_on_time (minimum total amount of time (float) during which the flux is nonzero)
    :param: rel_on_time_factors (iterable of factors (float) that scale the minimum on-time)
    :param: flux_norm_factors (iterable of factors (float) that scale the maximum flux magnitude)
    :param: flux_files (iterable of paths (str) to flux files)
    :param: time_unit (unit (str) of pulse length)
    """
    training_child_dicts = np.ndarray((len(rel_on_time_factors), len(flux_norm_factors), len(flux_files)), dtype=object)
    for (rel_on_time_factor_idx, rel_on_time_factor), (flux_norm_factor_idx, flux_norm_factor), (flux_file_idx, flux_file) in product(
                enumerate(rel_on_time_factors),
                enumerate(flux_norm_factors),
                enumerate(flux_files)):
        if rel_on_time_factor * flux_norm_factor > 1:
            training_child_dicts[rel_on_time_factor_idx, flux_norm_factor_idx, flux_file_idx] = None
        else:
            training_child_dicts[rel_on_time_factor_idx, flux_norm_factor_idx, flux_file_idx] = {
                                        'type': 'pulse_entry',
                                        'pulse_length': min_on_time * rel_on_time_factor,
                                        'pulse_length_unit': time_unit,
                                        'flux_filepath': flux_file,
                                        'flux_norm': flux_norm_factor,
                                        'pulse_history': [(1, 0, 's')],
                                        'delay_dur': 0.0,
                                        'delay_dur_unit': 's'}
    return training_child_dicts
