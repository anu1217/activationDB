import pytest
import training_inp_params_to_dict
import numpy as np

@pytest.mark.parametrize("min_on_time, rel_on_time_factors, flux_norm_factors, flux_files, time_unit, exp_training_child_dicts", [
    (1,
    np.array([2, 4]),
     np.array([1, 0.3]),
     np.array(['../iter_flux', '../frascati_flux']),
     'y',
    np.array([
        [
            [
            None,
            None
            ],

            [
            {'type': 'pulse_entry',
      'pulse_length' : 2,
      'pulse_length_unit': 'y',
      'flux_filepath' : '../iter_flux',
      'flux_norm' : 0.3,
      'pulse_history': [(1, 0, 's')],
      'delay_dur' : 0.0,
      'delay_dur_unit' : 's'},
      
      {'type': 'pulse_entry',
      'pulse_length' : 2,
      'pulse_length_unit': 'y',
      'flux_filepath' : '../frascati_flux',
      'flux_norm' : 0.3,
      'pulse_history': [(1, 0, 's')],
      'delay_dur' : 0.0,
      'delay_dur_unit' : 's'}
            ]
        ],

        [
            [
            None,
            None
            ],

            [
            None,
            None
            ]
        ]
      ])
     )
     ])

def test_write_training_params_dict(min_on_time, rel_on_time_factors, flux_norm_factors, flux_files, time_unit, exp_training_child_dicts):
    obs_training_child_dicts = training_inp_params_to_dict.write_training_params_dict(min_on_time, rel_on_time_factors, flux_norm_factors, flux_files, time_unit)
    assert np.array_equal(obs_training_child_dicts, exp_training_child_dicts)