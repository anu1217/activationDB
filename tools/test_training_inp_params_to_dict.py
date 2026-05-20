import pytest
import training_inp_params_to_dict
import numpy as np

@pytest.mark.parametrize("on_times, flux_norm_factors, flux_files, time_unit, exp_training_child_dicts", [
    (np.array([1, 40]),
     np.array([1, 0.3]),
     np.array(['../iter_flux', '../frascati_flux']),
     'y',
    np.array([
        [
            [
                {'type': 'pulse_entry',
      'pulse_length' : 1,
      'pulse_length_unit': 'y',
      'flux_filepath' : '../iter_flux',
      'flux_norm' : 1,
      'pulse_history': [1, 0, 's'],
      'delay_dur' : 0.0,
      'delay_dur_unit' : 's'},

      {'type': 'pulse_entry',
      'pulse_length' : 1,
      'pulse_length_unit': 'y',
      'flux_filepath' : '../frascati_flux',
      'flux_norm' : 1,
      'pulse_history': [1, 0, 's'],
      'delay_dur' : 0.0,
      'delay_dur_unit' : 's'}
            ],

            [
          {'type': 'pulse_entry',
      'pulse_length' : 1,
      'pulse_length_unit': 'y',
      'flux_filepath' : '../iter_flux',
      'flux_norm' : 0.3,
      'pulse_history': [1, 0, 's'],
      'delay_dur' : 0.0,
      'delay_dur_unit' : 's'},
      
      {'type': 'pulse_entry',
      'pulse_length' : 1,
      'pulse_length_unit': 'y',
      'flux_filepath' : '../frascati_flux',
      'flux_norm' : 0.3,
      'pulse_history': [1, 0, 's'],
      'delay_dur' : 0.0,
      'delay_dur_unit' : 's'}
            ]
        ],

        [
            [
                {'type': 'pulse_entry',
      'pulse_length' : 40,
      'pulse_length_unit': 'y',
      'flux_filepath' : '../iter_flux',
      'flux_norm' : 1,
      'pulse_history': [1, 0, 's'],
      'delay_dur' : 0.0,
      'delay_dur_unit' : 's'},

      {'type': 'pulse_entry',
      'pulse_length' : 40,
      'pulse_length_unit': 'y',
      'flux_filepath' : '../frascati_flux',
      'flux_norm' : 1,
      'pulse_history': [1, 0, 's'],
      'delay_dur' : 0.0,
      'delay_dur_unit' : 's'}
            ],

            [
                 {'type': 'pulse_entry',
      'pulse_length' : 40,
      'pulse_length_unit': 'y',
      'flux_filepath' : '../iter_flux',
      'flux_norm' : 0.3,
      'pulse_history': [1, 0, 's'],
      'delay_dur' : 0.0,
      'delay_dur_unit' : 's'},

      {'type': 'pulse_entry',
      'pulse_length' : 40,
      'pulse_length_unit': 'y',
      'flux_filepath' : '../frascati_flux',
      'flux_norm' : 0.3,
      'pulse_history': [1, 0, 's'],
      'delay_dur' : 0.0,
      'delay_dur_unit' : 's'}
            ]
        ]
      ])
     )
     ])

def test_write_training_params_dict(on_times, flux_norm_factors, flux_files, time_unit, exp_training_child_dicts):
    obs_training_child_dicts = training_inp_params_to_dict.write_training_params_dict(on_times, flux_norm_factors, flux_files, time_unit)
    assert np.array_equal(obs_training_child_dicts, exp_training_child_dicts)