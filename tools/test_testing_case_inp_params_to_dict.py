import numpy as np
import pytest
import testing_case_inp_params_to_dict as tcipd

@pytest.mark.parametrize("fluences, duty_cycles, nums_pulses, exp_pulse_lengths, exp_abs_dwell_times", [
                            (np.array([5, 10]),
                             np.array([1, 0.2]),
                             np.array([2, 5]),
                             np.array([[5/2, 5/5], 
                                       [10/2, 10/5]]),
                             np.array([
                                 [[0, 0], [0, 0]],
                                 [[(0.8/0.2)*5/2, (0.8/0.2)*5/5], [(0.8/0.2)*10/2, (0.8/0.2)*10/5]]
                                 ])          
                            )
                          ])

def test_calc_testing_simple_ph_time_params(fluences, duty_cycles, nums_pulses, exp_pulse_lengths, exp_abs_dwell_times):
    obs_pulse_lengths, obs_abs_dwell_times = tcipd.calc_testing_simple_ph_time_params(fluences, duty_cycles, nums_pulses)
    assert np.array_equal(obs_pulse_lengths, exp_pulse_lengths)
    assert np.array_equal(obs_abs_dwell_times, exp_abs_dwell_times)

@pytest.mark.parametrize("flux_filepaths, pulse_lengths, nums_pulses, abs_dwell_times, time_unit, exp_testing_pe_array", [
                            (np.array(['../iter', '../frascati']),
                             np.array([[5, 6], [1, 2]]),
                             np.array([2, 4]),
                             np.array([[[0.3, 0.2], [0.1, 1]], [[0.5, 0.1], [0.4, 0.9]]]),
                             'y',
                             np.array([
                                 [[
                                     [{'type': 'pulse_entry', 
                                     'pulse_length': 5, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../iter', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [2, 0.3, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'},

                                     {'type': 'pulse_entry', 
                                     'pulse_length': 6, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../iter', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [4, 0.2, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'}], 

                                     [{'type': 'pulse_entry', 
                                     'pulse_length': 1, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../iter', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [2, 0.1, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'},
                                     
                                     {'type': 'pulse_entry', 
                                     'pulse_length': 2, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../iter', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [4, 1, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'}]],

                                     [[{'type': 'pulse_entry', 
                                     'pulse_length': 5, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../iter', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [2, 0.5, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'},

                                     {'type': 'pulse_entry', 
                                     'pulse_length': 6, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../iter', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [4, 0.1, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'}],

                                     [{'type': 'pulse_entry', 
                                     'pulse_length': 1, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../iter', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [2, 0.4, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'},
                                     
                                     {'type': 'pulse_entry', 
                                     'pulse_length': 2, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../iter', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [4, 0.9, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'}]]],



                                     [[[{'type': 'pulse_entry', 
                                     'pulse_length': 5, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../frascati', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [2, 0.3, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'},

                                     {'type': 'pulse_entry', 
                                     'pulse_length': 6, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../frascati', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [4, 0.2, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'}], 

                                     [{'type': 'pulse_entry', 
                                     'pulse_length': 1, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../frascati', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [2, 0.1, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'},
                                     
                                     {'type': 'pulse_entry', 
                                     'pulse_length': 2, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../frascati', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [4, 1, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'}]],

                                     [[{'type': 'pulse_entry', 
                                     'pulse_length': 5, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../frascati', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [2, 0.5, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'},

                                     {'type': 'pulse_entry', 
                                     'pulse_length': 6, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../frascati', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [4, 0.1, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'}],

                                     [{'type': 'pulse_entry', 
                                     'pulse_length': 1, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../frascati', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [2, 0.4, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'},
                                     
                                     {'type': 'pulse_entry', 
                                     'pulse_length': 2, 
                                     'pulse_length_unit': 'y',
                                     'flux_filepath': '../frascati', 
                                     'flux_norm': 1.0, 
                                     'pulse_history': [4, 0.9, 'y'], 
                                     'delay_dur': 0.0, 
                                     'delay_dur_unit': 'y'}]]]
                                     
                                 ])
                            )
                            ])

def test_populate_simple_child_dict(flux_filepaths, pulse_lengths, nums_pulses, abs_dwell_times, time_unit, exp_testing_pe_array):
    obs_testing_pe_array = tcipd.populate_simple_child_dict(flux_filepaths, pulse_lengths, nums_pulses, abs_dwell_times, time_unit)
    assert np.array_equal(exp_testing_pe_array, obs_testing_pe_array)