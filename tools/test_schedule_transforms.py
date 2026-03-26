import pytest
import schedule_transforms as st

@pytest.mark.parametrize( "pulse_length,num_pulses,dwell_time,exp_tirr,exp_ff",
                          [
                            (1, 1, 1, 1, 1),
                            (1, 2, 1, 3, 2/3),
                            (2, 4, 6, 26, 8/26)
                          ])
def test_single_pulse_history(pulse_length, num_pulses, dwell_time, exp_tirr, exp_ff):
    obs_tirr, obs_ff = st.flatten_pulse_history(pulse_length, num_pulses, dwell_time)

    assert obs_tirr == exp_tirr
    assert obs_ff == exp_ff

@pytest.mark.parametrize( "pulse_length,num_tot_pulses,dwell_time,num_final_pulses,exp_tirr,exp_ff",
                          [
                            (1, 5, 2, 1, 10, 4/10),
                            (5, 5, 1, 1, 23, 20/23),
                            (5, 5, 1, 3, 11, 10/11)
                          ])
def test_flatten_ph_exact_pulses(pulse_length, num_tot_pulses, dwell_time, num_final_pulses, exp_tirr, exp_ff):
    obs_tirr, obs_ff = st.flatten_ph_exact_pulses(pulse_length, num_tot_pulses, dwell_time, num_final_pulses)

    assert obs_tirr == exp_tirr
    assert obs_ff == exp_ff
@pytest.mark.parametrize( "pulse_length,num_pulses,exp_tirr",
                          [
                            (1, 1, 1),
                            (1, 2, 2),
                            (10, 5, 50)
                          ])
def test_compress_pulse_history(pulse_length, num_pulses, exp_tirr):
    obs_tirr = st.compress_pulse_history(pulse_length, num_pulses)

    assert obs_tirr == exp_tirr

@pytest.mark.parametrize( "pulse_length,nums_pulses,dwell_times,exp_tot_tirr,exp_tot_ff",
                          [
                            (1, [1,1], [1,1], 1, 1),
                            (1, [2,2], [1,2], 8, 4/8),
                            (2, [1,2], [2,2], 6, 4/6)
                          ])
def test_flatten_ph_levels(pulse_length, nums_pulses, dwell_times, exp_tot_tirr, exp_tot_ff):
    obs_tot_tirr, obs_tot_ff = st.flatten_ph_levels(pulse_length, nums_pulses, dwell_times)

    assert obs_tot_tirr == exp_tot_tirr
    assert obs_tot_ff == exp_tot_ff

@pytest.mark.parametrize( "pulse_length,nums_pulses,exp_tot_tirr",
                          [
                            (1, [1,1], 1),
                            (1, [2,2], 4),
                            (2, [5,7], 70)
                          ])
def test_compress_ph_levels(pulse_length, nums_pulses, exp_tot_tirr):
    obs_tot_tirr = st.compress_ph_levels(pulse_length, nums_pulses)

    assert obs_tot_tirr == exp_tot_tirr


@pytest.mark.parametrize( "child_dicts, sched_delay_dur, sched_np, sched_ph_dt, exp_tirr, exp_ff",
                          [
                            ({'children':
                              [{
                                'type': 'schedule',
                                'sched_delay_dur': 1,
                                'nums_pulses': [1],
                                'ph_dwell_times':[1],
                                'children':
                                    [
                                    {'type': 'pulse_entry',
                                    'pulse_length': 1,
                                    'pe_delay_dur' : 1,
                                    'nums_pulses': [1],
                                    'ph_dwell_times': [1]},

                                    {'type': 'pulse_entry',
                                    'pulse_length': 1,
                                    'pe_delay_dur' : 1,
                                    'nums_pulses': [1],
                                    'ph_dwell_times': [1]}
                                    ]},
                               ]
                               },

                               0,
                               [1],
                               [0],
                               5,
                               2/5),

                            ({'children':
                              [
                                {
                                'type': 'schedule',
                                'sched_delay_dur': 5,
                                'nums_pulses': [1],
                                'ph_dwell_times':[1],
                                'children':
                                    [
                                    {'type': 'pulse_entry',
                                    'pulse_length': 10,
                                    'pe_delay_dur' : 20,
                                    'nums_pulses': [1],
                                    'ph_dwell_times': [1]},

                                    {'type': 'pulse_entry',
                                    'pulse_length': 2,
                                    'pe_delay_dur' : 10,
                                    'nums_pulses': [1],
                                    'ph_dwell_times': [1]},

                                    {'type': 'schedule',
                                    'sched_delay_dur': 2,
                                    'nums_pulses': [1],
                                    'ph_dwell_times': [1],
                                    'children':
                                        [
                                        {'type': 'pulse_entry',
                                        'pulse_length': 5,
                                        'pe_delay_dur' : 10,
                                        'nums_pulses': [1],
                                        'ph_dwell_times': [1]
                                        }
                                        ]
                                    },
                                    {'type': 'schedule',
                                    'sched_delay_dur': 2,
                                    'nums_pulses': [1],
                                    'ph_dwell_times': [1],
                                    'children':
                                        [
                                        {'type': 'pulse_entry',
                                        'pulse_length': 5,
                                        'pe_delay_dur' : 3,
                                        'nums_pulses': [1],
                                        'ph_dwell_times': [1]
                                        }
                                        ]
                                    }

                                    ]
                                },
                               ]
                               },

                               0,
                               [1],
                               [0],
                               74,
                               22/74),

                            ({'children':
                              [{
                                'type': 'schedule',
                                'sched_delay_dur': 1,
                                'nums_pulses': [2],
                                'ph_dwell_times':[0],
                                'children':
                                    [
                                    {'type': 'pulse_entry',
                                    'pulse_length': 1,
                                    'pe_delay_dur' : 1,
                                    'nums_pulses': [2],
                                    'ph_dwell_times': [0]}
                                    ]},
                               ]
                               },

                               0,
                               [1],
                               [0],
                               7,
                               4/7),

                          ])
def test_flatten_sub_sched(child_dicts, sched_delay_dur, sched_np, sched_ph_dt,
                           exp_tirr, exp_ff):
    obs_tirr, obs_ff = st.flatten_sub_sched(child_dicts, sched_delay_dur,
                                            sched_np, sched_ph_dt)

    assert obs_tirr == exp_tirr
    assert obs_ff == exp_ff
