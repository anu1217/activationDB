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

@pytest.mark.parametrize( "pulse_lengths,nums_pulses,dwell_times,exp_total_tirr,exp_total_ff",
                          [
                            ([1], [1], [1], 1, 1),
                            ([2, 3], [2, 3], [2, 3], 23, 11/10),
                            ([5, 5], [5, 6], [3, 3], 85, 31/24)
                          ])

def test_mult_pulse_histories(pulse_lengths, nums_pulses, dwell_times,
                              exp_total_tirr, exp_total_ff):
    obs_total_tirr, obs_total_ff = st.flatten_all_ph_levels(
        pulse_lengths, nums_pulses, dwell_times)

    assert obs_total_tirr == exp_total_tirr
    assert obs_total_ff == exp_total_ff
