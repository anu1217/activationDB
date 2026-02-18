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


