import numpy as np
import schedule_transforms as st

"""
Create dwell times for pulse histories of varying complexity.
"""

def calc_simple_ph_time_params(fluences, duty_cycles, nums_pulses):
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
     

    
def calc_time_params_complex_ph(fluences, rel_dcs_lists, nums_multi_pulses, calc_dwell_time):
    '''

    rel_dcs_lists and nums_multi_pulses must have the same shape (chosen number x number of levels in ph)
    fluences has an arbitrary length
    pulse_lengths: 2D numpy array of shape (len(fluences), len(nums_multi_pulses))
    '''
    accumulated_fluence = 0
    accumulated_tot_dur = 0
    all_dwell_times = np.array(len(rel_dcs_lists))
    pulse_lengths = np.array(len(rel_dcs_lists))
    for fluence in fluences:
        for rel_dcs_list_idx, rel_dc_list in enumerate(rel_dcs_lists):
            pulse_length = fluence / sum(nums_multi_pulses[rel_dcs_list_idx])
            pulse_lengths[rel_dcs_list_idx] = pulse_length
            lvl_dwell_times = np.array(len(rel_dc_list))
            for rel_dc_idx, rel_dc in rel_dc_list:
                accumulated_off_time = accumulated_tot_dur - accumulated_fluence
                lvl_dwell_times[rel_dc_idx] = calc_dwell_time(
                    accumulated_fluence=accumulated_fluence,
                    rel_dc = rel_dc,
                    pulse_length = pulse_length,
                    n_pulses = nums_multi_pulses[rel_dcs_list_idx][rel_dc_idx],
                    accumulated_off_time=accumulated_off_time
                    )
        
                accumulated_tot_dur, accumulated_fluence = st.flatten_ph_levels(pulse_length, list(zip((nums_multi_pulses[rel_dcs_list_idx], lvl_dwell_times))))
            all_dwell_times[rel_dcs_list_idx] = lvl_dwell_times
    return pulse_lengths, all_dwell_times

# avg rel dc:
def calc_dwell_time_avg_rel_dc(accumulated_fluence, rel_dc, pulse_length, n_pulses, accumulated_off_time):
    dwell_time = (accumulated_fluence + pulse_length * n_pulses,
                                 - accumulated_fluence * rel_dc
                                 - pulse_length * n_pulses * rel_dc
                                 - accumulated_off_time * rel_dc)
    return dwell_time

# rel dc:
def calc_dwell_time_rel_dc(accumulated_fluence, rel_dc, pulse_length, accumulated_off_time, **kwargs):
    dwell_time = (accumulated_fluence + pulse_length)/rel_dc - (accumulated_fluence + pulse_length + accumulated_off_time)
    return dwell_time