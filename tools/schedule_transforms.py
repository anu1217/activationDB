def flatten_pulse_history(pulse_length, num_pulses, dwell_time):
    """
    Apply the flux flattening approximation to a series of pulses.

    Consider a series of pulses of uniform magnitude and duration, separated by
    a dwell time with zero flux and a uniform duration.  The flux flattening
    approximation calculates an equivalent steady state flux magnitude that
    preserves both the total time between the beginning of the first pulse and
    the end of the last pulse, and the total fluence.

    :param pulse_length: (float) the duration of each pulse
    :param num_pulses: (int) the number of pulses
    :param dwell_time: (float) the duration of the gap between each pulse
    """
    t_irr = (num_pulses-1) * (pulse_length + dwell_time) + pulse_length
    flux_factor = num_pulses * pulse_length / t_irr

    return t_irr, flux_factor

def flatten_ph_levels(pulse_length, nums_pulses, dwell_times):
    '''
    Apply the flattening algorithm to all levels of a multi-level pulsing history
    with a single-level schedule block.  
    
    :param pulse_lengths: active irradiation time from schedule block
    :param nums_pulses: (iterable) number of pulses at each level
    :param dwell_times: (iterable) the duration of the gap between each pulse at each level
    '''
    tot_ph_ff = 1
    tot_ph_t_irr = pulse_length
    for num_pulses, dwell_time in zip(nums_pulses, dwell_times):
        tot_ph_t_irr, ff = flatten_pulse_history(tot_ph_t_irr, num_pulses, dwell_time)
        tot_ph_ff *= ff
    return tot_ph_t_irr, tot_ph_ff

def read_sched_entry(pulse_length, nums_pulses, ph_dwell_times, sched_dwell_time):
    ph_t_irr, ph_ff = flatten_ph_levels(pulse_length, nums_pulses, ph_dwell_times)
    sched_entry_t_irr = ph_t_irr + sched_dwell_time
    sched_entry_active_burn_time = ph_ff * ph_t_irr
    return sched_entry_t_irr, sched_entry_active_burn_time

def calc_simple_sched_flattened_params(pulse_lengths, sched_dwell_times, nums_pulses, ph_dwell_times):
    '''
    Calculate irradiation time and flux factor for a schedule that uses a single pulse history in all entries.
    This method does not account for sub-schedules.
    
    :param pulse_lengths: (iterable) of pulse lengths from the schedule entries
    :param sched_dwell_times: (iterable) of dwell times from the schedule entries
    :param nums_pulses: (iterable) number of pulses at each pulsing level
    :param ph_dwell_times: (iterable) the duration of the gap between each pulse at each pulsing level
    '''
    tot_active_burn_times = 0
    tot_sched_t_irr = 0
    for pulse_length, sched_dwell_time in zip(pulse_lengths, sched_dwell_times[:-1] + [0]): # ignore last schedule entry's dwell time
        sched_entry_t_irr, sched_entry_active_burn_time = read_sched_entry(pulse_length, nums_pulses, ph_dwell_times, sched_dwell_time)
        tot_sched_t_irr += sched_entry_t_irr
        tot_active_burn_times += sched_entry_active_burn_time
    tot_sched_ff = tot_active_burn_times / tot_sched_t_irr   
    return tot_sched_t_irr, tot_sched_ff