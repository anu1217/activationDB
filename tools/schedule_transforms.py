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
    active_burn_time = nums_pulses[0] * pulse_length
    tot_t_irr, ff = flatten_pulse_history(pulse_length, nums_pulses[0], dwell_times[0])
    for num_pulses, dwell_time in zip(nums_pulses[1:], dwell_times[1:]):
        tot_t_irr, ff = flatten_pulse_history(tot_t_irr, num_pulses, dwell_time)
        active_burn_time = active_burn_time * num_pulses
    tot_ff = active_burn_time / tot_t_irr    
    return tot_t_irr, tot_ff

