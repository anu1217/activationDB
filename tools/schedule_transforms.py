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
    t_irr_flat = (num_pulses-1) * (pulse_length + dwell_time) + pulse_length
    flux_factor_flat = num_pulses * pulse_length / t_irr_flat

    return t_irr_flat, flux_factor_flat

def compress_pulse_history(pulse_length, num_pulses):
    '''
    Applies the compression algorithm to a series of pulses. This algorithm
    preserves the total active irradiation time between the start of the first
    and end of the last pulse, and the total fluence.
    
    :param pulse_length: (float) the duration of each pulse
    :param num_pulses: (int) the number of pulses
    '''
    t_irr_comp = num_pulses * pulse_length

    return t_irr_comp


def flatten_ph_exact_pulses(pulse_length, num_init_pulses, dwell_time,
                            final_pulses):
    '''
    Applies the flattening approximation to a series of pulses. Preserves an arbitrary
    number of final pulses, and the total amount of time elapsed between the 
    start of the first of the initial set of pulses and the end of the last. The
    set of final pulses is considered to be exact in duration and delay time as the initial set.

    :param pulse_length: (float) the duration of each initial pulse
    :param num_init_pulses: (int) the number of initial pulses
    :param dwell_time: (float) the duration of the gap between each initial pulse
    :param final_pulses: (int) the number of final pulses
    '''
    t_irr = (num_init_pulses - final_pulses) * pulse_length + (
        num_init_pulses - final_pulses - 1) * dwell_time
    flux_factor = (num_init_pulses - final_pulses) * pulse_length / t_irr
    return t_irr, flux_factor


def flatten_ph_levels(pulse_length, nums_pulses, dwell_times):
    '''
    Apply the flattening algorithm to all levels of a multi-level pulsing history
    with a single-level schedule block.  
    
    :param pulse_lengths: active irradiation time from schedule block
    :param nums_pulses: (iterable) number of pulses at each level
    :param dwell_times: (iterable) the duration of the gap between each pulse at each level
    '''
    tot_ff_flat = 1
    tot_t_irr_flat = pulse_length
    for num_pulses, dwell_time in zip(nums_pulses, dwell_times):
        tot_t_irr_flat, ff = flatten_pulse_history(tot_t_irr_flat, num_pulses, dwell_time)
        tot_ff_flat *= ff
    return tot_t_irr_flat, tot_ff_flat

def compress_ph_levels(pulse_length, nums_pulses):
    '''
    Apply the compression algorithm to all levels of a multi-level pulsing history
    with a single-level schedule block.  
    
    :param pulse_lengths: active irradiation time from schedule block
    :param nums_pulses: (iterable) number of pulses at each level
    '''
    tot_t_irr_comp = pulse_length
    for num_pulses in nums_pulses:
        tot_t_irr_comp = compress_pulse_history(tot_t_irr_comp, num_pulses)
    return tot_t_irr_comp