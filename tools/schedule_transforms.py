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

    t_irr = (num_pulses - 1) * (pulse_length + dwell_time) + pulse_length
    flux_factor = num_pulses * pulse_length / t_irr

    return t_irr, flux_factor


def flatten_all_ph_levels(pulse_lengths, nums_pulses, dwell_times):
    '''
    Apply the flattening algorithm to all levels of a multi-level pulsing history
    with a single-level schedule block.  
    
    :param pulse_lengths: (iterable) pulse durations at each level
    :param nums_pulses: (iterable) number of pulses at each level
    :param dwell_times: (iterable) the duration of the gap between each pulse at each level
    '''
    total_t_irr = 0
    total_flux_factor = 0
    for pulse_length_idx, pulse_length in enumerate(pulse_lengths):
        t_irr, flux_factor = flatten_pulse_history(
            pulse_length, nums_pulses[pulse_length_idx],
            dwell_times[pulse_length_idx])

        if pulse_length_idx == len(dwell_times) - 1:
            total_t_irr += t_irr
        else:
            total_t_irr += t_irr + dwell_times[pulse_length_idx]
            flux_factor = nums_pulses[pulse_length_idx] * pulse_length / (t_irr + dwell_times[pulse_length_idx])
        total_flux_factor += flux_factor
    return total_t_irr, total_flux_factor
