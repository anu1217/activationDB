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

