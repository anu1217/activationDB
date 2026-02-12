def flatten_pulse_history(pulse_length, num_pulses, dwell_time):
    t_irr = (num_pulses-1) * (pulse_length + dwell_time) + pulse_length
    flux_factor = num_pulses * pulse_length / t_irr

    return t_irr, flux_factor

