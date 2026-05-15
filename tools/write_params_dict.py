import numpy as np
from string import Template

'''
The following data structure (child_dicts) is
an iterable of dictionaries, where each dictionary contains the details
of a schedule entry or a pulse entry. In the case of a schedule entry,
the value of the "children" key is a dictionary that follows the same format
as its parent.
[
{'type': 'pulse_entry',
    'pulse_length': (float),
    'pulse_length_unit': (str),
    'flux_name' : (str),
    'pulse_history': (iterable of (int, float, str)),
    'delay_dur' : (float),
    'delay_dur_unit': (str)
}
]
'''
#Take pulse lengths, dwell times (regardless of where they come from) and write to a dictionary?

# pe_dict = {
#     'type': 'pulse_entry',
#     'pulse_length': (float),
#     'pulse_length_unit': (str),
#     'flux_name' : (str),
#     'pulse_history': [((int, float, str))],
#     'delay_dur' : (float),
#     'delay_dur_unit': (str)}

pulse_lengths = np.asarray([1,2,3,4,5])
num_pulses = np.asarray([2,4,8,32,64])
abs_dwell_times = np.array([[5, 10, 15, 20, 25], [7, 10, 1, 9, 2], [5, 10, 15, 20, 25], [5, 10, 15, 20, 25]])
flux_name = 'frascati_ng'

pe_dicts = [{
    'type': 'pulse_entry',
    'pulse_length': pulse_lengths[c],
    'pulse_length_unit': 's',
    'flux_name' : flux_name,
    'pulse_history': [((num_pulses[c], value, 's'))],
    'delay_dur' : 0.0,
    'delay_dur_unit' : 's'
    }
    for r, row in enumerate(abs_dwell_times)
    for c, value in enumerate(row)]


# pass each dict in pe_dicts into function to make schedule/ph blocks
for pe_dict in pe_dicts:
    print([pe_dict])





