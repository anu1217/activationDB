import argparse
import yaml
import numpy as np
from sympy import symbols, Eq, solve

def calc_time_params(active_burn_time, duty_cycle_list, num_pulses):
    '''
    Uses provided pulsing information to determine dwell time and total irradiation time.
    Assumes that the active irradiation time per pulse and dwell time between pulses both remain constant in any given simulation.
    Iterates over the number of pulses, and for each number, calculates dwell time.
    inputs:
        active_burn_time : total active irradiation time (float) in any chosen unit
        duty_cycle_list : list of chosen duty cycles (float)
        num_pulses : list of number of pulses (int) that the active irradiation period is divided into
    '''
    t_irr_arr = np.ndarray((len(num_pulses), len(duty_cycle_list)), dtype=float)
    dwell_time_arr = t_irr_arr.copy()
    dwell_time = symbols('dwell_time')
    pulse_length_list = []
    
    for num_idx, num in enumerate(num_pulses):
        pulse_length = active_burn_time / num
        pulse_length_list.append(pulse_length)
        for duty_cycle_idx, duty_cycle in enumerate(duty_cycle_list):
            dwell_time_eq = Eq(pulse_length / (pulse_length + dwell_time), duty_cycle)
            dwell_time_sol = solve((dwell_time_eq),(dwell_time))
            dwell_time_arr[num_idx, duty_cycle_idx] = dwell_time_sol[0]
            t_irr = pulse_length * num + dwell_time_sol[0] * (num - 1) # the dwell time is applied to the first N-1 pulses
            t_irr_arr[num_idx, duty_cycle_idx] = t_irr   
    return pulse_length_list, dwell_time_arr, t_irr_arr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_yaml', default = "iter_dt_out.yaml", help="Path (str) to YAML containing inputs")
    args = parser.parse_args()
    return args

def read_yaml(yaml_arg):
    '''
    input:
        yaml_arg : output of parse_args() corresponding to args.db_yaml
    '''
    with open(yaml_arg, 'r') as yaml_file:
        inputs = yaml.safe_load(yaml_file)
    return inputs

def main():        
    args = parse_args()
    inputs = read_yaml(args.db_yaml)  

    active_burn_time = inputs['active_burn_time']
    duty_cycle_list = inputs['duty_cycles']
    num_pulses = inputs['num_pulses']
    pulse_length_list, dwell_time_arr, t_irr_arr = calc_time_params(active_burn_time, duty_cycle_list, num_pulses)

if __name__ == "__main__":
    main()