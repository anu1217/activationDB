import argparse
import yaml
import numpy as np
import example_case_reader as ecr

'''
Runs and tests the methods in script_template.py using a series of run dictionaries
with various pulse numbers and duty cycles.
'''
def test_write_to_adf(adf, run_dicts):
    '''
    Ensure that entries from all cases have been written into the adf.
    '''
    assert len(adf['run_lbl'].unique()) == sum([len(run_lbl) for _, run_lbl in run_dicts.items()])

def test_assign_adf_tirr_arr_mod(adf):
    '''
    Ensure that two of the run labels are associated with the correct irradiation time.
    '''
    assert not ((adf["run_lbl"] == "iter_dt_2p_100_4y")
                & (adf["t_irr"] != 4)).any()
    assert not ((adf["run_lbl"] == "iter_dt_64p_25_4y")
                & (adf["t_irr"] != 15.8125)).any()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_yaml",
        default="iter_dt_out.yaml",
        help="Path (str) to YAML containing inputs",
    )
    args = parser.parse_args()
    return args


def read_yaml(yaml_arg):
    """
    input:
        yaml_arg : output of parse_args() corresponding to args.db_yaml
    """
    with open(yaml_arg, "r") as yaml_file:
        inputs = yaml.safe_load(yaml_file)
    return inputs


def main():
    args = parse_args()
    inputs = read_yaml(args.db_yaml)

    active_burn_time = np.asarray(inputs["active_burn_time"])
    duty_cycle_list = np.asarray(inputs["duty_cycles"])
    num_pulses = np.asarray(inputs["num_pulses"])
    t_irr_arr = ecr.calc_time_params(active_burn_time, duty_cycle_list,
                                             num_pulses)

    run_dicts = inputs["run_dicts"]
    adf = ecr.write_to_adf(run_dicts)
    test_write_to_adf(adf, run_dicts)

    num_pulses = inputs["num_pulses"]
    duty_cycles = inputs["duty_cycles"]

    adf = ecr.assign_adf_tirr_arr_mod(adf, t_irr_arr, num_pulses,
                            duty_cycles)
    test_assign_adf_tirr_arr_mod(adf)


if __name__ == "__main__":
    main()