import subprocess
import os
import argparse
import yaml

def run_inp_files(inp_file_folder, out_file_folder, alara_exec_path):
    inp_file_list = os.listdir(inp_file_folder)
    for inp_file in inp_file_list:
        if os.path.islink(inp_file_folder+"/"+inp_file) or inp_file.endswith(".dmp"):
            continue
        else:
            subprocess.run([alara_exec_path, "-o", out_file_folder+"/"+inp_file+"_out", inp_file_folder+"/"+inp_file], check=True) 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_case_yaml', default = "make_testing_inps.yaml", help="Path (str) to YAML containing inputs to construct testing data")
    args = parser.parse_args()
    return args

def read_yaml(yaml_arg):
    '''
    input:
        yaml_arg : output of parse_args() corresponding to args.testing_case_yaml
    '''
    with open(yaml_arg, 'r') as yaml_file:
        inputs = yaml.safe_load(yaml_file)
    return inputs


def main():
    args = parse_args()
    inputs = read_yaml(args.testing_case_yaml)

    inp_file_folder = inputs['inp_file_folder']
    out_file_folder = inputs['out_file_folder']
    alara_exec_path = inputs['alara_exec_path']

    run_inp_files(inp_file_folder, out_file_folder, alara_exec_path)

if __name__ == "__main__":
    main()