import alara_output_processing as aop
import numpy as np
import openmc
import argparse
import yaml
import sqlite3

def open_files():
    flux_lines = open('/filespace/a/asrajendra/research/activationDB/ref_flux_files/iter_dt_flux', 'r').readlines()
    return flux_lines

def filter_output(inputs):
    #runs_list = [inputs['runs_100'], inputs['runs_90'], inputs['runs_50'], inputs['runs_25']]
    #start with one dictionary of runs first
    runs_list = [inputs['runs_100']]
    total_values = []
    for runs_idx, runs in enumerate(runs_list):
        lib = aop.DataLibrary()
        adf = aop.DataLibrary.make_entries(lib, runs)
        #Be = Zone 1, W = Zone 2
        for run_idx, run in enumerate(runs):
            filtered_adf = adf.filter_rows(
                filter_dict={
                    "variable" : adf.VARIABLE_ENUM["Number Density"],
                    "nuclide"  : "total",
                    "block" : adf.BLOCK_ENUM["Zone"],
                    "run_lbl" : f"run{run_idx+1}_{inputs['duty_cycles'][runs_idx]}"
                }
            )   
            run_total_values = []
            for value in filtered_adf['value']:
                run_total_values.append(value) #make a separate list for all total_values in a run
            total_values.append(run_total_values)
    return total_values, len(filtered_adf['block']), adf

def find_child_nuclides(inputs, adf, num_blocks):
    #start with one dictionary of runs first
    runs_list = [inputs['runs_100']]
    for runs_idx, runs in enumerate(runs_list):
        dict_nuclides = [] #list of nuclide number densities for entire dictionary
        for run_idx, run in enumerate(runs):
            run_nuclides = []
            filtered_adf = adf.filter_rows(
                filter_dict={
                    "variable" : adf.VARIABLE_ENUM["Number Density"],
                    "block" : adf.BLOCK_ENUM["Zone"],
                    "run_lbl" : f"run{run_idx+1}_{inputs['duty_cycles'][runs_idx]}"
                }
            )    
            for block_num in range(num_blocks):
                
                block_filtered_adf = filtered_adf.filter_rows(
                    filter_dict = {
                        "block_num" : f"{block_num+1}"
                    }
                )
                individual_block_nuclides = []
                for nuclide in block_filtered_adf['nuclide']:
                    individual_block_nuclides.append(nuclide)
                run_nuclides.append(individual_block_nuclides)        
            dict_nuclides.append(run_nuclides)

def store_flux_lines(flux_lines):
    energy_bins = openmc.mgxs.GROUP_STRUCTURES['VITAMIN-J-175'] 
    bin_widths = []
    for bin_index in range(len(energy_bins) - 1):
        bin_width = energy_bins[bin_index + 1] - energy_bins[bin_index]
        bin_widths.append(bin_width)

    all_entries = []
    for flux_line in flux_lines:
        if flux_line.strip(): #if the current line is not blank
            all_entries.extend(flux_line.split())
    all_entries = np.array(all_entries, dtype=float)
    return bin_widths, all_entries

def normalize_flux_spectrum(all_entries, bin_widths, num_blocks):
    flux_array = all_entries.reshape(num_blocks, len(bin_widths))
    total_flux = np.sum(all_entries)

    for zone_idx in range(num_blocks):
        flux_array[zone_idx,:] = (flux_array[zone_idx,:] / bin_widths) * (1 / total_flux)  
    return flux_array

# Will pull t_irr and avg flux magnitude once full schedule history is available in output files

# Need to actually insert data into the database... 
def write_sqlite():
    database = 'activation_results.db'
    create_table = 'CREATE TABLE contacts ( \
	t_irr REAL PRIMARY KEY, \
	parent_nuc REAL NOT NULL, \
	child_nuc REAL NOT NULL, \
	avg_flux_mag REAL NOT NULL UNIQUE, \
	norm_flux_spectrum REAL NOT NULL UNIQUE \
    );'

    try:
        with sqlite3.connect(database) as conn:
            cursor = conn.cursor()
            cursor.execute(create_table)   
            conn.commit()

            result = cursor.fetchall()

            cursor.close()

        if conn:
            conn.close()
    except sqlite3.OperationalError as e:
        print(e)        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_yaml', default = "/filespace/a/asrajendra/research/activationDB/write_sqlite_dir/tools/iter_dt_out.yaml", help="Path (str) to YAML containing inputs")
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

    flux_lines = open_files()
    total_values, num_blocks, adf = filter_output(inputs)
    find_child_nuclides(inputs, adf, num_blocks)
    bin_widths, all_entries = store_flux_lines(flux_lines)
    flux_array = normalize_flux_spectrum(all_entries, bin_widths, num_blocks)
    write_sqlite()

if __name__ == "__main__":
    main()
