import adf_to_sqlite
import create_adf
import sqlite3
import training_inp_params_to_dict as tiptd
import make_training_filenames as make_train_f
import make_testing_filenames as make_test_f
import argparse
import yaml
import numpy as np
import uuid
import sched_post_processor
import schedule_transforms
import alara_bookkeeping

def make_tirr(sch_tree, filename_idx_len, adf, flux_array, sqlite_conn, flux_norm):
    t_irr_flat = schedule_transforms.flatten_schedule(sch_tree)[0]
    adf = adf_to_sqlite.map_adf_flux_tirr(adf, flux_array, sqlite_conn, t_irr_flat, flux_norm)
    if filename_idx_len == 6:
        adf.rename(columns={'t_irr' : 't_irr_flat'}, inplace=True)
        t_irr_compressed = schedule_transforms.compress_schedule(sch_tree)
        adf = adf_to_sqlite.map_adf_flux_tirr(adf, flux_array, sqlite_conn, t_irr_compressed, flux_norm)
        adf.rename(columns={'t_irr' : 't_irr_compressed'}, inplace=True)
    return adf    

def save_out_to_db(max_fluence_factors, filename_array, inp_file_folder, out_file_folder, flux_path_modifier, sqlite_conn, git_hash, table_name):
    for idx, _ in np.ndenumerate(filename_array):
        if len(idx) == 4:
            min_on_time_idx, rel_on_time_factor_idx, flux_norm_factor_idx, flux_file_idx = idx
        elif len(idx) == 6:
            num_pulse_idx, dwell_time_idx, min_on_time_idx, rel_on_time_factor_idx, flux_norm_factor_idx, flux_file_idx = idx
        else:
            raise ValueError(
                f"Unexpected filename_array dimensions: {filename_array.ndim}"
            )       
        inp_filename = filename_array[idx]
        if inp_filename is None:
            # by construction, a filename_array entry is None only whenever a max_fluence_factors entry is None (and vice versa)
            continue
        else:
            output_path = out_file_folder + "/" + inp_filename + "_out"
            run_lbl = str(uuid.uuid4())

            _, flux_norm, flux_file = max_fluence_factors[rel_on_time_factor_idx, flux_norm_factor_idx, flux_file_idx]
            all_flux_entries = adf_to_sqlite.open_flux_file(flux_file+flux_path_modifier)
            # 286 = # of stable target nuclides
            num_groups = int(len(all_flux_entries) / 286)
            flux_array = adf_to_sqlite.parse_flux_str(all_flux_entries, num_groups)
            adf = create_adf.generate_single_adf(run_lbl, output_path)
            adf = adf_to_sqlite.modify_adf_for_db(adf)

            lines = sched_post_processor.read_out(output_path)
            pulse_dict = sched_post_processor.read_pulse_histories(lines)
            sch_tree = sched_post_processor.make_nested_dict(lines)
            sch_tree = sched_post_processor.add_ph_to_sch_tree(sch_tree, pulse_dict)['top_schedule']['children']
            
            #t_irr = schedule_transforms.flatten_schedule(sch_tree)[0]
            # need to account for dwell times
            #adf = adf_to_sqlite.map_adf_flux_tirr(adf, flux_array, sqlite_conn, t_irr, flux_norm)

            adf = make_tirr(sch_tree, len(idx), adf, flux_array, sqlite_conn, flux_norm)
            conn_cursor = adf_to_sqlite.write_to_sqlite(adf, table_name, sqlite_conn)

            alara_bookkeeping.create_sqlite_table(conn_cursor)
            data_dict = {"id" : [run_lbl],
                         "input_file" : [inp_file_folder + "/" + inp_filename],
                         "output_file" : [output_path],
                         "flux_file" : [flux_file],
                         "git_hash" : [git_hash]
                         }
            alara_bookkeeping.populate_table(conn_cursor, data_dict)
            conn_cursor.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_yaml', '-t', help="Path (str) to YAML containing inputs to construct training or testing data")
    args = parser.parse_args()
    return args

def read_yaml(yaml_arg):
    '''
    input:
        yaml_arg : output of parse_args() corresponding to args.date_yaml
    '''
    with open(yaml_arg, 'r') as yaml_file:
        inputs = yaml.safe_load(yaml_file)
    return inputs

def main():
    args = parse_args()
    inputs = read_yaml(args.training_case_yaml)

    min_on_times = inputs['min_on_times']
    time_unit = inputs['min_on_time_unit']
    rel_on_time_factors = inputs['rel_on_time_factors']
    flux_norm_factors = inputs['flux_norm_factors']
    flux_files = inputs['flux_files']
    sqlite_conn_db_name = inputs['sqlite_db_name']
    flux_path_modifier = inputs['flux_path_modifier']
    trunc_tolerance = inputs['trunc_tolerance']
    inp_file_folder = inputs['inp_file_folder']
    out_file_folder = inputs['out_file_folder']
    git_hash = inputs['git_hash']
    table_name = inputs['table_name']

    max_fluence_factors = tiptd.make_flux_tirr_combos(rel_on_time_factors, flux_norm_factors, flux_files)
    sqlite_conn = sqlite3.connect(sqlite_conn_db_name)
    filename_array = make_train_f.make_filename_strings(max_fluence_factors, sqlite_conn, min_on_times, time_unit, trunc_tolerance)
    save_out_to_db(max_fluence_factors, filename_array, inp_file_folder, out_file_folder, flux_path_modifier, sqlite_conn, git_hash, table_name)
    sqlite_conn.close()


if __name__ == "__main__":
    main()
