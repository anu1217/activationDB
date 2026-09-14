import numpy as np

def make_training_features_outputs(partial_training_df, dict_df, child_nucs, parent_nucs):
    out_arr_list = []
    feature_arr_list = []
    for run_lbl in partial_training_df['run_lbl'].unique():
        parent_child_nuc_arr = np.zeros((len(child_nucs), len(parent_nucs)))
        reduced_df = partial_training_df.loc[partial_training_df['run_lbl'] == run_lbl]
        num_dens_idx = reduced_df.columns.get_loc("num_dens_(atoms/cm3)")
        for df_row in reduced_df.itertuples(index=False):
            child_nuc = df_row.nuclide
            parent_nuc = df_row.block_name
            parent_child_nuc_arr[child_nucs.index(child_nuc), parent_nucs.index(parent_nuc)] = df_row[num_dens_idx]
            t_irr = df_row.t_irr
            avg_flux_mag = df_row.avg_flux_mag
            flux_spec_shape = eval(dict_df[df_row.flux_spec_shape_id])
        #Each run_lbl is associated with a single combination of each of the features    
        feature_arr_list.append(np.array((t_irr, avg_flux_mag, *flux_spec_shape)))
        out_arr_list.append(parent_child_nuc_arr)
    return np.array(feature_arr_list), np.array(out_arr_list)
