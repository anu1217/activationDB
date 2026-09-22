import numpy as np
import polars as pl


def make_training_features_outputs(partial_training_df, dict_df, child_nucs,
                                   parent_nucs):

    out_arr_list = []
    feature_arr_list = []

    for run_lbl in partial_training_df["run_lbl"].unique():

        parent_child_nuc_arr = np.zeros((len(child_nucs), len(parent_nucs)),
                                        dtype=np.float32)

        reduced_df = partial_training_df.filter(pl.col("run_lbl") == run_lbl)

        for df_row in reduced_df.iter_rows(named=True):

            child_nuc = df_row["nuclide"]
            parent_nuc = df_row["block_name"]

            parent_child_nuc_arr[
                child_nucs.index(child_nuc),
                parent_nucs.index(parent_nuc)] = df_row["num_dens_(atoms/cm3)"]

            t_irr = df_row["t_irr"]
            avg_flux_mag = df_row["avg_flux_mag"]

            flux_spec_shape = eval(dict_df[df_row["flux_spec_shape_id"]])

        # Each run_lbl is associated with a single combination of each of the features
        feature_arr_list.append(
            np.array((t_irr, avg_flux_mag, *flux_spec_shape),
                     dtype=np.float32))

        out_arr_list.append(parent_child_nuc_arr)
    return np.array(feature_arr_list), np.array(out_arr_list)


def make_training_features_outputs_per_parent(partial_training_df, dict_df,
                                              child_nucs, parent_nucs):

    all_out = []

    for parent in parent_nucs:

        all_features = []
        run_out = []

        parent_reduced_df = partial_training_df.filter(
            pl.col("block_name") == parent)

        for run_lbl in parent_reduced_df["run_lbl"].unique():

            parent_run_reduced_df = parent_reduced_df.filter(
                pl.col("run_lbl") == run_lbl)

            child_num_dens = np.zeros(len(child_nucs), dtype=np.float32)

            for df_row in parent_run_reduced_df.iter_rows(named=True):

                child_idx = child_nucs.index(df_row["nuclide"])

                child_num_dens[child_idx] = (df_row["num_dens_(atoms/cm3)"])

            flux_spec_shape = eval(dict_df[df_row["flux_spec_shape_id"]])

            run_features = np.array(
                (df_row["t_irr"], df_row["avg_flux_mag"], *flux_spec_shape),
                dtype=np.float32)

            all_features.append(run_features)
            run_out.append(child_num_dens)

        all_out.append(np.array(run_out, dtype=np.float32))

    return np.array(all_features, dtype=np.float32), np.array(all_out, dtype=np.float32)
