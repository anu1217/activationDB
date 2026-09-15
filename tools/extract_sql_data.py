import polars as pl
import numpy as np

def make_dict_from_small_table(conn):
    query = """
    SELECT flux_spectra.flux_spec_shape_id, flux_spectra.flux_spec_shape
    FROM flux_spectra
    """
    dict_df = pl.read_database(query=query,
                               connection=conn,
                               schema_overrides = {'flux_spec_shape_id' : pl.UInt8,
                                                   'flux_spec_shape' : pl.Categorical
                                                   }
                                )
    dict_df = dict(zip(dict_df['flux_spec_shape_id'], dict_df['flux_spec_shape']))
    return dict_df

def make_df_from_num_dens_table(conn, filter_str, batch_size, child_nucs, parent_nucs):
    query = """
    SELECT nuclide, run_lbl, block_name, [num_dens_(atoms/cm3)], flux_spec_shape_id, t_irr, avg_flux_mag
    FROM number_densities
    JOIN alara_simulations
        ON number_densities.run_lbl = alara_simulations.id
    WHERE alara_simulations.input_file LIKE ?    
    """
    partial_training_df_chunks = pl.read_database(query=query,
                                   connection=conn,
                                   iter_batches = True,
                                   batch_size = batch_size,
                                   execute_options={"parameters":(filter_str,)},
                                   schema_overrides={'nuclide' : pl.Enum(child_nucs),
                                                     'run_lbl' : pl.Categorical,
                                                     'block_name' : pl.Enum(parent_nucs),
                                                    'num_dens_(atoms/cm3)' : pl.Float32,
                                                    'flux_spec_shape_id' : pl.UInt8,
                                                    't_irr' : pl.Float32,
                                                    'avg_flux_mag' : pl.Float32
                                                    }
                                                    )
    return partial_training_df_chunks