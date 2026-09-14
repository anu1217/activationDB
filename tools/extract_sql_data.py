import polars as pl
import numpy as np

def make_complete_df(conn, filter_str, ordered_nucs):
    '''
    Create a Pandas DataFrame using a SQLite connection object
    :param: conn (SQLite Connection object)
    :param: filter_str (string used to select rows from only certain simulations.
                        Set to "%" to access all simulations.)
    '''
    query = """
        SELECT number_densities.nuclide, number_densities.run_lbl, number_densities.block_name, 
        number_densities.[num_dens_(atoms/cm3)], number_densities.flux_spec_shape_id, number_densities.t_irr,
        number_densities.avg_flux_mag, flux_spectra.flux_spec_shape, alara_simulations.input_file
        FROM number_densities
        JOIN flux_spectra
            ON number_densities.flux_spec_shape_id = flux_spectra.flux_spec_shape_id
        JOIN alara_simulations
            ON number_densities.run_lbl = alara_simulations.id   
        WHERE alara_simulations.input_file LIKE ?    
        """
    complete_training_df = pl.read_database(query=query,
                                   connection=conn,
                                   execute_options={"parameters":(filter_str,)},
                                   schema_overrides={'nuclide' : pl.Enum(ordered_nucs),
                                                     'run_lbl' : pl.Categorical,
                                                     'block_name' : pl.Enum(
                                                        [ordered_nuc.replace("-", ":") for ordered_nuc in ordered_nucs]
                                                        ),
                                                    'flux_spec_shape_id' : pl.UInt8,
                                                    'input_file' : pl.Categorical
                                                    })
    return complete_training_df

def make_dicts_from_small_tables(conn):
    query = """
    SELECT flux_spectra.flux_spec_shape_id, flux_spectra.flux_spec_shape, alara_simulations.id, alara_simulations.input_file
    """
    dict_df = pl.read_database(query=query,
                               connection=conn,
                               schema_overrides = {'flux_spec_shape_id' : pl.UInt8,
                                                   'flux_spec_shape' : pl.Categorical,
                                                   'id' : pl.Categorical,
                                                   'input_file' : pl.Categorical
                                                   }
                                                   )
    dict_df = zip(dict_df['flux_spec_shape_id'], dict_df['flux_spec_shape'])
    return dict_df

def make_df_from_num_dens_table(conn, filter_str, ordered_nucs):
    query = """
    SELECT nuclide, run_lbl, block_name, [num_dens_(atoms/cm3)], flux_spec_shape_id, t_irr, avg_flux_mag
    FROM number_densities
    JOIN alara_simulations
        ON number_densities.run_lbl = alara_simulations.id
    WHERE alara_simulations.input_file LIKE ?    
    """
    partial_training_df = pl.read_database(query=query,
                                   connection=conn,
                                   execute_options={"parameters":(filter_str,)},
                                   schema_overrides={'nuclide' : pl.Enum(ordered_nucs),
                                                     'run_lbl' : pl.Categorical,
                                                     'block_name' : pl.Enum(
                                                        [ordered_nuc.replace("-", ":") for ordered_nuc in ordered_nucs]
                                                        ),
                                                    'num_dens_(atoms/cm3)' : pl.Float32,
                                                    'flux_spec_shape_id' : pl.UInt8,
                                                    't_irr' : pl.Float32,
                                                    'avg_flux_mag' : pl.Float32
                                                    }
                                                    )
    return partial_training_df