import pandas as pd
import numpy as np

def make_pdf_from_sql(conn, filter_str):
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
    training_df = pd.read_sql_query(query, conn, params=(filter_str,), dtype={'flux_spec_shape_id' : np.int8,
                                                                              'avg_flux_mag' : np.float32,
                                                                              't_irr' : np.float32,
                                                                              'num_dens_(atoms/cm3)': np.float32})
    conn.close()
    return training_df