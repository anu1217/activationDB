import pandas as pd

def make_pdf_from_sql(conn, filter_str):
    '''
    Create a Pandas DataFrame using a SQLite connection object
    :param: conn (SQLite Connection object)
    :param: filter_str (string used to select rows from only certain simulations.
                        Set to "%" to access all simulations.)
    '''
    query = """SELECT nuclide, run_lbl, block_name, [num_dens_(atoms/cm3)], number_densities.flux_spec_shape_id, 
        avg_flux_mag, t_irr, flux_spec_shape, alara_simulations.output_file
        FROM number_densities
        JOIN flux_spectra
            ON number_densities.flux_spec_shape_id = flux_spectra.flux_spec_shape_id
        JOIN alara_simulations
            ON number_densities.run_lbl = alara_simulations.id   
        WHERE output_file LIKE ?    
        """
    training_df = pd.read_sql_query(query, conn, params=(filter_str,))
    conn.close()
    return training_df