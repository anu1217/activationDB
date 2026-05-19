import numpy as np
import alara_output_processing as aop
import script_template as script_temp
import sqlite3
import pytest

@pytest.mark.parametrize( "all_flux_entries, num_groups, exp_flux_arr",
                          [(np.array([2,4,6,8,10,2,4,6,8,10]), 5, np.array([[2,4,6,8,10], [2,4,6,8,10]]))
                           ])

def test_parse_flux_str(all_flux_entries, num_groups, exp_flux_arr):
    obs_flux_arr = script_temp.parse_flux_str(all_flux_entries, num_groups)
    assert obs_flux_arr.all() == exp_flux_arr.all()

@pytest.mark.parametrize( "flux_arr, exp_norm_flux_arr",
                          [
                            (np.array([[2,4,6,8,10], [2,4,6,8,10]]), np.array([[num / 30 for num in [2,4,6,8,10]], 
                                                                               [num / 30 for num in [2,4,6,8,10]]
                                                                               ])
                                                                               )
                          ])

def test_normalize_flux(flux_arr, exp_norm_flux_arr):
    obs_norm_flux_arr = script_temp.normalize_flux(flux_arr)
    assert obs_norm_flux_arr.all() == exp_norm_flux_arr.all()

@pytest.mark.parametrize( "adf",
                          [
                            (aop.ALARADFrame(data=
        {"value": [5.678e-11]*2,
        "time": [-1]*2,
        "time_unit" : ["s"]*2,
        "variable": [0]*2,
        "var_unit": ["atoms/cm3"]*2,
        "block" : [1,2],
        "block_name": ["Be", "W"],
        "block_num": [2]*2,
        "nuclide": ["h-1"]*2,
        "half_life": ["-1"]*2,
        "run_lbl": ["test_case", "new_test_case"]}))
                          ])

def test_modify_adf_for_db(adf):
    '''
    Ensure that the expected columns exist in the adf.
    '''
    adf = script_temp.modify_adf_for_db(adf)
    assert not any(col in adf for col in [
        "value",
        "time",
        "time_unit",
        "variable",
        "var_unit",
        "block",
        "block_num",
    ])
    assert (all(col in adf
                for col in ["num_dens_(atoms/cm3)", "half_life", "block_name"]))

@pytest.mark.parametrize( "mod_adf",
                          [
                            (aop.ALARADFrame(data=
        {"num_dens_(atoms/cm3)": [5.678e-11]*2,
        "block_name": ["Be", "W"],
        "nuclide": ["h-1"]*2,
        "half_life": ["-1"]*2,
        "run_lbl": ["test_case", "new_test_case"]}))
                          ])
    
def test_write_to_sqlite(mod_adf):
    cursor = script_temp.write_to_sqlite(mod_adf, db_name = ":memory:")
    cursor.execute("SELECT * FROM number_densities")
    result = cursor.fetchall()
    assert len(result) == len(mod_adf["half_life"])
    assert len(result[0]) == 6

@pytest.mark.parametrize( "test_cursor",
                          [sqlite3.connect(":memory").cursor()])    

def test_close_sqlite_conn(test_cursor):
    # This function has a built-in test in the form of catching operational errors
    script_temp.close_sqlite_conn(test_cursor)