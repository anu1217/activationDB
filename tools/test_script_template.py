import alara_output_processing as aop
import script_template as script_temp

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
    return adf
    
def test_write_to_sqlite(adf, cursor):
    cursor.execute("SELECT * FROM number_densities")
    result = cursor.fetchall()
    assert len(result) == len(adf["half_life"])
    assert len(result[0]) == 6

def main():
    adf = aop.ALARADFrame(data=
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
        "run_lbl": ["test_case", "new_test_case"]})
    adf = test_modify_adf_for_db(adf)
    cursor = script_temp.write_to_sqlite(adf, db_name = ":memory:")
    test_write_to_sqlite(adf, cursor)
    script_temp.close_sqlite_conn(cursor) #this function has a built-in test in the form of catching any operational errors

if __name__ == "__main__":
    main()