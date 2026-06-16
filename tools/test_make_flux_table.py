import pytest
import make_flux_table as mft
import sqlite3
import uuid
import json
import numpy as np

@pytest.mark.parametrize(
    "cur,flux_data_dict",
    [
        (
            sqlite3.connect("activation_results.db").cursor(),
            {
                'flux_spec_shape_id' : [1,2],
                'flux_file' : ['../fnsf', '../iter_dt'],
                'flux_spectrum' : [json.dumps(np.array([3, 9, 12]).tolist()), json.dumps([5, 7, 11, 13, 25])]
            },
        ),
        (
            sqlite3.connect(":memory:").cursor().execute(
                """
                CREATE TABLE flux_spectra (
                flux_spec_shape_id INT PRIMARY KEY,
                flux_file TEXT,
                flux_spectrum TEXT,
                UNIQUE(flux_file, flux_spectrum)
                )
                """
            ),
            {
                'flux_spec_shape_id': [str(uuid.uuid4()), 5],
                'flux_file' : ['../fnsf', '../iter_dt'],
                'flux_spectrum' : [json.dumps(np.array([3, 9, 12]).tolist()), json.dumps([5, 7, 11, 13, 25])]
            },
        ),
    ],
)
def test_populate_flux_table(cur, flux_data_dict):
    """
    Ensure that the "INSERT into" statement was executed successfully,
    without committing the operation into the actual database.
    """
    cur.execute("BEGIN") # stops subsequent statements (e.g. CREATE TABLE) from being committed automatically
    mft.create_flux_table(cur)
    mft.populate_flux_table(cur, flux_data_dict)
    rows = cur.execute("SELECT * from flux_spectra").fetchall()
    assert len(rows) == len(flux_data_dict["flux_spec_shape_id"]) == len(flux_data_dict["flux_file"]) == len(flux_data_dict["flux_spectrum"])
    cur.connection.close()