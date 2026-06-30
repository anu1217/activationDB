import pytest
import numpy as np
import sqlite3
import json
import make_training_inps as mti

@pytest.mark.parametrize("training_inp_info, sqlite_conn, min_on_times, time_unit, exp_filenames", [
    (
np.array([
        [
            [
            None,
            None
            ],

            [
            (2, 0.3, '../iter_flux'),
      
      (2, 0.3, '../frascati_flux')
            ]
        ],

        [
            [
            None,
            None
            ],

            [
            None,
            None
            ]
        ]
      ], dtype=object),
      sqlite3.connect(":memory:").cursor().execute(
                """
                CREATE TABLE flux_spectra (
                flux_spec_shape_id INTEGER PRIMARY KEY,
                flux_file TEXT,
                flux_spectrum TEXT,
                UNIQUE(flux_file, flux_spectrum)
                )
                """
            ).executemany("INSERT INTO flux_spectra (flux_file, flux_spectrum) VALUES (?, ?)",
            list(zip(*{
                'flux_file' : ['../fnsf', '../iter_dt'],
                'flux_spectrum' : ['[3, 9,12]', json.dumps([5,7, 11, 13, 25])]
            }.values()))).connection,
        [10, 20],
        's',
        np.array([[
        [
            [
            None,
            None
            ],

            [
            f"1_0.3_10_'s'_2",
      
      f"2_0.3_10_'s'_2"
            ]
        ],

        [
            [
            None,
            None
            ],

            [
            None,
            None
            ]
        ]
    ],
    [
        [
            [
            None,
            None
            ],

            [
            f"1_0.3_20_'s'_2",
      
      f"2_0.3_20_'s'_2"
            ]
        ],

        [
            [
            None,
            None
            ],

            [
            None,
            None
            ]
        ]
    ]
    ], dtype=object)
)])

def test_make_filename_strings(training_inp_info, sqlite_conn, min_on_times, time_unit, exp_filenames):
    obs_filenames = mti.make_filename_strings(training_inp_info, sqlite_conn, min_on_times, time_unit)
    assert obs_filenames == exp_filenames