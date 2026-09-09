import sqlite3
import pandas as pd

def create_training_adf_from_sqlite():
    conn = sqlite3.connect("activation_results.db")
    query = "SELECT * FROM number_densities"
    training_df = pd.read_sql_query(query, conn)
    conn.close()
    return training_df