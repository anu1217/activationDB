from alara_output_processing import FileParser, ALARADFrame, DataLibrary

filepath = '/filespace/a/asrajendra/research/ALARA_fork/ALARA/examples/compare_tirr/1y_tirr/1y_out.txt'

alara_data = FileParser(filepath)
output_tables = alara_data.extract_tables() 

runs = {
    'run1' : filepath,
}

dfs = DataLibrary.make_entries(runs)
print((dfs.keys()))

totals = ALARADFrame.extract_totals()

# total t_irr from top of output file for now
# avg flux magnitude from flux file

