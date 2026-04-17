import sqlite3
import uuid

def populate_table(cur):
    '''
    Create the sqlite table and populate it with data. The combination of pulse history, schedule data,
    flux file, and git hash used must be unique for each entry.
    '''
    cur.execute('''
    CREATE TABLE IF NOT EXISTS alara_simulations (
        id TEXT PRIMARY KEY,
        pulse_history TEXT,
        sched_history TEXT,
        flux_file TEXT,
        git_hash TEXT,
        UNIQUE(pulse_history, sched_history, flux_file, git_hash)
        )
    ''')

    cur.executemany("INSERT INTO alara_simulations (id, pulse_history, sched_history, flux_file, git_hash) VALUES (?, ?, ?, ?, ?)", 
                    [(str(uuid.uuid4()),'ph_1.json', 'sh_1.json', 'flux_file_1', 'git_hash_1'), 
                     (str(uuid.uuid4()),'ph_2.json', 'sh_2.json', 'flux_file_2', 'git_hash_2')])

def main():
    con = sqlite3.connect("alara_ids.sqlite")
    cur = con.cursor()
    populate_table(cur)
    con.commit()
    con.close()

if __name__ == "__main__":
    main()