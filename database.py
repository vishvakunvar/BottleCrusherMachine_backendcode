import sqlite3 as sql
import datetime
import json

config = json.load(open('config.json','r'))

db = sql.connect('database.db')

def create_data_table():
    with db:
        db.execute('CREATE TABLE IF NOT EXISTS data (id TEXT PRIMARY KEY, timestamp TIMESTAMP NOT NULL, bottles INTEGER NOT NULL, cans INTEGER NOT NULL, polybags INTEGER NOT NULL, phone TEXT)')

def create_blob_table():
    with db:
        db.execute('CREATE TABLE IF NOT EXISTS blob (id TEXT PRIMARY KEY, date TIMESTAMP NOT NULL, image BLOB NOT NULL)')


def insert_data(bottles, cans, polybags, phone=None):
    mcid = config["machineInfo"]["mcid"]
    id = mcid + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    timestamp = datetime.datetime.now()
    with db:
        db.execute('INSERT INTO data VALUES (?,?,?,?,?,?)', (id, timestamp, bottles, cans, polybags, phone))

insert_data(1,2,3)

def get_data():
    with db:
        data = db.execute('SELECT * FROM data')
        return data

for rows in get_data().fetchall():
    print(rows)