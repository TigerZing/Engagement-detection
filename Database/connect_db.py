import sqlite3 as lite
import os
import os.path as osp
import sys
from sqlite3 import Error

# Create connection to database
def create_connection(path):
    """ create a database connection to the SQLite database specified by path
    :param path: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = lite.connect(path)
    except Error as e:
        print(e)
    return conn


if __name__=="__main__":
    
    # DATABASE path
    DATABASE_path = "/home/hont/LA/DATABASE"
    conn = None
    # load configurations
    config_path = osp.join(DATABASE_path, "LA.db")
    # connect db
    conn = create_connection(config_path)
    with conn:
        cur = conn.cursor()
        cur.execute('SELECT SQLITE_VERSION()')
        intial_tables(cur)

        