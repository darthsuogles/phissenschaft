""" Duplicate email
""" 

import sqlite3 as db

conn = db.connect(':memory:')
curs = conn.cursor()

curs.execute('CREATE TABLE Person (Id int, Email varchar);')
curs.executemany('INSERT INTO Person VALUES (?, ?)', [
    (1, 'a@b.com'),
    (2, 'c@d.com'),
    (3, 'a@b.com')
])

# Check the result
# list(curs.execute('SELECT * FROM lc182;'))
sql_stmt = """
SELECT Email FROM (
  SELECT count(*) AS cnt, Email from Person GROUP BY Email
) AS _tmp WHERE cnt > 1
"""
list(curs.execute(sql_stmt))
