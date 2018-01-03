""" Customers who never order
""" 

import sqlite3 as db

conn = db.connect(':memory:')
curs = conn.cursor()

curs.execute('CREATE TABLE Customers (Id int, Name varchar);')
curs.executemany('INSERT INTO Customers VALUES (?, ?)', [
    (1, 'Joe'),
    (2, 'Henry'),
    (3, 'Sam'),
    (4, 'Max')
])

curs.execute('CREATE TABLE Orders (Id int, CustomerId int);')
curs.executemany('INSERT INTO Orders VALUES (?, ?)', [
    (1, 3), 
    (2, 1)
])

# Check the result
# list(curs.execute('SELECT * FROM lc182;'))
sql_stmt = """
SELECT Name AS Customers FROM Customers LEFT OUTER JOIN (
  SELECT distinct(CustomerId) as cId from Orders
) AS _tmp ON cId = Id WHERE cId is NULL
"""
print(list(curs.execute(sql_stmt)))

