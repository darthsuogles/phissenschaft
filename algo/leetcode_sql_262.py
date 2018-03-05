"""
Trips and users
"""

import mysql.connector as mariadb

mariadb_connection = mariadb.connect(user='python_user', password='some_pass', database='employees')
cursor = mariadb_connection.cursor()


import sqlite3 as db
from collections import namedtuple

conn = db.connect(':memory:')
curs = conn.cursor()

curs.execute("DROP TABLE IF EXISTS Trips;")
curs.execute("""
CREATE TABLE Trips (
  Id INTEGER PRIMARY KEY,
  Client_Id INTEGER,
  Driver_Id INTEGER,
  City_Id INTEGER,
  Status VARCHAR(32),
  Request_at VARCHAR(16)
);
""")

Trips = namedtuple("Trips",
                   "id, client_id, driver_id, city_id, status, requested_at")

curs.executemany("INSERT INTO Trips VALUES (?, ?, ?, ?, ?, ?)", [
    Trips(1, 1, 10, 1, "completed", "2013-10-01"),
    Trips(2, 2, 11, 1, "cancelled_by_driver", "2013-10-01"),
    Trips(3, 3, 12, 6, "completed", "2013-10-01"),
    Trips(4, 4, 13, 6, "cancelled_by_client", "2013-10-01"),
    Trips(5, 1, 10, 1, "completed", "2013-10-02"),
    Trips(6, 2, 11, 6, "completed", "2013-10-02"),
    Trips(7, 3, 12, 6, "completed", "2013-10-02"),
    Trips(8, 2, 12, 12, "completed", "2013-10-03"),
    Trips(9, 3, 10, 12, "completed", "2013-10-03"),
    Trips(10, 4, 13, 12, "cancelled_by_driver", "2013-10-03"),
])

curs.execute("DROP TABLE IF EXISTS Users;")
curs.execute("""
CREATE TABLE Users (
  Users_Id INTEGER PRIMARY KEY,
  Banned BOOLEAN,
  Role VARCHAR(10)
);
""")

Users = namedtuple("Users", "users_id, banned, role")

curs.executemany("INSERT INTO Users VALUES (?, ?, ?)", [
    Users(1, False, "client"),
    Users(2, True, "client"),
    Users(3, False, "client"),
    Users(4, False, "client"),
    Users(10, False, "driver"),
    Users(11, False, "driver"),
    Users(12, False, "driver"),
    Users(13, False, "driver"),
])


def sql(stmt):
    return list(curs.execute(stmt))


sql("SELECT * FROM Trips;")
sql("SELECT * FROM Users;")

sql("""
SELECT Day,
       (sum(CASE WHEN Status <> 'completed' THEN 1 ELSE 0 END) / (1.0 * count(*))) AS Cancellation_Rate
FROM (
  (SELECT Users_Id FROM Users
   WHERE NOT Banned AND
         Role = 'client'
  ) UnbannedClients
  INNER JOIN
  (SELECT Client_Id, Status, Request_at AS Day FROM Trips
   WHERE Day <= '2013-10-03' and Day >= '2013-10-01'
  ) TripsInRange
  ON Users_Id = Client_Id
) UserTrips

GROUP BY Day
""")
