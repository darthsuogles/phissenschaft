
DROP TABLE social_graph;
CREATE TABLE social_graph (followee VARCHAR, follower VARCHAR);
INSERT INTO social_graph VALUES ('Alice', 'Bob');
INSERT INTO social_graph VALUES ('Bob', 'Alice');
INSERT INTO social_graph VALUES ('Bob', 'Carol');
INSERT INTO social_graph VALUES ('Alice', 'Carol');
INSERT INTO social_graph VALUES ('Bob', 'Dylan');

WITH
rel1 AS (
SELECT followee AS a, follower AS b from social_graph
),
rel2 AS (
SELECT followee AS b, follower AS c from social_graph
)
SELECT a, COUNT(a) FROM (
SELECT a, c FROM (rel1 INNER JOIN rel2 on rel1.b = rel2.b)
EXCEPT
SELECT a, b AS c FROM rel1)
WHERE a <> c
GROUP BY a;

