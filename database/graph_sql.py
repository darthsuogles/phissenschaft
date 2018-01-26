import numpy as np
import numpy.random as rand
import sqlite3

NAME_LEN = 4

conn = sqlite3.connect('graph.db')
cursor = conn.cursor()
cursor.execute("DROP TABLE social_graph")
cursor.execute("""
CREATE TABLE social_graph 
(followee varchar({name_len}), follower varchar({name_len}))
""".format(name_len=NAME_LEN))

def gen_rand_name(name_len=NAME_LEN):
    return ''.join(map(
        chr, rand.randint(low=ord('a'), high=ord('z'), size=name_len)))

pop_size = 100
people = list(map(gen_rand_name, np.repeat(NAME_LEN, pop_size)))
rel_size = 2 * pop_size // 3
rel_follows = list(zip(people[:rel_size], people[-1:-rel_size:-1]))

for leader, follower in rel_follows:
    if leader == follower: continue
    cursor.execute("""
    INSERT INTO social_graph VALUES ('{leader}', '{follower}')
    """.format(leader=leader, follower=follower))

conn.commit()    
conn.close()
