
LMDB_ROOT = $(shell brew --prefix lmdb)
LMDB_FLAGS = -I$(LMDB_ROOT)/include -L$(LMDB_ROOT)/lib -llmdb

all: hw_lmdb

hw_lmdb: hw_lmdb.cxx
	g++ -std=c++1z -O2 -g -fsanitize=address -o $@ $^ $(LMDB_FLAGS)
