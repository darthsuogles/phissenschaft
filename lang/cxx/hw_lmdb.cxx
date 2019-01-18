#include <iostream>
#include "db.h"
#include "lmdb_cxx.hpp"

int main() {
  auto kvbase = std::make_unique<db::LMDB>("src.lmdb", db::WRITE);
  auto trans = kvbase->NewTransaction();
  trans->Put("salut", "toutlemonde");
  trans->Commit();
  auto cursor = kvbase->NewCursor();
  while (cursor->Valid()) {
    std::cout << "\tkey: " << cursor->key()
	      << "\tvalue: " << cursor->value()
	      << std::endl;
    cursor->Next();
  }
}
