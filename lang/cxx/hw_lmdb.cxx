#include "db.h"
#include "lmdb_cxx.hpp"

int main() {
  auto kvbase = db::LMDB("src.lmdb", db::WRITE);
}
