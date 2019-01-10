#ifndef PHI9T_CORE_DB_H_
#define PHI9T_CORE_DB_H_

#include <mutex>
#include <string>

namespace db {

/**
 * The mode of the database, whether we are doing a read, write, or creating
 * a new database.
 */
enum Mode { READ, WRITE, NEW };

/**
 * An abstract class for the cursor of the database while reading.
 */
class Cursor {
 public:
  Cursor() {}
  virtual ~Cursor() {}
  /**
   * Seek to a specific key (or if the key does not exist, seek to the
   * immediate next). This is optional for dbs, and in default, SupportsSeek()
   * returns false meaning that the db cursor does not support it.
   */
  virtual void Seek(const std::string& key) = 0;
  virtual bool SupportsSeek() {
    return false;
  }
  /**
   * Seek to the first key in the database.
   */
  virtual void SeekToFirst() = 0;
  /**
   * Go to the next location in the database.
   */
  virtual void Next() = 0;
  /**
   * Returns the current key.
   */
  virtual std::string key() = 0;
  /**
   * Returns the current value.
   */
  virtual std::string value() = 0;
  /**
   * Returns whether the current location is valid - for example, if we have
   * reached the end of the database, return false.
   */
  virtual bool Valid() = 0;
};

/**
 * An abstract class for the current database transaction while writing.
 */
class Transaction {
 public:
  Transaction() { }
  virtual ~Transaction() { }
  /**
   * Puts the key value pair to the database.
   */
  virtual void Put(const std::string& key, const std::string& value) = 0;
  /**
   * Commits the current writes.
   */
  virtual void Commit() = 0;
};

/**
 * An abstract class for accessing a database of key-value pairs.
 */
class DB {
 public:
  DB(const std::string& /*source*/, Mode mode) : mode_(mode) {}
  virtual ~DB() { }
  /**
   * Closes the database.
   */
  virtual void Close() = 0;
  /**
   * Returns a cursor to read the database. The caller takes the ownership of
   * the pointer.
   */
  virtual std::unique_ptr<Cursor> NewCursor() = 0;
  /**
   * Returns a transaction to write data to the database. The caller takes the
   * ownership of the pointer.
   */
  virtual std::unique_ptr<Transaction> NewTransaction() = 0;

 protected:
  Mode mode_;
};

} // namespace db

#endif // PHI9T_CORE_DB_H_
