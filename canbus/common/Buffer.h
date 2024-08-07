#pragma once
#define NO_ASIO_DEPENDENCIES
#define NO_INTERPROCESS_DEPENDENCIES
#include <cassert>
#include <fstream>
#include <string>
#include <stdexcept>
#include <boost/endian/conversion.hpp>
#ifndef NO_ASIO_DEPENDENCIES
#include <boost/asio.hpp>
#endif
#ifndef NO_INTERPROCESS_DEPENDENCIES
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#endif

#ifdef _WIN32
#define snprintf _snprintf
#endif

class rewound_too_far : public std::exception {
public:
  rewound_too_far(size_t rewind_size, size_t offset) {
    snprintf(msg, 128, "Request to rewind by %zu bytes, when the current position is only %zu bytes.", rewind_size, offset);
  }
  virtual ~rewound_too_far() throw() {}
  virtual const char* what() const throw() { return msg; }
protected:
  char msg[128];
};

class end_of_buffer : public std::exception {
public:
  end_of_buffer(size_t needed, size_t available) {
    snprintf(msg, 128, "End of buffer reached during a request for %lu bytes (%lu available)", needed, available);
  }
  virtual ~end_of_buffer() throw() {}
  virtual const char* what() const throw() { return msg; }
protected:
  char msg[128];
};

// A serialization and deserialization buffer with shared backing and deserialization state
// Passing buffers by value will share the offset for deserialization, unless you explicitly
// call separate(). Kind of like a file descriptor in that respect.

class Buffer {
protected:
  struct payload_base {
    payload_base() : refcount(1), offset(0) {}
    virtual ~payload_base() {}
    virtual const char* data() const = 0;
    virtual char* data() = 0;
    virtual size_t size() = 0;
    virtual void append(const char* data, size_t size) = 0;
    virtual void clear() = 0;
    virtual void reserve(size_t new_size) = 0;
    virtual payload_base* clone() const = 0;
#ifndef NO_ASIO_DEPENDENCIES
    virtual boost::asio::const_buffer asio_buffer() = 0;
#endif
    virtual size_t convert_to_absolute_offset(size_t local_offset) { return local_offset; }

    size_t refcount, offset;
  };

  struct string_payload : public payload_base {
    string_payload() {}
    string_payload(const std::string& _str) : str(_str) {}
    string_payload(const char* data, size_t len) : str(data, len) {}
    template <typename StartIterator, typename EndIterator> string_payload(StartIterator begin, EndIterator end) : str(begin, end) {}
    ~string_payload() {}
    const char* data() const { return str.data(); }
    char* data() { return const_cast<char*>(str.data()); }
    size_t size() { return str.size(); }
    void append(const char* data, size_t size) { str.append(data, size); }
    void clear() { str.clear(); }
    void reserve(size_t new_size) { str.reserve(new_size); }
    payload_base* clone() const { return new string_payload(str); }
#ifndef NO_ASIO_DEPENDENCIES
    boost::asio::const_buffer asio_buffer() { return boost::asio::buffer(str); }
#endif
  protected:
    std::string str;
  };

  struct payload_window : public payload_base {
    payload_window(payload_base* _base, size_t _offset, size_t _length) : base(_base), window_offset(_offset), window_length(_length) {
      if ((window_offset + window_length) > base->size()) throw end_of_buffer(window_offset + window_length, base->size());
      ++(base->refcount);
    }
    ~payload_window() {
      if ((--(base->refcount)) == 0) delete base;
    }
    const char* data() const { return base->data() + window_offset; }
    char* data() { return base->data() + window_offset; }
    size_t size() { return window_length; }
    void append(const char* data, size_t size) { assert(0 && "append() on Buffer::payload_window is meaningless"); }
    void clear() { assert(0 && "clear() on Buffer::payload_window is meaningless"); }
    void reserve(size_t new_size) { assert(0 && "reserve() on Buffer::payload_window is meaningless"); }
    payload_base* clone() const { return new payload_window(base, window_offset, window_length); }
#ifndef NO_ASIO_DEPENDENCIES
    boost::asio::const_buffer asio_buffer() { return boost::asio::buffer(data(), size()); }
#endif
    size_t convert_to_absolute_offset(size_t local_offset) { return base->convert_to_absolute_offset(local_offset + window_offset); }
  protected:
    payload_base* base;
    size_t window_offset, window_length;
  };

#ifndef NO_INTERPROCESS_DEPENDENCIES
  struct immutable_file_payload : public payload_base {
    immutable_file_payload(const char* filename) : fm(filename, boost::interprocess::read_only), mr(fm, boost::interprocess::read_only) {}
    const char* data() const { return reinterpret_cast<const char*>(mr.get_address()); }
    char* data() { return reinterpret_cast<char*>(mr.get_address()); }
    size_t size() { return mr.get_size(); }
    void append(const char* data, size_t size) { assert(0 && "attempted write to Buffer::immutable_file_payload"); }
    void clear() { assert(0 && "attempted clear() of Buffer::immutable_file_payload"); }
    void reserve(size_t new_size) { assert(0 && "attempted reserve() on Buffer::immutable_file_payload"); }
    payload_base* clone() const { return new immutable_file_payload(fm.get_name()); }
#ifndef NO_ASIO_DEPENDENCIES
    boost::asio::const_buffer asio_buffer() { return boost::asio::buffer(data(), size()); }
#endif
  protected:
    boost::interprocess::file_mapping fm;
    boost::interprocess::mapped_region mr;
  };
#endif

  Buffer(payload_base* _payload) : payload(_payload)
#ifndef NO_ASIO_DEPENDENCIES
  , buffer_(payload->asio_buffer())
#endif
  { }

public:
  Buffer() : payload(new string_payload())
#ifndef NO_ASIO_DEPENDENCIES
  , buffer_(payload->asio_buffer())
#endif
  { }
  
  Buffer(const char* _data, size_t _len) : payload(new string_payload(_data, _len))
#ifndef NO_ASIO_DEPENDENCIES
  , buffer_(payload->asio_buffer())
#endif
  { }
  
  template <typename StartIterator, typename EndIterator> Buffer(StartIterator begin, EndIterator end) : payload(new string_payload(begin, end))
#ifndef NO_ASIO_DEPENDENCIES
  , buffer_(payload->asio_buffer())
#endif
{ }

  Buffer(const std::string& _str) : payload(new string_payload(_str))
#ifndef NO_ASIO_DEPENDENCIES
  , buffer_(payload->asio_buffer())
#endif
{ }

#ifndef NO_INTERPROCESS_DEPENDENCIES
  static Buffer withFile(const char* filename) { return Buffer(new immutable_file_payload(filename)); }
  static Buffer withFile(const std::string& filename) { return Buffer(new immutable_file_payload(filename.c_str())); }
#endif

  void writeFile(const char* filename) {
    std::filebuf fbuf;
    fbuf.open(filename, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    fbuf.sputn(payload->data(), payload->size());
    fbuf.close();
  }

  Buffer& operator=(const Buffer& right) {
    if (right.payload == payload) return *this; // self-assignment?
    release();
    assign(right);
    return *this;
  }

  Buffer(const Buffer& right) : payload(NULL) {
    assign(right);
  }

  ~Buffer() {
    release();
  }

  void clear() {
    if (payload->refcount == 1) { 
      payload->clear();
      payload->offset = 0;
    } else {
      --(payload->refcount);
      payload = new string_payload();
    }
  }

  void separate() {
    if (payload->refcount == 1) return;
    payload_base* old_payload = payload;
    payload = old_payload->clone();
    payload->offset = old_payload->offset;
    --(old_payload->offset);
  }

  Buffer& reserve(size_t expected_size) {
    payload->reserve(payload->size() + expected_size);
    return *this;
  }

  Buffer& put_u8(uint8_t i8) {
    payload->append((char*)&i8, 1);
    return *this;
  }

  Buffer& put_u16(uint16_t i16) {
    uint16_t res = boost::endian::native_to_big(i16);
    payload->append((char*)&res, 2);
    return *this;
  }

  Buffer& put_u32(uint32_t i32) {
    uint32_t res = boost::endian::native_to_big(i32);
    payload->append((char*)&res, 4);
    return *this;
  }

  Buffer& put_u64(uint64_t i64) {
    uint64_t res = boost::endian::native_to_big(i64);
    payload->append((char*)&res, 8);
    return *this;
  }
  Buffer& put_float(float f) {
    union {
      float f;
      uint32_t i;
    } a;
    a.f = f;
    uint32_t res = boost::endian::native_to_big(a.i);
    payload->append((char*)&res, 4);
    return *this;
  }

  Buffer& put_bytes(const char* b, size_t l) {
    payload->append(b, l);
    return *this;
  }

  Buffer& put_buffer(const Buffer& b) {
    payload->append(b.data(), b.size());
    return *this;
  }

  Buffer& put_u16_prefixed_string(const std::string& str) {
    put_u16(str.size());
    return put_bytes(str.data(), str.size());
  }

  Buffer& append(const Buffer& b) {
    payload->append(b.payload->data(), b.payload->size());
    return *this;
  }
  const char* data() const { return payload->data(); }
  size_t size() const { return payload->size(); }
// Deserialization methods:

  void rewind() {
    payload->offset = 0;
  }
  void rewind(size_t bytes) {
    if (payload->offset < bytes) throw rewound_too_far(bytes, payload->offset);
    payload->offset -= bytes;
  }
  void advance(size_t bytes) {
    consume(bytes);
  }
  size_t tell() const {
    return payload->offset;
  }
  // mosty for debugging
  size_t convert_to_absolute_offset(size_t local_offset) const {
    return payload->convert_to_absolute_offset(local_offset);
  }
  void seek_set(size_t target_offset) {
    payload->offset = target_offset;
    if (payload->offset > payload->size()) throw end_of_buffer(target_offset, payload->size());
  }
  size_t remaining() const {
    return (payload->size() - payload->offset);
  }
  bool eof() const {
    return (payload->size() == payload->offset);
  }
  uint8_t get_u8() {
    return *consume(1);
  }
  uint16_t get_u16() {
    return boost::endian::big_to_native(*reinterpret_cast<const uint16_t*>(consume(2)));
  }
  int16_t get_s16() {
    return boost::endian::big_to_native(*reinterpret_cast<const int16_t*>(consume(2)));
  }
  uint32_t get_u32() {
    return boost::endian::big_to_native(*reinterpret_cast<const uint32_t*>(consume(4)));
  }
  int32_t get_s32() {
    return boost::endian::big_to_native(*reinterpret_cast<const int32_t*>(consume(4)));
  }
  uint64_t get_u64() {
    return boost::endian::big_to_native(*reinterpret_cast<const uint64_t*>(consume(8)));
  }
  uint16_t get_u16_le() {
    return boost::endian::little_to_native(*reinterpret_cast<const uint16_t*>(consume(2)));
  }
  int16_t get_s16_le() {
    return boost::endian::little_to_native(*reinterpret_cast<const int16_t*>(consume(2)));
  }
  uint32_t get_u32_le() {
    return boost::endian::little_to_native(*reinterpret_cast<const uint32_t*>(consume(4)));
  }
  int32_t get_s32_le() {
    return boost::endian::little_to_native(*reinterpret_cast<const int32_t*>(consume(4)));
  }
  uint64_t get_u64_le() {
    return boost::endian::little_to_native(*reinterpret_cast<const uint64_t*>(consume(8)));
  }
  float get_float() {
    union { float f; uint32_t i; } a;
    a.i = boost::endian::big_to_native(*reinterpret_cast<const uint32_t*>(consume(4)));
    return a.f;
  }
  float get_float_le() {
    union { float f; uint32_t i; } a;
    a.i = boost::endian::little_to_native(*reinterpret_cast<const uint32_t*>(consume(4)));
    return a.f;
  }

  std::string get_u16_prefixed_string() {
    uint16_t len = get_u16();
    return std::string(consume(len), len);
  }

  std::string get_null_terminated_string() {
    const char* start = consume(1);

    const char* r = start;
    size_t len = 0;
    while (*r) {
      r = consume(1);
      ++len;
    }
    return std::string(start, len);
  }

  const char* peek(size_t bytes) {
    if (! bytes) return NULL;
    else if ((payload->size() - payload->offset) < bytes) throw end_of_buffer(bytes, payload->size() - payload->offset);
    return payload->data() + payload->offset;
  }

  const char* consume(size_t bytes) {
    if (! bytes) return NULL;
    else if ((payload->size() - payload->offset) < bytes) throw end_of_buffer(bytes, payload->size() - payload->offset);

    const char* r = payload->data() + payload->offset;
    payload->offset += bytes;
    return r;
  }

  Buffer consume_buffer(size_t bytes) {
    payload_window* window = new payload_window(payload, payload->offset, bytes);
    payload->offset += bytes;
    return Buffer(window);
  }

  operator std::string() const {
    return std::string(payload->data() + payload->offset, payload->size() - payload->offset);
  }

#ifndef NO_ASIO_DEPENDENCIES
// ConstBufferSequence requirements for boost::asio
  typedef boost::asio::const_buffer value_type;
  typedef const boost::asio::const_buffer* const_iterator;
  const boost::asio::const_buffer* begin() const { return &buffer_; }
  const boost::asio::const_buffer* end() const { return &buffer_ + 1; }
#endif

protected:
  void assign(const Buffer& right) {
    payload = right.payload;
    ++(payload->refcount);
#ifndef NO_ASIO_DEPENDENCIES
    buffer_ = payload->asio_buffer();
#endif
  }

  void release() {
    if ((--(payload->refcount)) == 0) {
      delete payload;
    }
    payload = NULL;
  }

  payload_base* payload;
#ifndef NO_ASIO_DEPENDENCIES
  boost::asio::const_buffer buffer_;
#endif
};

