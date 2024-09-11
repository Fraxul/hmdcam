#pragma once
#include <fstream>
#include <string>
#include <stdexcept>
#include <cassert>
#include <boost/endian/conversion.hpp>
#ifdef FILE_SUPPORT
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

// A serialization and deserialization buffer with shared backing.

class SerializationBuffer {
protected:
  struct payload_base {
    payload_base() : refcount(1) {}
    virtual ~payload_base() {}
    virtual const char* data() const = 0;
    virtual char* data() = 0;
    virtual size_t size() = 0;
    virtual void append(const char* data, size_t size) = 0;
    virtual void clear() = 0;
    virtual void reserve(size_t new_size) = 0;
    virtual payload_base* clone() const = 0;
    virtual size_t convert_to_absolute_offset(size_t local_offset) { return local_offset; }

    size_t refcount;
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
    void append(const char* data, size_t size) { assert(0 && "append() on SerializationBuffer::payload_window is meaningless"); }
    void clear() { assert(0 && "clear() on SerializationBuffer::payload_window is meaningless"); }
    void reserve(size_t new_size) { assert(0 && "reserve() on SerializationBuffer::payload_window is meaningless"); }
    payload_base* clone() const { return new payload_window(base, window_offset, window_length); }
    size_t convert_to_absolute_offset(size_t local_offset) { return base->convert_to_absolute_offset(local_offset + window_offset); }
  protected:
    payload_base* base;
    size_t window_offset, window_length;
  };

#ifdef FILE_SUPPORT
  struct immutable_file_payload : public payload_base {
    immutable_file_payload(const char* filename) : fm(filename, boost::interprocess::read_only), mr(fm, boost::interprocess::read_only) {}
    const char* data() const { return reinterpret_cast<const char*>(mr.get_address()); }
    char* data() { return reinterpret_cast<char*>(mr.get_address()); }
    size_t size() { return mr.get_size(); }
    void append(const char* data, size_t size) { assert(0 && "attempted write to SerializationBuffer::immutable_file_payload"); }
    void clear() { assert(0 && "attempted clear() of SerializationBuffer::immutable_file_payload"); }
    void reserve(size_t new_size) { assert(0 && "attempted reserve() on SerializationBuffer::immutable_file_payload"); }
    payload_base* clone() const { return new immutable_file_payload(fm.get_name()); }
  protected:
    boost::interprocess::file_mapping fm;
    boost::interprocess::mapped_region mr;
  };
#endif

  SerializationBuffer(payload_base* _payload) : payload(_payload), offset(0)
  { }

public:
  SerializationBuffer() : payload(new string_payload()), offset(0)
  { }

  SerializationBuffer(const char* _data, size_t _len) : payload(new string_payload(_data, _len)), offset(0)
  { }

  template <typename StartIterator, typename EndIterator> SerializationBuffer(StartIterator begin, EndIterator end) : payload(new string_payload(begin, end)), offset(0)
{ }

  SerializationBuffer(const std::string& _str) : payload(new string_payload(_str)), offset(0)
{ }

#ifdef FILE_SUPPORT
  static SerializationBuffer withFile(const char* filename) { return SerializationBuffer(new immutable_file_payload(filename)); }
  static SerializationBuffer withFile(const std::string& filename) { return SerializationBuffer(new immutable_file_payload(filename.c_str())); }
#endif


  void writeFile(const char* filename) {
    std::filebuf fbuf;
    fbuf.open(filename, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    fbuf.sputn(payload->data(), payload->size());
    fbuf.close();
  }

  SerializationBuffer& operator=(const SerializationBuffer& right) {
    if (right.payload == payload) return *this; // self-assignment?
    release();
    assign(right);
    return *this;
  }

  SerializationBuffer(const SerializationBuffer& right) : payload(NULL) {
    assign(right);
  }

  ~SerializationBuffer() {
    release();
  }

  void clear() {
    if (payload->refcount == 1) {
      payload->clear();
    } else {
      --(payload->refcount);
      payload = new string_payload();
    }
    offset = 0;
  }

  void separate() {
    if (payload->refcount == 1) return;
    payload_base* old_payload = payload;
    payload = old_payload->clone();
    --(old_payload->refcount);
  }

  SerializationBuffer& reserve(size_t expected_size) {
    payload->reserve(payload->size() + expected_size);
    return *this;
  }

  SerializationBuffer& pad(size_t count) {
    char c = 0;
    for (size_t i = 0; i < count; ++i)
      payload->append(&c, 1);
    return *this;
  }

  SerializationBuffer& padToLength(size_t targetLength) {
    if (targetLength < payload->size()) {
      char buf[160];
      snprintf(buf, 128, "SerializationBuffer::padToLength: target length 0x%zx is smaller than current length (0x%zx)", targetLength, payload->size());
      throw std::runtime_error(buf);
    }
    pad(targetLength - payload->size());
    return *this;
  }

  SerializationBuffer& padToAlignment(size_t alignment) {
    size_t newLength = ((payload->size() + (alignment - 1)) / alignment) * alignment;
    if (newLength != payload->size())
      padToLength(newLength);
    return *this;
  }

  SerializationBuffer& put_u8(uint8_t i8) {
    payload->append((char*)&i8, 1);
    return *this;
  }

  SerializationBuffer& put_u16(uint16_t i16) {
    uint16_t res = boost::endian::native_to_big(i16);
    payload->append((char*)&res, 2);
    return *this;
  }

  SerializationBuffer& put_u32(uint32_t i32) {
    uint32_t res = boost::endian::native_to_big(i32);
    payload->append((char*)&res, 4);
    return *this;
  }

  SerializationBuffer& put_u64(uint64_t i64) {
    uint64_t res = boost::endian::native_to_big(i64);
    payload->append((char*)&res, 8);
    return *this;
  }

  SerializationBuffer& put_float(float f) {
    union {
      float f;
      uint32_t i;
    } a;
    a.f = f;
    uint32_t res = boost::endian::native_to_big(a.i);
    payload->append((char*)&res, 4);
    return *this;
  }

  SerializationBuffer& put_bytes(const char* b, size_t l) {
    payload->append(b, l);
    return *this;
  }

  SerializationBuffer& put_buffer(const SerializationBuffer& b) {
    payload->append(b.data(), b.size());
    return *this;
  }

  SerializationBuffer& put_u16_prefixed_string(const std::string& str) {
    put_u16(str.size());
    return put_bytes(str.data(), str.size());
  }

  SerializationBuffer& append(const SerializationBuffer& b) {
    payload->append(b.payload->data(), b.payload->size());
    return *this;
  }
  const char* data() const { return payload->data(); }
  size_t size() const { return payload->size(); }
  bool empty() const { return size() == 0; }
// Deserialization methods:

  void rewind() {
    offset = 0;
  }
  void rewind(size_t bytes) {
    if (offset < bytes) throw rewound_too_far(bytes, offset);
    offset -= bytes;
  }
  void advance(size_t bytes) {
    consume(bytes);
  }
  size_t tell() const {
    return offset;
  }
  // mosty for debugging
  size_t convert_to_absolute_offset(size_t local_offset) const {
    return payload->convert_to_absolute_offset(local_offset);
  }
  void seek_set(size_t target_offset) {
    offset = target_offset;
    if (offset > payload->size()) throw end_of_buffer(target_offset, payload->size());
  }
  size_t remaining() const {
    return (payload->size() - offset);
  }
  bool eof() const {
    return (payload->size() == offset);
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
  int64_t get_s64() {
    return boost::endian::big_to_native(*reinterpret_cast<const int64_t*>(consume(8)));
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
  int64_t get_s64_le() {
    return boost::endian::little_to_native(*reinterpret_cast<const int64_t*>(consume(8)));
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

  const char* peek(size_t bytes) {
    if (! bytes) return NULL;
    else if ((payload->size() - offset) < bytes) throw end_of_buffer(bytes, payload->size() - offset);
    return payload->data() + offset;
  }

  const char* consume(size_t bytes) {
    if (! bytes) return NULL;
    else if ((payload->size() - offset) < bytes) throw end_of_buffer(bytes, payload->size() - offset);

    const char* r = payload->data() + offset;
    offset += bytes;
    return r;
  }

  SerializationBuffer consume_buffer(size_t bytes) {
    if ((payload->size() - offset) < bytes) throw end_of_buffer(bytes, payload->size() - offset);

    payload_window* window = new payload_window(payload, offset, bytes);
    offset += bytes;
    return SerializationBuffer(window);
  }

  operator std::string() const {
    return std::string(payload->data() + offset, payload->size() - offset);
  }

protected:
  void assign(const SerializationBuffer& right) {
    payload = right.payload;
    ++(payload->refcount);
    offset = right.offset;
  }

  void release() {
    if ((--(payload->refcount)) == 0) {
      delete payload;
    }
    payload = NULL;
    offset = 0;
  }

  payload_base* payload;
  size_t offset;
};

