#pragma once
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>
#include <boost/noncopyable.hpp>

class RHIObject : public boost::intrusive_ref_counter<RHIObject>, public boost::noncopyable {
public:
  typedef boost::intrusive_ptr<RHIObject> ptr;
  virtual ~RHIObject();
};

