#include "rhi/RHIQuery.h"
#include "rhi/gl/GLCommon.h"

class RHITimerQueryGL : public RHITimerQuery {
public:
  typedef boost::intrusive_ptr<RHITimerQueryGL> ptr;
  RHITimerQueryGL();
  virtual ~RHITimerQueryGL();

  GLuint glId() const { return m_glId; }
protected:
  GLuint m_glId;
};

class RHIOcclusionQueryGL : public RHIOcclusionQuery {
public:
  typedef boost::intrusive_ptr<RHIOcclusionQueryGL> ptr;
  RHIOcclusionQueryGL(RHIOcclusionQueryMode);
  virtual ~RHIOcclusionQueryGL();

  GLuint glId() const { return m_glId; }
  RHIOcclusionQueryMode mode() const { return m_mode; }

protected:
  GLuint m_glId;
  RHIOcclusionQueryMode m_mode;
};
