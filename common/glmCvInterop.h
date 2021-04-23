#pragma once
#include <opencv2/core.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <assert.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

static glm::mat4 glmMat4FromCVMatrix(cv::Mat matin) {
  glm::mat4 out(1.0f);
  switch (CV_MAT_DEPTH(matin.type())) {
    case CV_64F:
      for (int y = 0; y < matin.rows; y++) { for (int x = 0; x < matin.cols; x++) { out[y][x] = (float)matin.at<double>(y, x); } }
      break;
    case CV_32F:
      for (int y = 0; y < matin.rows; y++) { for (int x = 0; x < matin.cols; x++) { out[y][x] = matin.at<float>(y, x); } }
      break;
    default:
      assert(false);
  }
  return out;
}

static cv::Matx44f CVMatrixFromGlmMat4(glm::mat4 matin) {
  cv::Matx44f out;
  for (int y = 0; y < 4; y++) { for (int x = 0; x < 4; x++) { out(y,x) = matin[y][x]; } }
  return out;
}

static glm::vec3 glmVec3FromCV(const cv::Vec3f& v) {
  return glm::make_vec3(v.val);
}

static glm::vec3 glmVec3FromCV(const cv::Mat& m) {
  glm::vec3 res;
  switch (CV_MAT_DEPTH(m.type())) {
    case CV_64F:
      for (size_t i = 0; i < 3; ++i) res[i] = m.ptr<double>(0)[i];
      break;
    case CV_32F:
      for (size_t i = 0; i < 3; ++i) res[i] = m.ptr<float>(0)[i];
      break;
  }
  return res;
}

static cv::Vec3f cvVec3FromGlm(const glm::vec3& v) {
  return cv::Vec3f(&v[0]);
}


#pragma clang diagnostic pop

