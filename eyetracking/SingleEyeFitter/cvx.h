#ifndef __SINGLEEYEFITTER_CVX_H__
#define __SINGLEEYEFITTER_CVX_H__

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace singleeyefitter {

const double SQRT_2 = std::sqrt(2.0);
const double PI = CV_PI;

namespace cvx
{
    inline cv::Mat& line(cv::Mat& dst, cv::Point2f from, cv::Point2f to, cv::Scalar color, int thickness=1, int linetype=cv::LINE_AA, int shift=8) {
        auto from_i = cv::Point(from.x * (1<<shift), from.y * (1<<shift));
        auto to_i = cv::Point(to.x * (1<<shift), to.y * (1<<shift));

        cv::line(dst, from_i, to_i, color, thickness, linetype, shift);
        return dst;
    }

    inline void cross(cv::Mat& img, cv::Point2f centre, double radius, const cv::Scalar& colour, int thickness = 1, int lineType = cv::LINE_AA, int shift = 8)
    {
        cvx::line(img, centre + cv::Point2f(-radius, -radius), centre + cv::Point2f(radius, radius), colour, thickness, lineType, shift);
        cvx::line(img, centre + cv::Point2f(-radius, radius), centre + cv::Point2f(radius, -radius), colour, thickness, lineType, shift);
    }
    inline void plus(cv::Mat& img, cv::Point2f centre, double radius, const cv::Scalar& colour, int thickness = 1, int lineType = cv::LINE_AA, int shift = 8)
    {
        cvx::line(img, centre + cv::Point2f(0, -radius), centre + cv::Point2f(0, radius), colour, thickness, lineType, shift);
        cvx::line(img, centre + cv::Point2f(-radius, 0), centre + cv::Point2f(radius, 0), colour, thickness, lineType, shift);
    }

    /*inline void cross(cv::Mat& img, cv::Point centre, int radius, const cv::Scalar& colour, int thickness = 1, int lineType = 8, int shift = 0)
    {
        cv::line(img, centre + cv::Point(-radius, -radius), centre + cv::Point(radius, radius), colour, thickness, lineType, shift);
        cv::line(img, centre + cv::Point(-radius, radius), centre + cv::Point(radius, -radius), colour, thickness, lineType, shift);
    }
    inline void plus(cv::Mat& img, cv::Point centre, int radius, const cv::Scalar& colour, int thickness = 1, int lineType = 8, int shift = 0)
    {
        cv::line(img, centre + cv::Point(0, -radius), centre + cv::Point(0, radius), colour, thickness, lineType, shift);
        cv::line(img, centre + cv::Point(-radius, 0), centre + cv::Point(radius, 0), colour, thickness, lineType, shift);
    }*/

    inline cv::Mat resize(const cv::Mat& src, cv::Size size, int interpolation=cv::INTER_LINEAR) {
        cv::Mat dst;
        cv::resize(src, dst, size, 0, 0, interpolation);
        return dst;
    }
    inline cv::Mat resize(const cv::Mat& src, double fx, double fy=0, int interpolation=cv::INTER_LINEAR) {
        if (fy == 0) fy = fx;
        cv::Mat dst;
        cv::resize(src, dst, cv::Size(), fx, fy, interpolation);
        return dst;
    }
}

} //namespace singleeyefitter

#endif // __SINGLEEYEFITTER_CVX_H__
