#ifndef SingleEyeFitter_h__
#define SingleEyeFitter_h__

#include <mutex>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <SingleEyeFitter/cvx.h>
#include <SingleEyeFitter/Circle.h>
#include <SingleEyeFitter/Ellipse.h>
#include <SingleEyeFitter/Sphere.h>

namespace singleeyefitter {


    template<typename Scalar>
    inline Eigen::Matrix<Scalar, 2, 1> toEigen(const cv::Point2f& point) {
        return Eigen::Matrix<Scalar, 2, 1>(static_cast<Scalar>(point.x),
            static_cast<Scalar>(point.y));
    }
    template<typename Scalar>
    inline cv::Point2f toPoint2f(const Eigen::Matrix<Scalar, 2, 1>& point) {
        return cv::Point2f(static_cast<float>(point[0]),
            static_cast<float>(point[1]));
    }
    template<typename Scalar>
    inline cv::Point toPoint(const Eigen::Matrix<Scalar, 2, 1>& point) {
        return cv::Point(static_cast<int>(point[0]),
            static_cast<int>(point[1]));
    }
    template<typename Scalar>
    inline cv::RotatedRect toRotatedRect(const Ellipse2D<Scalar>& ellipse) {
        return cv::RotatedRect(toPoint2f(ellipse.centre),
            cv::Size2f(static_cast<float>(2 * ellipse.major_radius),
            static_cast<float>(2 * ellipse.minor_radius)),
            static_cast<float>(ellipse.angle * 180 / PI));
    }
    template<typename Scalar>
    inline Ellipse2D<Scalar> toEllipse(const cv::RotatedRect& rect) {
        return Ellipse2D<Scalar>(toEigen<Scalar>(rect.center),
            static_cast<Scalar>(rect.size.width / 2),
            static_cast<Scalar>(rect.size.height / 2),
            static_cast<Scalar>(rect.angle*PI / 180));
    }
    template<typename Scalar>
    inline Ellipse2D<Scalar> toEllipseWithOffset(const cv::RotatedRect& rect, Scalar offsetX, Scalar offsetY) {
        return Ellipse2D<Scalar>(
            Eigen::Matrix<Scalar, 2, 1>(
                static_cast<Scalar>(rect.center.x) - offsetX,
                static_cast<Scalar>(rect.center.y) - offsetY),
            static_cast<Scalar>(rect.size.width / 2),
            static_cast<Scalar>(rect.size.height / 2),
            static_cast<Scalar>(rect.angle*PI / 180));
    }

    class EyeModelFitter {
    public:
        // Typedefs
        typedef Eigen::Matrix<double, 2, 1> Vector2;
        typedef Eigen::Matrix<double, 3, 1> Vector3;
        typedef Eigen::ParametrizedLine<double, 2> Line;
        typedef Eigen::ParametrizedLine<double, 3> Line3;
        typedef singleeyefitter::Circle3D<double> Circle;
        typedef singleeyefitter::Ellipse2D<double> Ellipse;
        typedef singleeyefitter::Sphere<double> Sphere;
        typedef size_t Index;

        static const Vector3 camera_centre;

        // Public fields
        double focal_length = 0;

#if USE_SPII
        // Contrast fitting parameters
        double region_band_width = 5;
        double region_step_epsilon = 0.5;
        double region_scale = 1;
#endif

        // Constructors
        EyeModelFitter();

        Index add_observation(cv::Mat image, Ellipse pupil, int n_pseudo_inliers = 0);
        Index add_observation(cv::Mat image, Ellipse pupil, std::vector<cv::Point2f> pupil_inliers);

        void reset();

        //
        // Global (eye+pupils) calculations
        //

        bool unproject_observations(double pupil_radius /*= 1*/, double eye_z /*= 20*/, bool use_ransac = true);

        void initialise_model();


        typedef std::function<void(const Sphere&, const std::vector<Circle>&)> CallbackFunction;
        void refine_with_inliers(const CallbackFunction& callback = CallbackFunction());
#ifdef USE_SPII
        void refine_with_region_contrast(const CallbackFunction& callback = CallbackFunction());
#endif


        struct Observation {
            cv::Mat image;
            Ellipse ellipse;
            std::vector<cv::Point2f> inliers;

            Observation();
            Observation(cv::Mat image, Ellipse ellipse, std::vector<cv::Point2f> inliers);
        };
        struct PupilParams {
            double theta, psi, radius;
            PupilParams();
            PupilParams(double theta, double psi, double radius);
        };
        struct Pupil {
            Observation observation;
            Circle circle;
            PupilParams params;
            bool init_valid;

            Pupil();
            Pupil(Observation observation);
        };

        //
        // Local (single pupil) calculations
        //
        const Circle& unproject_single_observation(Index id, double pupil_radius);
        const Circle& initialise_single_observation(Index id);
#ifdef USE_SPII
        const Circle& refine_single_with_contrast(Index id);
        double single_contrast_metric(Index id) const;
        void print_single_contrast_metric(Index id) const;
#endif

        Sphere eye;
        std::vector<Pupil> pupils;
        std::mutex model_mutex;
        // Model version gets incremented on initialisation/reset, so that long-running background-thread refines don't overwrite the model
        int model_version = 0;

        bool hasEyeModel() const { return (!(eye == Sphere::Null)); }

        bool unproject_single_observation(Circle& outCircle, const Ellipse& pupil, double pupil_radius) const;
        const Circle& unproject_single_observation(Pupil& pupil, double pupil_radius) const;
        const Circle& initialise_single_observation(Pupil& pupil);
#ifdef USE_SPII
        const Circle& refine_single_with_contrast(Pupil& pupil);
        double single_contrast_metric(const Pupil& pupil) const;
        void print_single_contrast_metric(const Pupil& pupil) const;
#endif

        Circle circleFromParams(const PupilParams& params) const;
        static Circle circleFromParams(const Sphere& eye, const PupilParams& params);
    };

}

#endif // SingleEyeFitter_h__
