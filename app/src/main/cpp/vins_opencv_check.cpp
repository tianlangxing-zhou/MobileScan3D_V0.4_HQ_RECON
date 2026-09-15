#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

namespace gsslam {

void vins_opencv_check() {
    cv::Mat image(64, 64, CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(image, corners, 20, 0.01, 3.0);
    (void)corners;
}

}  // namespace gsslam
