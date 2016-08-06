#ifndef POINT_DETECTOR_H
#define POINT_DETECTOR_H

#include <opencv2/opencv.hpp>

class PointDetector
{
public:
	PointDetector();
	~PointDetector();

	void init();
	void release();
	void match(const cv::Mat &query, const cv::Mat &train, std::vector<cv::DMatch> &vmatch) const;
	void describe(const cv::Mat &img, std::vector<cv::KeyPoint> &vkpt, cv::Mat &vdesc) const;
	void detect(const cv::Mat &img, std::vector<cv::KeyPoint> &vkpt) const;

private:
	cv::Ptr<cv::AKAZE> mDetector;
	cv::Ptr<cv::DescriptorMatcher> mMatcher;
};

#endif // POINT_DETECTOR_H