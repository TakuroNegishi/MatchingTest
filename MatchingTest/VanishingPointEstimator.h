#ifndef VANISHING_POINT_ESTIMATOR_H
#define VANISHING_POINT_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include "MovingAverageFilter.h"
#include "WeightedAverageFilter.h"

class VanishingPointEstimator
{
private:
	std::vector<cv::Point2f> pointHistory;
	std::vector<cv::Point2f> pointHistoryMA;
	std::vector<cv::Point2f> pointHistoryWA;
	MovingAverageFilter* maFilterX;
	MovingAverageFilter* maFilterY;
	WeightedAverageFilter* waFilterX;
	WeightedAverageFilter* waFilterY;
public:
	static const cv::Scalar SCALAR_CYAN;
	static const cv::Scalar SCALAR_RED;
	static const cv::Scalar SCALAR_GREEN;
	static const cv::Scalar SCALAR_BLACK;
	static const cv::Scalar SCALAR_WHITE;
	static const int ERROR_VP;
	static const int VPOINT_SIZE;
	static const std::string RESULT_PATH;
	static const int IMG_WIDTH;
	static const double W_MAX;
	static const int GRAPH_WIDTH;
	static const int GRAPH_HEIGHT;
	static const double INF;

	VanishingPointEstimator();
	~VanishingPointEstimator();
	void clear();
	void estimate(const std::vector<cv::DMatch>& matchVector, const std::vector<cv::KeyPoint>& currentKpts, const std::vector<cv::KeyPoint>& prevKpts);
	cv::Point2f VanishingPointEstimator::getCrossPoint(const std::vector<cv::DMatch>& matchVector, const std::vector<cv::KeyPoint>& currentKpts, const std::vector<cv::KeyPoint>& prevKpts);
	cv::Point2f VanishingPointEstimator::getCrossPointWeighted(const cv::Point2f& crossP, const std::vector<cv::DMatch>& matchVector, const std::vector<cv::KeyPoint>& currentKpts, const std::vector<cv::KeyPoint>& prevKpts);
	void VanishingPointEstimator::normalize(cv::Point2f &vec);
	void drawLastVP(cv::Mat &outImg);
	cv::Mat getVanishPointHistory();
	void drawDashedLine(cv::Mat& out, const cv::Point& p1, const cv::Point& p2, const cv::Scalar& color, const int lineWidth, const int interval);
	void drawVPLine(cv::Mat& out, const std::vector<cv::Point2f>& vpHistory, const cv::Scalar color, const int i,
		const float widthScale, const int graphWidthHalf, const float heightSpan);
	void logVanishPointHistoryAll(const std::string& fileName);
	void logVanishPointHistory(const std::string& fileName, const std::vector<cv::Point2f>& vpHistory);
	void readVanishPointHistoryAll(const std::string& filePath);
	void readVanishPointHistory(const std::string& filePath, std::vector<cv::Point2f>& vpHistory);
	//double getDTWDistance(const std::vector<int>& v, const std::vector<int>& w);
	void addVPHistory(cv::Point2f vp);
};

#endif // VANISHING_POINT_ESTIMATOR_H
