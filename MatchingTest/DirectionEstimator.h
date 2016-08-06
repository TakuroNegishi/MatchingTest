#ifndef DIRECTION_ESTIMATOR_H
#define DIRECTION_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include "PointDetector.h"
#include "MovingAverageFilter.h"
#include "WeightedAverageFilter.h"

class DirectionEstimator
{
public:
	DirectionEstimator();
	~DirectionEstimator();
	void clear();
	void estimate(const cv::Mat &rgbaImg);
	void calcMatchingFlow();
	cv::Point2f getCrossPoint();
	float getDistance(const cv::Point2f &pt1, const cv::Point2f &pt2);
	void draw(const cv::Mat &rgbaImg);
	void drawVanishPointHistory();
	void drawVPLine(cv::Mat& out, const std::vector<cv::Point2f>& vpHistory, const cv::Scalar color, const int i,
		const float widthScale, const int graphWidthHalf, const float heightSpan);
	void logVanishPointHistoryAll(const std::string& fileName);
	void logVanishPointHistory(const std::string& fileName, const std::vector<cv::Point2f>& vpHistory);
	void readVanishPointHistoryAll(const std::string& filePath);
	void readVanishPointHistory(const std::string& filePath, std::vector<cv::Point2f>& vpHistory);
	void setIsSaveImg(bool flag);

private:
	static const int POINT_SIZE;			// 特徴点の描画半径
	static const cv::Scalar SCALAR_RED;
	static const cv::Scalar SCALAR_GREEN;
	static const cv::Scalar SCALAR_BLUE;
	static const cv::Scalar SCALAR_YELLOW;
	static const cv::Scalar SCALAR_PURPLE;
	static const cv::Scalar SCALAR_CYAN;
	static const cv::Scalar SCALAR_BLACK;
	static const cv::Scalar SCALAR_WHITE;
	static const int FLOW_LINE_MIN_LIMIT;	// 許容する特徴点の最小距離距離
	static const int FLOW_LINE_MAX_LIMIT;	// 許容する特徴点の最大距離距離
	static const int FRAME_SPAN;
	static const std::string RESULT_PATH;
	static const int ERROR_VP;
	static const int IMG_WIDTH;
	static const int IMG_HEIGHT;

	bool isFirstFrame;
	bool isSaveImg;
	int count;
	PointDetector detector;
	MovingAverageFilter* maFilter;
	WeightedAverageFilter* waFilter;
	cv::Mat grayImg;		// 現在フレームのグレー画像
	cv::Mat prevImg;		// 1フレーム前のカラー画像
	cv::Mat prevGrayImg;	// 1フレーム前のグレー画像
	std::vector<cv::KeyPoint> currentKpts;
	std::vector<cv::KeyPoint> prevKpts;
	cv::Mat currentDesc;
	cv::Mat prevDesc;
	std::vector<cv::DMatch> matchVector;
	std::vector<cv::Point2f> vanishPointVector;
	std::vector<cv::Point2f> vanishPointVectorMA;
	std::vector<cv::Point2f> vanishPointVectorWA;
};

#endif // DIRECTION_ESTIMATOR_H