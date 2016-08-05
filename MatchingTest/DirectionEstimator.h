#pragma once

#include <opencv2/opencv.hpp>
#include "PointDetector.h"

using namespace std;
using namespace cv;

class DirectionEstimator
{
public:
	DirectionEstimator();
	~DirectionEstimator();
	void clear();
	void estimate(const Mat &rgbaImg);
	void calcMatchingFlow();
	Point2f getCrossPoint();
	float getDistance(const Point2f &pt1, const Point2f &pt2);
	void draw(const Mat &rgbaImg);
	void drawVanishPointHistory();
	void logVanishPointHistory(const string& fileName);
	void readVanishPointHistory(const string& filePath);
	void setIsSaveImg(bool flag);

private:
	static const int POINT_SIZE;			// 特徴点の描画半径
	static const Scalar SCALAR_RED;
	static const Scalar SCALAR_GREEN;
	static const Scalar SCALAR_BLUE;
	static const Scalar SCALAR_YELLOW;
	static const Scalar SCALAR_PURPLE;
	static const Scalar SCALAR_CYAN;
	static const Scalar SCALAR_BLACK;
	static const Scalar SCALAR_WHITE;
	static const int FLOW_LINE_MIN_LIMIT;	// 許容する特徴点の最小距離距離
	static const int FLOW_LINE_MAX_LIMIT;	// 許容する特徴点の最大距離距離
	static const int FRAME_SPAN;

	bool isFirstFrame;
	bool isSaveImg;
	int count;
	PointDetector detector;
	Mat grayImg;		// 現在フレームのグレー画像
	Mat prevImg;		// 1フレーム前のカラー画像
	Mat prevGrayImg;	// 1フレーム前のグレー画像
	vector<KeyPoint> currentKpts;
	vector<KeyPoint> prevKpts;
	Mat currentDesc;
	Mat prevDesc;
	vector<DMatch> matchVector;
	vector<Point2f> vanishPointVector;
};
