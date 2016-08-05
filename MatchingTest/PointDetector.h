#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class PointDetector
{
public:
	PointDetector();
	~PointDetector();

	void init();
	void release();
	void match(const Mat &query, const Mat &train, vector<DMatch> &vmatch) const;
	void describe(const Mat &img, vector<KeyPoint> &vkpt, Mat &vdesc) const;
	void detect(const Mat &img, vector<KeyPoint> &vkpt) const;

private:
	Ptr<AKAZE> mDetector;
	Ptr<DescriptorMatcher> mMatcher;
};

