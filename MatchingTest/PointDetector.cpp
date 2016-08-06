#include "PointDetector.h"

using namespace cv;
using namespace std;

PointDetector::PointDetector()
{
	init();
}

PointDetector::~PointDetector()
{
	release();
}

void PointDetector::init()
{
	//mDetector = AKAZE::create();
	// threshold=0.001(default)
	mDetector = AKAZE::create(5, 0, 3, 0.0005, 4, 4, 1);
	mMatcher = DescriptorMatcher::create("BruteForce-Hamming");
}

void PointDetector::release()
{
	mDetector.release();
	mMatcher.release();
}

void PointDetector::match(const Mat &query, const Mat &train, vector<DMatch> &vmatch) const
{
	mMatcher->match(query, train, vmatch);
}

void PointDetector::describe(const Mat &img, vector<KeyPoint> &vkpt, Mat &vdesc) const
{
	mDetector->compute(img, vkpt, vdesc);
}

void PointDetector::detect(const Mat &img, vector<KeyPoint> &vkpt) const
{
	mDetector->detect(img, vkpt);
}
