#include "DirectionEstimator.h"

#include <chrono>
//#include <iostream>
//#include <sstream>
#include <iomanip>

using namespace std;
using namespace cv;

const int DirectionEstimator::POINT_SIZE = 5;
//const int DirectionEstimator::POINT_SIZE = 2;
const Scalar DirectionEstimator::SCALAR_GREEN(0, 255, 0);
const Scalar DirectionEstimator::SCALAR_LIGHT_GREEN(144, 238, 144);
const Scalar DirectionEstimator::SCALAR_BLUE(255, 0, 0);
const Scalar DirectionEstimator::SCALAR_RED(0, 0, 255);
const Scalar DirectionEstimator::SCALAR_YELLOW(0, 255, 255);
const Scalar DirectionEstimator::SCALAR_BLACK(0, 0, 0);
const Scalar DirectionEstimator::SCALAR_WHITE(255, 255, 255);
const int DirectionEstimator::FLOW_LINE_MIN_LIMIT = 0;
const int DirectionEstimator::FLOW_LINE_MAX_LIMIT = 9999;
const int DirectionEstimator::FRAME_SPAN = 1;
const string DirectionEstimator::RESULT_PATH = ".\\result\\";
const int DirectionEstimator::IMG_WIDTH = 640;
const int DirectionEstimator::IMG_HEIGHT = 480;

DirectionEstimator::DirectionEstimator()
{
	vanishingPointEstimator = new VanishingPointEstimator();
	clear();
	detector.init();
}

DirectionEstimator::~DirectionEstimator()
{
	prevImg.release();
	grayImg.release();
	prevGrayImg.release();
	currentDesc.release();
	prevDesc.release();
	detector.release();
	delete vanishingPointEstimator;
}

void DirectionEstimator::clear()
{
	isFirstFrame = true;
	isSaveImg = false;
	count = 1;
	currentKpts.clear();
	prevKpts.clear();
	matchVector.clear();
	inlierMatches.clear();
	vanishingPointEstimator->clear();
	grayImg = Mat(Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC1);
	prevImg = Mat(Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC4);
	prevGrayImg = Mat(Size(IMG_WIDTH, IMG_HEIGHT), CV_8UC1);
	writer = VideoWriter(RESULT_PATH + "output.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), 2.0f, 
		Size(IMG_WIDTH + VanishingPointEstimator::GRAPH_WIDTH, VanishingPointEstimator::GRAPH_HEIGHT));
	if (!writer.isOpened())
		cout << "video writer >> open failed." << endl;
	else
		cout << "video writer >> open successed." << endl;
}

void DirectionEstimator::estimate(const Mat &rgbaImg)
{
	if (!isFirstFrame && count % FRAME_SPAN != 0) {
		count++;
		return;
	}
	cvtColor(rgbaImg, grayImg, COLOR_BGR2GRAY); // グレースケール
	currentKpts.clear();	// 特徴点リストクリア

	chrono::system_clock::time_point  start, end;
	start = chrono::system_clock::now();
	detector.detect(grayImg, currentKpts);
	detector.describe(grayImg, currentKpts, currentDesc);

	if (!isFirstFrame) {
		calcMatchingFlow();
		vanishingPointEstimator->estimate(inlierMatches, currentKpts, prevKpts);
	}
	end = chrono::system_clock::now();
	long long elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	cout << "elapsed time(ms):" << elapsed << endl;

	if (!isFirstFrame) {
		if (isSaveImg) draw(rgbaImg); // 特徴点等の描画
	}

	// 現在フレーム情報 -> 過去フレーム情報
	rgbaImg.copyTo(prevImg); // カラー画像
	grayImg.copyTo(prevGrayImg); // グレー画像
	prevKpts = currentKpts;
	currentDesc.copyTo(prevDesc);
	isFirstFrame = false;
	count++;
}

void DirectionEstimator::calcMatchingFlow()
{
	matchVector.clear();
	vector<DMatch> matchPrevToCur, matchCurToPrev;
	if (prevDesc.rows == 0 || currentDesc.rows == 0) return;
	detector.match(prevDesc, currentDesc, matchPrevToCur); // prev=query, current=train
	detector.match(currentDesc, prevDesc, matchCurToPrev);
	if ( matchPrevToCur.size() == 0) return;
	vector<Point2f> goodPrevPts, goodCurrentPts;
	for (int i = 0; i < matchPrevToCur.size(); i++)
	{
		DMatch forward = matchPrevToCur[i]; // prev=query, current=train
		DMatch backward = matchCurToPrev[forward.trainIdx];
		if (backward.trainIdx == forward.queryIdx) {
			float dist = getDistance(prevKpts[forward.queryIdx].pt, currentKpts[forward.trainIdx].pt);
			if (dist <= FLOW_LINE_MAX_LIMIT) {
				matchVector.push_back(forward);
				goodPrevPts.push_back(prevKpts[forward.queryIdx].pt);
				goodCurrentPts.push_back(currentKpts[forward.trainIdx].pt);
			}
		}
	}

	//ホモグラフィ行列推定
	Mat masks;
	Mat H;
	H = findHomography(goodPrevPts, goodCurrentPts, masks, RANSAC, 6.f);

	//RANSACで使われた対応点のみ抽出
	inlierMatches.clear();
	for (auto i = 0; i < masks.rows; ++i) {
		uchar *inlier = masks.ptr<uchar>(i);
		if (inlier[0] == 1) {
			inlierMatches.push_back(matchVector[i]);
		}
	}
}

float DirectionEstimator::getDistance(const Point2f &pt1, const Point2f &pt2)
{
	float dx = pt2.x - pt1.x;
	float dy = pt2.y - pt1.y;
	return sqrt(dx * dx + dy * dy);
}

// 描画処理
void DirectionEstimator::draw(const Mat &rgbaImg)
{
	Mat out;
	rgbaImg.copyTo(out);
	for (int i = 0; i < prevKpts.size(); ++i) {
		circle(out, prevKpts[i].pt, POINT_SIZE, SCALAR_BLUE, -1);
	}
	for (int i = 0; i < currentKpts.size(); ++i) {
		circle(out, currentKpts[i].pt, POINT_SIZE, SCALAR_YELLOW, -1);
	}

	// フロー描画
	for (int i = 0; i < matchVector.size(); ++i) {
		line(out, prevKpts[matchVector[i].queryIdx].pt, currentKpts[matchVector[i].trainIdx].pt, SCALAR_WHITE, 2);
	}
	for (int i = 0; i < inlierMatches.size(); ++i) {
		line(out, prevKpts[inlierMatches[i].queryIdx].pt, currentKpts[inlierMatches[i].trainIdx].pt, SCALAR_LIGHT_GREEN, 2);
	}

	// 消失点描画
	vanishingPointEstimator->drawLastVP(out);
	// 消失点履歴グラフ描画
	Mat vpHistory = vanishingPointEstimator->getVanishPointHistory();
	Mat combined = Mat::zeros(vpHistory.rows, out.cols + vpHistory.cols, CV_8UC3);
	Mat imageLeft(combined, Rect(0, 0, out.cols, out.rows));
	Mat imageRight(combined, Rect(out.cols, 0, vpHistory.cols, vpHistory.rows));
	out.copyTo(imageLeft);
	vpHistory.copyTo(imageRight);

	string countStr = to_string(count);
	ostringstream oss;
	oss << setw(4) << setfill('0') << countStr;
	imwrite(RESULT_PATH + "matching_" + oss.str() + ".jpg", combined);
	writer << combined;
}

void DirectionEstimator::drawFromDat(Mat &out, const vector<Point2f>& current, const vector<Point2f>& prev, const Point2f& vpMA)
{
	count++;
	//Mat out2;
	//out.copyTo(out2);
	//cvtColor(out2, out, COLOR_BGRA2RGB); // 赤青反転
	// 特徴点描画
	int flowNum = prev.size();
	cout << "flow: " << flowNum << endl;
	for (int i = 0; i < flowNum; ++i) {
		//cout << "prev: " << prev[i].x << ", " << prev[i].y << endl;
		circle(out, prev[i], POINT_SIZE, SCALAR_BLUE, -1);
	}
	for (int i = 0; i < flowNum; ++i) {
		circle(out, current[i], POINT_SIZE, SCALAR_YELLOW, -1);
	}

	// フロー描画
	for (int i = 0; i < flowNum; ++i) {
		line(out, prev[i], current[i], SCALAR_LIGHT_GREEN, 2);
	}

	// 消失点描画
	circle(out, vpMA, VanishingPointEstimator::VPOINT_SIZE, SCALAR_RED, -1);
	//imshow("" + count, out);
	//waitKey();

	// 消失点履歴グラフ描画
	vanishingPointEstimator->addVPHistory(vpMA); // 消失点計算

	Mat vpHistory = vanishingPointEstimator->getVanishPointHistory();
	Mat combined = Mat::zeros(vpHistory.rows, out.cols + vpHistory.cols, CV_8UC3);
	Mat imageLeft(combined, Rect(0, 0, out.cols, out.rows));
	Mat imageRight(combined, Rect(out.cols, 0, vpHistory.cols, vpHistory.rows));
	out.copyTo(imageLeft);
	vpHistory.copyTo(imageRight);

	string countStr = to_string(count);
	ostringstream oss;
	oss << setw(4) << setfill('0') << countStr;
	imwrite(RESULT_PATH + "matching_" + oss.str() + ".jpg", combined);
	writer << combined;
}

void DirectionEstimator::logVPHistory(const string& fileName)
{
	vanishingPointEstimator->logVanishPointHistoryAll(fileName);
}

void DirectionEstimator::readVPHistory(const string& filePath)
{
	vanishingPointEstimator->readVanishPointHistoryAll(filePath);
}

void DirectionEstimator::drawVPHistory()
{
	//vanishingPointEstimator->drawVanishPointHistory();
}

void DirectionEstimator::setIsSaveImg(bool flag)
{
	isSaveImg = flag;
}

