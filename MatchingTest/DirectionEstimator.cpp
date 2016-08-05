#include "DirectionEstimator.h"

#include <fstream>
#include <chrono>

const int DirectionEstimator::POINT_SIZE = 5;
const Scalar DirectionEstimator::SCALAR_RED(0, 0, 255);
const Scalar DirectionEstimator::SCALAR_GREEN(0, 255, 0);
const Scalar DirectionEstimator::SCALAR_BLUE(255, 0, 0);
const Scalar DirectionEstimator::SCALAR_YELLOW(0, 255, 255);
const Scalar DirectionEstimator::SCALAR_PURPLE(255, 0, 255);
const Scalar DirectionEstimator::SCALAR_CYAN(255, 255, 0);
const Scalar DirectionEstimator::SCALAR_BLACK(0, 0, 0);
const Scalar DirectionEstimator::SCALAR_WHITE(255, 255, 255);
const int DirectionEstimator::FLOW_LINE_MIN_LIMIT = 0;
const int DirectionEstimator::FLOW_LINE_MAX_LIMIT = 100;
const int DirectionEstimator::FRAME_SPAN = 9;

DirectionEstimator::DirectionEstimator()
{
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
}

void DirectionEstimator::clear()
{
	isFirstFrame = true;
	isSaveImg = false;
	count = 1;
	currentKpts.clear();
	prevKpts.clear();
	matchVector.clear();
	vanishPointVector.clear();
	// RGBAとGrayのサイズ
	int width = 640;
	int height = 480;
	grayImg = Mat(Size(width, height), CV_8UC1);
	prevImg = Mat(Size(width, height), CV_8UC4);
	prevGrayImg = Mat(Size(width, height), CV_8UC1);
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
	end = chrono::system_clock::now();
	double elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	cout << "elapsed time(ms):" << elapsed << endl;

	if (!isFirstFrame) {
		calcMatchingFlow();
		Point2f vp = getCrossPoint();
		vanishPointVector.push_back(vp); // 消失点計算
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
	for (int i = 0; i < matchPrevToCur.size(); i++)
	{
		DMatch forward = matchPrevToCur[i]; // prev=query, current=train
		DMatch backward = matchCurToPrev[forward.trainIdx];
		if (backward.trainIdx == forward.queryIdx) {
			float dist = getDistance(prevKpts[forward.queryIdx].pt, currentKpts[forward.trainIdx].pt);
			if (dist <= FLOW_LINE_MAX_LIMIT)
				matchVector.push_back(forward);
		}
	}
}

Point2f DirectionEstimator::getCrossPoint()
{
	float X = 320;
	float Y = 240;
	int flowNum = matchVector.size();
	float a = 0;
	float b = 0;
	float p = 0;
	float c = 0;
	float d = 0;
	float q = 0;
	float bunbo = 0;
	for (int i = 0; i < flowNum; ++i) {
		Point2f p1 = currentKpts[matchVector[i].trainIdx].pt;
		Point2f p2 = prevKpts[matchVector[i].queryIdx].pt;

		// 連立方程式公式 - https://t-sv.sakura.ne.jp/text/num_ana/ren_eq22/ren_eq22.html
//		sumX += 2*X * (p1.y - p2.y) * (p1.y - p2.y) + 2*Y * (p2.x - p1.x) * (p1.y - p2.y)
//				+ 2 * (p1.x * p2.y - p2.x * p1.y) * (p1.y - p2.y); // = 0 偏微分X
//		sumY += 2*Y * (p2.x - p1.x) * (p2.x - p1.x) + 2*X * (p2.x - p1.x) * (p1.y - p2.y)
//				+ 2 * (p1.x * p2.y - p2.x * p1.y) * (p2.x - p1.x); // = 0 偏微分Y

		a += (p1.y - p2.y) * (p1.y - p2.y);
		b += (p2.x - p1.x) * (p1.y - p2.y);
		p += (p1.x * p2.y - p2.x * p1.y) * (p1.y - p2.y);
		c += (p2.x - p1.x) * (p1.y - p2.y);
		d += (p2.x - p1.x) * (p2.x - p1.x);
		q += (p1.x * p2.y - p2.x * p1.y) * (p2.x - p1.x);
	}
	p *= -1;
	q *= -1;
	bunbo = (a * d - b * c);
	if (bunbo == 0) return Point2f(-999, -999);
	X = (d * p - b * q) / bunbo;
	Y = (a * q - c * p) / bunbo;
	return Point2f(X, Y);
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
	string path = ".\\img\\";
	Mat out;
	rgbaImg.copyTo(out);
	for (int i = 0; i < matchVector.size(); ++i) {
		circle(out, prevKpts[matchVector[i].queryIdx].pt, POINT_SIZE, SCALAR_BLUE, 3);
	}
	for (int i = 0; i < matchVector.size(); ++i) {
		circle(out, currentKpts[matchVector[i].trainIdx].pt, POINT_SIZE, SCALAR_YELLOW, 3);
	}

	// フロー描画
	for (int i = 0; i < matchVector.size(); ++i) {
		line(out, prevKpts[matchVector[i].queryIdx].pt, currentKpts[matchVector[i].trainIdx].pt, SCALAR_GREEN, 3);
	}

	// 消失点描画
	circle(out, vanishPointVector.back(), POINT_SIZE * 3, SCALAR_CYAN, -1);
	string countStr = to_string(count);
	imwrite(path + "matching_" + countStr + ".jpg", out);
}

void DirectionEstimator::drawVanishPointHistory()
{
	const int graphWidth = 500;
	const int graphHeight = 500;
	const int graphWidthHalf = graphWidth / 2;
	const int widthDigMax = 1000; // 片側(プラス)の目盛り最大値
	const float widthScale = (float)graphWidthHalf / widthDigMax;
	const float heightSpan = (float)graphHeight / (vanishPointVector.size() - 1);

	Mat out(graphHeight, graphWidth, CV_8UC3);
	rectangle(out, Point(0, 0), Point(graphWidth, graphHeight), SCALAR_WHITE, -1); // 背景描画
	line(out, Point(graphWidthHalf, 0), Point(graphWidthHalf, graphHeight), SCALAR_BLACK, 1); // 軸線
	for (int i = 1; i < vanishPointVector.size(); i++)
	{
		Point p1 = Point((vanishPointVector[i - 1].x - 320) * widthScale + graphWidthHalf, heightSpan * (i - 1));
		Point p2 = Point((vanishPointVector[i].x - 320) * widthScale + graphWidthHalf, heightSpan * i);
		if (vanishPointVector[i-1].x == -999 && vanishPointVector[i-1].y == -999) // 一つ前のデータがエラー値
			line(out, p2, p2, SCALAR_RED, 2);
		else if (vanishPointVector[i].x != -999 && vanishPointVector[i].y != -999) // 現在のデータがエラー値
			line(out, p1, p2, SCALAR_RED, 1);
	}
	imshow("vanish point history x", out);
	imwrite("vanishPointXHistory.jpg", out);
}

void DirectionEstimator::logVanishPointHistory(const string& fileName) {
	ofstream ofs(fileName);
	ofs << "x,y" << endl;
	for (int i = 0; i < vanishPointVector.size(); i++)
	{
		ofs << vanishPointVector[i].x << "," << vanishPointVector[i].y << endl;
	}
}

void DirectionEstimator::readVanishPointHistory(const string& filePath)
{
	ifstream ifs(filePath);
	string line;
	const char delimiter = ',';
	while (!ifs.eof())
	{
		getline(ifs, line);
		string separatedX, separatedY;
		istringstream lineSeparater(line);
		getline(lineSeparater, separatedX, delimiter);
		getline(lineSeparater, separatedY, delimiter);
		float x = 0;
		float y = 0;
		try {
			x = stof(separatedX);
			y = stof(separatedY);
		} catch (const invalid_argument& ia) {
			continue;
		}
		vanishPointVector.push_back(Point2f(x, y));
	}
}

void DirectionEstimator::setIsSaveImg(bool flag)
{
	isSaveImg = flag;
}

