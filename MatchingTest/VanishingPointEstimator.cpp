#include "VanishingPointEstimator.h"
#include <fstream>
#include "Utils.h"

using namespace cv;
using namespace std;

const Scalar VanishingPointEstimator::SCALAR_CYAN(255, 255, 0);
const Scalar VanishingPointEstimator::SCALAR_RED(0, 0, 255);
const Scalar VanishingPointEstimator::SCALAR_GREEN(0, 255, 0);
const Scalar VanishingPointEstimator::SCALAR_BLACK(0, 0, 0);
const Scalar VanishingPointEstimator::SCALAR_WHITE(255, 255, 255);
const int VanishingPointEstimator::ERROR_VP = -9999;
const int VanishingPointEstimator::VPOINT_SIZE = 10;
const string VanishingPointEstimator::RESULT_PATH = ".\\result\\";
const int VanishingPointEstimator::IMG_WIDTH = 640;
const double VanishingPointEstimator::W_MAX = 0.5;
const int VanishingPointEstimator::GRAPH_WIDTH = 500;
const int VanishingPointEstimator::GRAPH_HEIGHT = 700;
const double VanishingPointEstimator::INF = 100000000;

VanishingPointEstimator::VanishingPointEstimator()
{
	maFilterX = new MovingAverageFilter();
	maFilterY = new MovingAverageFilter();
	waFilterX = new WeightedAverageFilter();
	waFilterY = new WeightedAverageFilter();
	clear();
}

VanishingPointEstimator::~VanishingPointEstimator()
{
	delete maFilterX;
	delete maFilterY;
	delete waFilterX;
	delete waFilterY;
}

void VanishingPointEstimator::clear()
{
	pointHistory.clear();
	pointHistoryMA.clear();
	pointHistoryWA.clear();
	maFilterX->clear();
	maFilterY->clear();
	waFilterX->clear();
	waFilterY->clear();
}

void VanishingPointEstimator::estimate(const vector<DMatch>& matchVector, 
	const vector<KeyPoint>& currentKpts, const vector<KeyPoint>& prevKpts)
{
	//Point2f tmpVp = getCrossPoint(matchVector, currentKpts, prevKpts);
	Point2f vp = getCrossPoint(matchVector, currentKpts, prevKpts);
	pointHistory.push_back(vp); // 消失点計算
	if (vp.x == ERROR_VP && vp.y == ERROR_VP) {
		pointHistoryMA.push_back(vp);
		pointHistoryWA.push_back(vp);
	}
	else {
		Point2f vpMA = Point2f(maFilterX->update(vp.x), maFilterY->update(vp.y));
		pointHistoryMA.push_back(vpMA);
		Point2f vpWA = Point2f(waFilterX->update(vp.x), waFilterY->update(vp.y));
		pointHistoryWA.push_back(vpWA);
	}
}

Point2f VanishingPointEstimator::getCrossPoint(const vector<DMatch>& matchVector, 
	const vector<KeyPoint>& currentKpts, const vector<KeyPoint>& prevKpts)
{
	int flowNum = (int)matchVector.size();
	float a = 0;
	float b = 0;
	float p = 0;
	float c = 0;
	float d = 0;
	float q = 0;
	float bunbo = 0;
	for (int i = 0; i < flowNum; ++i) {
		Point2f pp = prevKpts[matchVector[i].queryIdx].pt;
		Point2f cp = currentKpts[matchVector[i].trainIdx].pt; 

		// 連立方程式公式 - https://t-sv.sakura.ne.jp/text/num_ana/ren_eq22/ren_eq22.html
		//		sumX += 2*X * (pp.y - cp.y) * (pp.y - cp.y) + 2*Y * (cp.x - pp.x) * (pp.y - cp.y)
		//				+ 2 * (pp.x * cp.y - cp.x * pp.y) * (pp.y - cp.y); // = 0 偏微分X
		//		sumY += 2*Y * (cp.x - pp.x) * (cp.x - pp.x) + 2*X * (cp.x - pp.x) * (pp.y - cp.y)
		//				+ 2 * (pp.x * cp.y - cp.x * pp.y) * (cp.x - pp.x); // = 0 偏微分Y

		a += (pp.y - cp.y) * (pp.y - cp.y);
		b += (cp.x - pp.x) * (pp.y - cp.y);
		p += (pp.x * cp.y - cp.x * pp.y) * (pp.y - cp.y);
		c += (cp.x - pp.x) * (pp.y - cp.y);
		d += (cp.x - pp.x) * (cp.x - pp.x);
		q += (pp.x * cp.y - cp.x * pp.y) * (cp.x - pp.x);
	}
	p *= -1;
	q *= -1;
	bunbo = (a * d - b * c);
	if (bunbo == 0) return Point2f(ERROR_VP, ERROR_VP);
	float X = (d * p - b * q) / bunbo;
	float Y = (a * q - c * p) / bunbo;
	return Point2f(X, Y);
}

Point2f VanishingPointEstimator::getCrossPointWeighted(const Point2f& crossP, const vector<DMatch>& matchVector,
	const vector<KeyPoint>& currentKpts, const vector<KeyPoint>& prevKpts)
{
	float a = 0;
	float b = 0;
	float p = 0;
	float c = 0;
	float d = 0;
	float q = 0;
	float bunbo = 0;

	int flowNum = (int)matchVector.size();
	double* weight = new double[flowNum];
	for (int i = 0; i < flowNum; ++i) {
		Point2f p1 = currentKpts[matchVector[i].trainIdx].pt;	// 現在フレーム特徴点(p1)
		Point2f p2 = prevKpts[matchVector[i].queryIdx].pt;	// 前フレーム特徴点(p2)

		// ベクトル計算
		Point2f crossToCurV = Point2f(p1.x - crossP.x, p1.y - crossP.y); // 現在フレーム特徴点 >> 消失点
		Point2f crossToPrevV = Point2f(p2.x - crossP.x, p2.y - crossP.y); // 前フレーム特徴点 >> 消失点
		//cout << "crossToCurV(" << crossToCurV.x << ", " << crossToCurV.y << ")" << endl;
		//cout << "crossToPrevV(" << crossToPrevV.x << ", " << crossToPrevV.y << ")" << endl;
		// 正規化
		normalize(crossToCurV);
		normalize(crossToPrevV);
		//cout << "WcrossToCurV(" << crossToCurV.x << ", " << crossToCurV.y << ")" << endl;
		//cout << "WcrossToPrevV(" << crossToPrevV.x << ", " << crossToPrevV.y << ")" << endl;
		// 外積計算
		double error = crossToCurV.x * crossToPrevV.y - crossToCurV.y * crossToPrevV.x;
		if (abs(error) > W_MAX) {
			weight[i] = 0;
			cout << "[" << i << "] error:" << error << ", weight:" << weight[i] << endl;
		}
		else {
			float dist = 1 - (error / W_MAX) * (error / W_MAX);
			weight[i] = dist * dist;
		}
		//cout << "[" << i << "] error:" << error << ", weight:" << weight[i] << endl;

		// 連立方程式公式 - https://t-sv.sakura.ne.jp/text/num_ana/ren_eq22/ren_eq22.html
		//		sumX += 2*X * (p1.y - p2.y) * (p1.y - p2.y) * w[i] + 2*Y * (p2.x - p1.x) * (p1.y - p2.y) * w[i]
		//				+ 2 * (p1.x * p2.y - p2.x * p1.y) * (p1.y - p2.y) * w[i]; // = 0 偏微分X
		//		sumY += 2*Y * (p2.x - p1.x) * (p2.x - p1.x) * w[i] + 2*X * (p2.x - p1.x) * (p1.y - p2.y) * w[i]
		//				+ 2 * (p1.x * p2.y - p2.x * p1.y) * (p2.x - p1.x) * w[i]; // = 0 偏微分Y

		// 重み付き最少二乗
		a += (p1.y - p2.y) * (p1.y - p2.y) * weight[i];
		b += (p2.x - p1.x) * (p1.y - p2.y) * weight[i];
		p += (p1.x * p2.y - p2.x * p1.y) * (p1.y - p2.y) * weight[i];
		c += (p2.x - p1.x) * (p1.y - p2.y) * weight[i];
		d += (p2.x - p1.x) * (p2.x - p1.x) * weight[i];
		q += (p1.x * p2.y - p2.x * p1.y) * (p2.x - p1.x) * weight[i];
	}
	p *= -1;
	q *= -1;
	bunbo = (a * d - b * c);
	delete[] weight;
	if (bunbo == 0) return Point2f(ERROR_VP, ERROR_VP);
	float X = (d * p - b * q) / bunbo;
	float Y = (a * q - c * p) / bunbo;
	return Point2f(X, Y);
}

void VanishingPointEstimator::normalize(Point2f &vec)
{
	float length = sqrtf(vec.x * vec.x + vec.y * vec.y);
	vec.x /= length;
	vec.y /= length;
}

void VanishingPointEstimator::drawLastVP(Mat &outImg)
{
	circle(outImg, pointHistory.back(), VPOINT_SIZE, SCALAR_CYAN, -1);
	circle(outImg, pointHistoryMA.back(), VPOINT_SIZE, SCALAR_RED, -1);
	circle(outImg, pointHistoryWA.back(), VPOINT_SIZE, SCALAR_GREEN, -1);
}

Mat VanishingPointEstimator::getVanishPointHistory()
{
	const int graphWidthHalf = GRAPH_WIDTH / 2;
	const int graphWidthQuarter = graphWidthHalf / 2;
	const int widthDigMax = 2000; // 片側(プラス)の目盛り最大値
	const float widthScale = (float)graphWidthHalf / widthDigMax;
	const float heightSpan = (float)GRAPH_HEIGHT / (pointHistory.size() - 1);

	Mat out(GRAPH_HEIGHT, GRAPH_WIDTH, CV_8UC3);
	rectangle(out, Point(0, 0), Point(GRAPH_WIDTH, GRAPH_HEIGHT), SCALAR_WHITE, -1); // 背景描画
	line(out, Point(graphWidthHalf, 0), Point(graphWidthHalf, GRAPH_HEIGHT), SCALAR_BLACK, 1); // 軸線
	const int imgWidthHalf = IMG_WIDTH / 2;
	// 320px(中心)線
	Point p1 = Point((int)(imgWidthHalf * widthScale + graphWidthHalf), 0);
	Point p2 = Point((int)(imgWidthHalf * widthScale + graphWidthHalf), GRAPH_HEIGHT);
	drawDashedLine(out, p1, p2, SCALAR_BLACK, 1, 40);
	// 640px(右端)線
	Point pp1 = Point((int)(IMG_WIDTH * widthScale + graphWidthHalf), 0);
	Point pp2 = Point((int)(IMG_WIDTH * widthScale + graphWidthHalf), GRAPH_HEIGHT);
	line(out, pp1, pp2, SCALAR_BLACK, 1); // 軸線
	for (int i = 1; i < pointHistory.size(); i++)
	{
		drawVPLine(out, pointHistory, SCALAR_CYAN, i, widthScale, graphWidthHalf, heightSpan);
		drawVPLine(out, pointHistoryMA, SCALAR_RED, i, widthScale, graphWidthHalf, heightSpan);
		drawVPLine(out, pointHistoryWA, SCALAR_GREEN, i, widthScale, graphWidthHalf, heightSpan);
	}
	return out;
}

void VanishingPointEstimator::drawDashedLine(cv::Mat& out, const cv::Point& p1, const cv::Point& p2, const cv::Scalar& color, const int lineWidth, const int interval)
{
	float dx = (p2.x - p1.x) / interval;
	float dy = (p2.y - p1.y) / interval;
	for (int i = 0; i < interval + 1; i++)
	{
		if (i % 2 == 0)
			line(out, Point(p1.x + dx * i, p1.y + dy * i), Point(p1.x + dx * (i + 1), p1.y + dy * (i + 1)), color, lineWidth);
	}
}

void VanishingPointEstimator::drawVPLine(Mat& out, const vector<Point2f>& vpHistory, const Scalar color, const int i,
	const float widthScale, const int graphWidthHalf, const float heightSpan)
{
	const int imgWidthHalf = IMG_WIDTH / 2;
	Point p1 = Point((int)((vpHistory[i - 1].x) * widthScale + graphWidthHalf), (int)(heightSpan * (i - 1)));
	Point p2 = Point((int)((vpHistory[i].x) * widthScale + graphWidthHalf), (int)(heightSpan * i));
	if (vpHistory[i - 1].x == ERROR_VP && vpHistory[i - 1].y == ERROR_VP) // 一つ前のデータがエラー値
		line(out, p2, p2, color, 1);
	else if (vpHistory[i].x != ERROR_VP && vpHistory[i].y != ERROR_VP) // 現在のデータがエラー値
		line(out, p1, p2, color, 1);
	circle(out, p2, VPOINT_SIZE / 2, color, -1);
}

void VanishingPointEstimator::logVanishPointHistoryAll(const string& fileName)
{
	logVanishPointHistory(fileName, pointHistory);
	logVanishPointHistory("MA_" + fileName, pointHistoryMA);
	logVanishPointHistory("WA_" + fileName, pointHistoryWA);
}

void VanishingPointEstimator::logVanishPointHistory(const string& fileName, const vector<Point2f>& vpHistory) {
	ofstream ofs(RESULT_PATH + fileName);
	ofs << "x,y" << endl;
	for (int i = 0; i < vpHistory.size(); i++)
	{
		ofs << vpHistory[i].x << "," << vpHistory[i].y << endl;
	}
}

void VanishingPointEstimator::readVanishPointHistoryAll(const string& filePath)
{
	readVanishPointHistory(filePath, pointHistory);
	readVanishPointHistory("MA_" + filePath, pointHistoryMA);
	readVanishPointHistory("WA_" + filePath, pointHistoryWA);
}

void VanishingPointEstimator::readVanishPointHistory(const string& filePath, vector<Point2f>& vpHistory)
{
	ifstream ifs(RESULT_PATH + filePath);
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
		}
		catch (const invalid_argument& ia) {
			continue;
		}
		vpHistory.push_back(Point2f(x, y));
	}
}

void VanishingPointEstimator::addVPHistory(Point2f vp)
{
	pointHistory.push_back(vp);
	pointHistoryMA.push_back(vp);
	pointHistoryWA.push_back(vp);
}
