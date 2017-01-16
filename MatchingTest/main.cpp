#include <opencv2/opencv.hpp>
#include <exception>
#include <Windows.h>
#include "DirectionEstimator.h"
#include <fstream>

using namespace cv;
using namespace std;

// 指定フォルダのファイルを取得する
vector<string> getFiles(const string& dir_path, const string& filter) {
	WIN32_FIND_DATAA fd;
	string ss = dir_path + filter;
	HANDLE hFind = FindFirstFileA(ss.c_str(), &fd);

	// 検索失敗
	if (hFind == INVALID_HANDLE_VALUE) throw exception("getFiles failed");

	vector<string> fileList;
	do {
		// フォルダは除く
		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
		// 隠しファイルは除く
		if (fd.dwFileAttributes & FILE_ATTRIBUTE_HIDDEN) continue;
		fileList.push_back(fd.cFileName);
	} while (FindNextFileA(hFind, &fd));
	FindClose(hFind);
	return fileList;
}

bool readMatBinary(std::ifstream& ifs, cv::Mat& in_mat, vector<Point2f>& currentKpts, vector<Point2f>& prevKpts, Point2f& vpMA)
{
	if (!ifs.is_open()){
		return false;
	}

	int rows, cols, type;
	ifs.read((char*)(&rows), sizeof(int));
	if (rows == 0){
		return true;
	}
	ifs.read((char*)(&cols), sizeof(int));
	ifs.read((char*)(&type), sizeof(int));

	in_mat.release();
	in_mat.create(rows, cols, type);
	ifs.read((char*)(in_mat.data), in_mat.elemSize() * in_mat.total());
	cout << "row: " << rows << ", col: " << cols << ", type: " << type << endl;

	currentKpts.clear();
	prevKpts.clear();
	int flowNum;
	ifs.read((char*)(&flowNum), sizeof(int));
	cout << "flowNum: " << flowNum << endl;
	for (int i = 0; i < flowNum; ++i)
	{
		Point2f prevP, currentP;
		ifs.read((char*)(&prevP), sizeof(Point2f));
		ifs.read((char*)(&currentP), sizeof(Point2f));
		prevKpts.push_back(prevP);
		currentKpts.push_back(currentP);
	}
	ifs.read((char*)(&vpMA), sizeof(Point2f));
	cout << "vpMA:(" << vpMA.x << ", " << vpMA.y << ")" << endl;

	return true;
}

bool LoadMatBinary(const std::string& filename, cv::Mat& output, vector<Point2f>& currentKpts, vector<Point2f>& prevKpts, Point2f& vpMA){
	std::ifstream ifs(filename, std::ios::binary);
	return readMatBinary(ifs, output, currentKpts, prevKpts, vpMA);
}

void saveJpgFromDat(const string& rootPath)
{
	DirectionEstimator* de = new DirectionEstimator();
	de->setIsSaveImg(true);
	vector<string> datFiles = getFiles(rootPath, "*.dat");
	while (datFiles.size() > 0) {
		cout << "-> " << datFiles.size() << endl;
		Mat out;
		vector<Point2f> currentKpts;
		vector<Point2f> prevKpts;
		Point2f vpMA;

		LoadMatBinary(rootPath + datFiles.front(), out, currentKpts, prevKpts, vpMA);
		de->drawFromDat(out, currentKpts, prevKpts, vpMA);
		//imwrite(rootPath + datFiles.front() + ".jpg", out);
		datFiles.erase(datFiles.begin());
		waitKey(50);
	}
	delete de;
}

int main() {
	//string rootPath = "D:\\Research\\院研究\\特徴点動き\\20160906\\around_and_left_curve_light\\img\\";
	//string rootPath = "D:\\Research\\院研究\\特徴点動き\\20161018\\moverio_2\\img\\";
	//string rootPath = "D:\\Research\\院研究\\特徴点動き\\20161018\\屋外_1\\img\\";
	//string rootPath = "D:\\Research\\院研究\\特徴点動き\\20170112\\横向き早い\\3F前\\";

	string rootPath = "D:\\Research\\院研究\\特徴点動き\\20170112\\log_test\\";
	saveJpgFromDat(rootPath);

/*	DirectionEstimator* de = new DirectionEstimator();
	de->setIsSaveImg(true);
	vector<string> imgFiles = getFiles(rootPath, "*.jpg");
	while (imgFiles.size() > 0) {
		cout << "-> " << imgFiles.size() << endl;
		Mat cameraImg = imread(rootPath + imgFiles.front());

		// 画像を回転
		//Mat rotImg(cv::Size(480, 640), cameraImg.type(), cv::Scalar(0, 0, 0));
		//cv::transpose(cameraImg, rotImg);  // 転置 左回り 反時計回りに90度回転 
		//cv::flip(rotImg, rotImg, 1);    // 左右反転 時計回りに90度回転

		de->estimate(cameraImg);
		imgFiles.erase(imgFiles.begin());
		waitKey(50);
	}
	de->logVPHistory("vanishPointHistory.txt");

	waitKey(0);
	destroyAllWindows();
	delete de;
*/	return 0;
}