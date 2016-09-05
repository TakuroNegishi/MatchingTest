#include <opencv2/opencv.hpp>
#include <exception>
#include <Windows.h>
#include "DirectionEstimator.h"

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

int main() {
	//string rootPath = "D:\\Research\\院研究\\特徴点動き\\20160802\\straight_right_curve\\img\\";
	//string rootPath = "D:\\Research\\院研究\\特徴点動き\\20160802\\straight_look_around\\img\\";
	//string rootPath = "D:\\Research\\院研究\\特徴点動き\\20160809\\straight_look_around_slow\\img\\";
	string rootPath = "D:\\Research\\院研究\\特徴点動き\\20160809\\straight_left_curve_slow\\img\\";
	//string rootPath = "D:\\Research\\院研究\\特徴点動き\\20160809\\curve_test4\\img\\";
	//string rootPath = "D:\\Research\\院研究\\特徴点動き\\20160825\\straight_left_curve_normal\\img\\";
	string gyroCsvPath = "D:\\Research\\院研究\\特徴点動き\\20160809\\straight_left_curve_slow\\2016-09-06 01;39;40_gyro.csv";

	DirectionEstimator* de = new DirectionEstimator();

	de->setIsSaveImg(true);
	vector<string> imgFiles = getFiles(rootPath, "*.jpg");
	while (imgFiles.size() > 0) {
		cout << "-> " << imgFiles.size() << endl;
		Mat cameraImg = imread(rootPath + imgFiles.front());
		de->estimate(cameraImg);
		imgFiles.erase(imgFiles.begin());
		waitKey(50);
	}
	de->logVPHistory("vanishPointHistory.txt");

	//de->readVPHistory("vanishPointHistory.txt");
	//de->drawVPHistory();
	waitKey(0);
	destroyAllWindows();

	delete de;
	return 0;
}