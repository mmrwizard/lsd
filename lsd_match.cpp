#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/fast_line_detector.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

#include <iostream>
#include <cairo.h>
#include <cairo-pdf.h>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

void printVector(const std::vector<std::pair<int, int>>& vec) {
	for (const auto& pair : vec) {
		std::cout << "(" << pair.first << ", " << pair.second << ") ";
	}
	std::cout << std::endl;
}

std::vector<std::pair<int, int>> minIndicesInEachRow(const Eigen::MatrixXf& A) {
	std::vector<std::pair<int, int>> indices;

	for (int i = 0; i < A.rows(); i++) {
		int minIndex = 0;
		float minValue = A(i, 0);

		for (int j = 1; j < A.cols(); j++) {
			if (A(i, j) < minValue) {
				minValue = A(i, j);
				minIndex = j;
			}
		}

		indices.push_back({ i, minIndex });
	}

	return indices;
}


// Convert the input line segments to Hough space
void lineToHoughSpace(const std::vector<cv::Vec4f>& lines, const cv::Size& imageSize, cv::Mat& houghSpace) {
	int max_rho = static_cast<int>(sqrt(imageSize.width * imageSize.width + imageSize.height * imageSize.height));
	int rho_bins = max_rho * 2 + 1;
	int theta_bins = 180;
	houghSpace = cv::Mat::zeros(rho_bins, theta_bins, CV_32SC1);

	for (const auto& line : lines) {
		cv::Point2f point1(line[0], line[1]);
		cv::Point2f point2(line[2], line[3]);

		// Iterate through the line segment points
		for (double t = 0; t <= 1; t += 0.01) {
			int x = static_cast<int>(point1.x * t + point2.x * (1 - t));
			int y = static_cast<int>(point1.y * t + point2.y * (1 - t));

			// Iterate through theta values
			for (int theta = 0; theta < theta_bins; ++theta) {
				double theta_rad = (CV_PI * theta) / 180.0;
				int rho = static_cast<int>(x * cos(theta_rad) + y * sin(theta_rad)) + max_rho;

				// Increment the Hough space accumulator
				houghSpace.at<int>(rho, theta) += 1;
			}
		}
	}
}

vector<Vec4f> detectLines(const Mat& inputImage, double _length_threshold) {
	// 1.1 转换图像为灰度图
	Mat grayImage;
	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);

	// 2. 创建FastLineDetector并检测直线

	double _distance_threshold = 1.41421356;
	double _canny_low_thresh = 50.0;
	double _canny_high_thresh = 100.0;
	int _canny_aperture_size = 3;
	bool _do_merge = true;
	Ptr<FastLineDetector> fld = createFastLineDetector(_length_threshold, _distance_threshold, _canny_low_thresh, _canny_high_thresh, _canny_aperture_size, _do_merge);
	vector<Vec4f> lines;
	fld->detect(grayImage, lines);

	return lines;
}


int main(int argc, char** argv) {
	if (argc != 3)
	{
		cout << "Usage: ./lsd_example <path_to_image>" << endl;
		return -1;
	}

	// 1. 加载图像
	Mat image1 = imread(argv[1], IMREAD_COLOR);
	Mat image2 = imread(argv[2], IMREAD_COLOR);
	int img1_height = image1.rows, img1_width = image1.cols;
	int img2_height = image2.rows, img2_width = image2.cols;

	if (image1.empty() || image2.empty())
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	vector<Vec4f> lines1;
	vector<Vec4f> lines2;

	//2.检测直线
	double _length_threshold = 40;
	lines1 = detectLines(image1, _length_threshold);
	lines2 = detectLines(image2, _length_threshold);

	//3.将line1和line2中的每条直线段的参数输出到数组h（r, theta）
	vector<Vec2f> h_lines1, h_lines2;
	for (const auto& line : lines1) {
		float detx = line[2] - line[0];
		float dety = line[3] - line[1];
		float theta = atan(-detx / dety);
		float r = line[0] * cos(theta) + line[1] * sin(theta);
		float d = (line[0] + line[1]) / 2 - img2_width / 2;
		h_lines1.push_back(Vec2f(r, theta));
	}
	for (const auto& line : lines2) {
		float detx = line[2] - line[0];
		float dety = line[3] - line[1];
		float theta = atan(-detx / dety);
		float r = line[0] * cos(theta) + line[1] * sin(theta);
		float d = (line[0] + line[1]) / 2 - img2_width / 2;
		h_lines2.push_back(Vec2f(r, theta));
	}

	//4.计算直线特征描述符，由ehline（r,theta,d(0,1),d)组成
	int rows1 = static_cast<int>(h_lines1.size());
	int rows2 = static_cast<int>(h_lines2.size());
	int theta_weight = 100;
		
	Eigen::MatrixXf ehline1(rows1, 5), ehline2(rows2, 5);
	for (int i = 0; i < rows1; ++i) {
		ehline1(i, 0) = h_lines1[i][0];
		ehline1(i, 1) = h_lines1[i][1]* theta_weight;

		// 计算中点坐标
		float mid_x = (lines1[i](0) + lines1[i](2)) / 2 - img1_width / 2;
		float mid_y = (lines1[i](1) + lines1[i](3)) / 2 - img1_height / 2;

		// 判断象限并存储独热编码
		if (mid_x >= 0 && mid_y >= 0) { // 第一象限
			ehline1(i, 2) = 1;
			ehline1(i, 3) = 1;
		}
		else if (mid_x < 0 && mid_y >= 0) { // 第二象限
			ehline1(i, 2) = -1;
			ehline1(i, 3) = 1;
		}
		else if (mid_x < 0 && mid_y < 0) { // 第三象限
			ehline1(i, 2) = -1;
			ehline1(i, 3) = -1;
		}
		else { // 第四象限
			ehline1(i, 2) = 1;
			ehline1(i, 3) = -1;
		}

		// 计算距离并存储
		ehline1(i, 4) = sqrt(mid_x * mid_x + mid_y * mid_y);
	}

	for (int i = 0; i < rows2; ++i) {
		ehline2(i, 0) = h_lines2[i][0];
		ehline2(i, 1) = h_lines2[i][1]* theta_weight;

		// 计算中点坐标
		float mid_x = (lines2[i](0) + lines2[i](2)) / 2 - img2_width / 2;
		float mid_y = (lines2[i](1) + lines2[i](3)) / 2 - img2_height / 2;

		// 判断象限并存储独热编码
		if (mid_x >= 0 && mid_y >= 0) { // 第一象限
			ehline2(i, 2) = 1;
			ehline2(i, 3) = 1;
		}
		else if (mid_x < 0 && mid_y >= 0) { // 第二象限
			ehline2(i, 2) = -1;
			ehline2(i, 3) = 1;
		}
		else if (mid_x < 0 && mid_y < 0) { // 第三象限
			ehline2(i, 2) = -1;
			ehline2(i, 3) = -1;
		}
		else { // 第四象限
			ehline2(i, 2) = 1;
			ehline2(i, 3) = -1;
		}

		// 计算距离并存储
		ehline2(i, 4) = sqrt(mid_x * mid_x + mid_y * mid_y);
	}

	//export to csv file 
	std::ofstream file1("ehline1.csv");
	file1 << "r,theta,x,y,d\n";
	for (int i = 0; i < ehline1.rows(); ++i) {
		for (int j = 0; j < ehline1.cols(); ++j) {
			file1 << ehline1(i, j);
			if (j < ehline1.cols() - 1) {
				file1 << ",";
			}
		}
		file1 << "\n";
	}
	file1.close();

	std::ofstream file2("ehline2.csv");
	file2 << "r,theta,x,y,d\n";
	for (int i = 0; i < ehline2.rows(); ++i) {
		for (int j = 0; j < ehline2.cols(); ++j) {
			file2 << ehline2(i, j);
			if (j < ehline2.cols() - 1) {
				file2 << ",";
			}
		}
		file2 << "\n";
	}
	file2.close();

	/*std::cout << ehline1.size() << std::endl;
	std::cout << ehline2.size() << std::endl;*/

	//依据r和theta以及线段中点的距离的差值判断两直线是否匹配，同时接近与0则表示候选匹配对
	
	
	//对每个象限构造特征描述符矩阵
	//对每个象限构造特征描述符矩阵
	Eigen::MatrixXf ehline1_1, ehline1_2, ehline1_3, ehline1_4;
	Eigen::MatrixXf ehline2_1, ehline2_2, ehline2_3, ehline2_4;

	std::vector<Eigen::Vector4f> v1_1, v1_2, v1_3, v1_4;
	std::vector<Eigen::Vector4f> v2_1, v2_2, v2_3, v2_4;

	for (int i = 0; i < rows1; i++) {
		Eigen::Vector4f temp = Eigen::Vector4f(ehline1.row(i)(0), ehline1.row(i)(1), ehline1.row(i)(4), i);
		if (ehline1(i, 2) == 1 && ehline1(i, 3) == 1) {
			v1_1.push_back(temp);
		}
		else if (ehline1(i, 2) == -1 && ehline1(i, 3) == 1) {
			v1_2.push_back(temp);
		}
		else if (ehline1(i, 2) == -1 && ehline1(i, 3) == -1) {
			v1_3.push_back(temp);
		}
		else if (ehline1(i, 2) == 1 && ehline1(i, 3) == -1) {
			v1_4.push_back(temp);
		}
	}

	for (int i = 0; i < rows2; i++) {
		Eigen::Vector4f temp = Eigen::Vector4f(ehline2.row(i)(0), ehline2.row(i)(1), ehline2.row(i)(4), i);
		if (ehline2(i, 2) == 1 && ehline2(i, 3) == 1) {
			v2_1.push_back(temp);
		}
		else if (ehline2(i, 2) == -1 && ehline2(i, 3) == 1) {
			v2_2.push_back(temp);
		}
		else if (ehline2(i, 2) == -1 && ehline2(i, 3) == -1) {
			v2_3.push_back(temp);
		}
		else if (ehline2(i, 2) == 1 && ehline2(i, 3) == -1) {
			v2_4.push_back(temp);
		}
	}

	if (!v1_1.empty()) ehline1_1 = Eigen::Map<Eigen::MatrixXf>(v1_1[0].data(), 4, v1_1.size()).transpose();
	if (!v1_2.empty()) ehline1_2 = Eigen::Map<Eigen::MatrixXf>(v1_2[0].data(), 4, v1_2.size()).transpose();
	if (!v1_3.empty()) ehline1_3 = Eigen::Map<Eigen::MatrixXf>(v1_3[0].data(), 4, v1_3.size()).transpose();
	if (!v1_4.empty()) ehline1_4 = Eigen::Map<Eigen::MatrixXf>(v1_4[0].data(), 4, v1_4.size()).transpose();

	if (!v2_1.empty()) ehline2_1 = Eigen::Map<Eigen::MatrixXf>(v2_1[0].data(), 4, v2_1.size()).transpose();
	if (!v2_2.empty()) ehline2_2 = Eigen::Map<Eigen::MatrixXf>(v2_2[0].data(), 4, v2_2.size()).transpose();
	if (!v2_3.empty()) ehline2_3 = Eigen::Map<Eigen::MatrixXf>(v2_3[0].data(), 4, v2_3.size()).transpose();
	if (!v2_4.empty()) ehline2_4 = Eigen::Map<Eigen::MatrixXf>(v2_4[0].data(), 4, v2_4.size()).transpose();

	std::cout << "ehline1_1:\n" << ehline1_1 << std::endl;
	std::cout << "ehline2_1:\n" << ehline2_1 << std::endl;
	std::cout << "ehline1_3:\n" << ehline1_3 << std::endl;
	std::cout << "ehline2_3:\n" << ehline2_3 << std::endl;

	//计算欧式距离矩阵，忽略最后一列（索引列）
	Eigen::MatrixXf distMatrix1, distMatrix2, distMatrix3, distMatrix4;

	if (ehline1_1.size() > 0 && ehline2_1.size() > 0) {
		distMatrix1 = Eigen::MatrixXf::Zero(ehline1_1.rows(), ehline2_1.rows());
		for (int i = 0; i < ehline1_1.rows(); ++i) {
			for (int j = 0; j < ehline2_1.rows(); ++j) {
				distMatrix1(i, j) = (ehline1_1.row(i).head(3) - ehline2_1.row(j).head(3)).norm();
			}
		}
	}

	if (ehline1_2.size() > 0 && ehline2_2.size() > 0) {
		distMatrix2 = Eigen::MatrixXf::Zero(ehline1_2.rows(), ehline2_2.rows());
		for (int i = 0; i < ehline1_2.rows(); ++i) {
			for (int j = 0; j < ehline2_2.rows(); ++j) {
				distMatrix2(i, j) = (ehline1_2.row(i).head(3) - ehline2_2.row(j).head(3)).norm();
			}
		}
	}

	if (ehline1_3.size() > 0 && ehline2_3.size() > 0) {
		distMatrix3 = Eigen::MatrixXf::Zero(ehline1_3.rows(), ehline2_3.rows());
		for (int i = 0; i < ehline1_3.rows(); ++i) {
			for (int j = 0; j < ehline2_3.rows(); ++j) {
				distMatrix3(i, j) = (ehline1_3.row(i).head(3) - ehline2_3.row(j).head(3)).norm();
			}
		}
	}

	if (ehline1_4.size() > 0 && ehline2_4.size() > 0) {
		distMatrix4 = Eigen::MatrixXf::Zero(ehline1_4.rows(), ehline2_4.rows());
		for (int i = 0; i < ehline1_4.rows(); ++i) {
			for (int j = 0; j < ehline2_4.rows(); ++j) {
				distMatrix4(i, j) = (ehline1_4.row(i).head(3) - ehline2_4.row(j).head(3)).norm();
			}
		}
	}

	std::cout << "distMatrix1:\n" << distMatrix1 << std::endl;
	std::cout << "distMatrix2:\n" << distMatrix2 << std::endl;
	std::cout << "distMatrix3:\n" << distMatrix3 << std::endl;
	std::cout << "distMatrix4:\n" << distMatrix4 << std::endl;

	std::vector<std::pair<int, int>> row_col1, row_col2, row_col3, row_col4;
	std::vector<std::vector<int>> match_lines;
	std::vector<int> match_line;
	std::vector<std::vector<double>> det_theta_values(rows1, std::vector<double>(rows2, std::numeric_limits<double>::max()));
	std::vector<std::vector<double>> det_r_values(rows1, std::vector<double>(rows2, 0));
	std::vector<bool> matched(rows2, false);

	if (distMatrix1.rows()) row_col1 = minIndicesInEachRow(distMatrix1);
	if (distMatrix2.rows()) row_col2 = minIndicesInEachRow(distMatrix2);
	if (distMatrix3.rows()) row_col3 = minIndicesInEachRow(distMatrix3);
	if (distMatrix4.rows()) row_col4 = minIndicesInEachRow(distMatrix4);

	std::cout << "row_col1: ";
	printVector(row_col1);

	std::cout << "row_col2: ";
	printVector(row_col2);

	std::cout << "row_col3: ";
	printVector(row_col3);

	std::cout << "row_col4: ";
	printVector(row_col4);
	int dis_thresold = 30;
	if (!row_col1.empty()) {
		for (int i=0;i<row_col1.size();i++)	{
			if (distMatrix1(row_col1[i].first, row_col1[i].second)< dis_thresold){
			match_line.push_back(ehline1_1(row_col1[i].first,3));
			match_line.push_back(ehline2_1(row_col1[i].second, 3));
			match_line.push_back(distMatrix1(row_col1[i].first, row_col1[i].second));
			match_lines.push_back(match_line);
			match_line.clear();
			}
		}
	}
	if (!row_col2.empty()) {		
		for (int i = 0; i < row_col2.size(); i++) {
			if (distMatrix2(row_col2[i].first, row_col2[i].second) < dis_thresold) {
				match_line.push_back(ehline1_2(row_col2[i].first, 3));
				match_line.push_back(ehline2_2(row_col2[i].second, 3));
				match_line.push_back(distMatrix2(row_col2[i].first, row_col2[i].second));
				match_lines.push_back(match_line);
				match_line.clear();
			}
		}
	}
	if (!row_col3.empty()) {
		for (int i = 0; i < row_col3.size(); i++) {
			if (distMatrix3(row_col3[i].first, row_col3[i].second) < dis_thresold) {
				match_line.push_back(ehline1_3(row_col3[i].first, 3));
				match_line.push_back(ehline2_3(row_col3[i].second, 3));
				match_line.push_back(distMatrix3(row_col3[i].first, row_col3[i].second));
				match_lines.push_back(match_line);
				match_line.clear();
			}
		}
	}
	if (!row_col4.empty()) {
		for (int i = 0; i < row_col4.size(); i++) {
			if (distMatrix4(row_col4[i].first, row_col4[i].second) < dis_thresold) {
				match_line.push_back(ehline1_4(row_col4[i].first, 3));
				match_line.push_back(ehline2_4(row_col4[i].second, 3));
				match_line.push_back(distMatrix4(row_col4[i].first, row_col4[i].second));
				match_lines.push_back(match_line);
				match_line.clear();
			}
		}
	}


	std::cout << "Matched lines:" << std::endl;
	for (const auto& match : match_lines) {
		std::cout << "Line1 index: " << match[0] << ", Line2 index: " << match[1] << ", det_r: " << match[2] << std::endl;
	}

	// 7. 使用 match_lines 的前两列作为索引，提取相应的匹配线段并分别显示它们
	Mat matchedLines1 = image1.clone();
	Mat matchedLines2 = image2.clone();

	// 在 image1 上绘制匹配线段
	for (size_t i = 0; i < match_lines.size(); i++) {
		int line1_index = match_lines[i][0];
		Point pt1(lines1[line1_index][0], lines1[line1_index][1]);
		Point pt2(lines1[line1_index][2], lines1[line1_index][3]);
		cv::line(matchedLines1, pt1, pt2, Scalar(0, 0, 255), 2);

		// 在线段中点添加数字标签
		Point midpoint((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2);
		cv::putText(matchedLines1, std::to_string(i + 1), midpoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

		//// 添加象限标签
		//int quadrant = (ehline1(line1_index, 2) == 1 ? (ehline1(line1_index, 3) == 1 ? 1 : 4) : (ehline1(line1_index, 3) == 1 ? 2 : 3));
		//cv::putText(matchedLines1, "Q" + std::to_string(quadrant), midpoint - Point(0, 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	}

	// 在 image2 上绘制匹配线段
	for (size_t i = 0; i < match_lines.size(); i++) {
		int line2_index = match_lines[i][1];
		Point pt1(lines2[line2_index][0], lines2[line2_index][1]);
		Point pt2(lines2[line2_index][2], lines2[line2_index][3]);
		cv::line(matchedLines2, pt1, pt2, Scalar(0, 255, 0), 2);

		// 在线段中点添加数字标签
		Point midpoint((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2);
		cv::putText(matchedLines2, std::to_string(i + 1), midpoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);

		//// 添加象限标签
		//int quadrant = (ehline2(line2_index, 2) == 1 ? (ehline2(line2_index, 3) == 1 ? 1 : 4) : (ehline2(line2_index, 3) == 1 ? 2 : 3));
		//cv::putText(matchedLines2, "Q" + std::to_string(quadrant), midpoint - Point(0, 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
	}

	// 显示匹配的线段
	cv::imshow("Matched Lines 1", matchedLines1);
	cv::imshow("Matched Lines 2", matchedLines2);


	// 等待用户按键并关闭窗口
	cv::waitKey(0);
	cv::destroyAllWindows();



	return 0;
}
