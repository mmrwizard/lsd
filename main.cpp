//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/ximgproc/fast_line_detector.hpp>
//#include <opencv2/opencv.hpp>
//#include <cmath>
//#include <vector>
//
//#include <iostream>
//#include <cairo.h>
//#include <cairo-pdf.h>
//#include <Eigen/Dense>
//
//using namespace std;
//using namespace cv;
//using namespace cv::ximgproc;
//
//// Convert the input line segments to Hough space
//void lineToHoughSpace(const std::vector<cv::Vec4f>& lines, const cv::Size& imageSize, cv::Mat& houghSpace) {
//	int max_rho = static_cast<int>(sqrt(imageSize.width * imageSize.width + imageSize.height * imageSize.height));
//	int rho_bins = max_rho * 2 + 1;
//	int theta_bins = 180;
//	houghSpace = cv::Mat::zeros(rho_bins, theta_bins, CV_32SC1);
//
//	for (const auto& line : lines) {
//		cv::Point2f point1(line[0], line[1]);
//		cv::Point2f point2(line[2], line[3]);
//
//		// Iterate through the line segment points
//		for (double t = 0; t <= 1; t += 0.01) {
//			int x = static_cast<int>(point1.x * t + point2.x * (1 - t));
//			int y = static_cast<int>(point1.y * t + point2.y * (1 - t));
//
//			// Iterate through theta values
//			for (int theta = 0; theta < theta_bins; ++theta) {
//				double theta_rad = (CV_PI * theta) / 180.0;
//				int rho = static_cast<int>(x * cos(theta_rad) + y * sin(theta_rad)) + max_rho;
//
//				// Increment the Hough space accumulator
//				houghSpace.at<int>(rho, theta) += 1;
//			}
//		}
//	}
//}
//
//vector<Vec4f> detectLines(const Mat& inputImage, double _length_threshold) {
//	// 1.1 ת��ͼ��Ϊ�Ҷ�ͼ
//	Mat grayImage;
//	cvtColor(inputImage, grayImage, COLOR_BGR2GRAY);
//
//	// 2. ����FastLineDetector�����ֱ��
//	
//	double _distance_threshold = 1.41421356;
//	double _canny_low_thresh = 50.0;
//	double _canny_high_thresh = 100.0;
//	int _canny_aperture_size = 3;
//	bool _do_merge = true;
//	Ptr<FastLineDetector> fld = createFastLineDetector(_length_threshold, _distance_threshold, _canny_low_thresh, _canny_high_thresh, _canny_aperture_size, _do_merge);
//	vector<Vec4f> lines;
//	fld->detect(grayImage, lines);
//
//	return lines;
//}
//
//
//int main(int argc, char** argv){
//	if (argc != 3)
//	{
//		cout << "Usage: ./lsd_example <path_to_image>" << endl;
//		return -1;
//	}
//
//	// 1. ����ͼ��
//	Mat image1 = imread(argv[1], IMREAD_COLOR);
//	Mat image2 = imread(argv[2], IMREAD_COLOR);
//	int img1_height = image1.rows, img1_width = image1.cols;
//	int img2_height = image2.rows, img2_width = image2.cols;
//
//	if (image1.empty()||image2.empty())
//	{
//		cout << "Could not open or find the image" << endl;
//		return -1;
//	}
//
//	vector<Vec4f> lines1;
//	vector<Vec4f> lines2;
//
//	//2.���ֱ��
//	double _length_threshold = 30;
//	lines1 = detectLines(image1, _length_threshold);
//	lines2 = detectLines(image2, _length_threshold);
//
//	int drwaLine = 0;
//	if(drwaLine)
//	{
//		// 3. ����һ���հ׵�ͼ�����ڻ��Ƽ�⵽��ֱ��
//		Mat lineImage1 = Mat::zeros(image1.size(), image1.type());
//		Mat lineImage2 = Mat::zeros(image2.size(), image2.type());
//
//		// 4. �ڿհ�ͼ���ϻ��Ƽ�⵽��ֱ��
//		for (vector<Vec4f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it)
//		{
//			Point pt1((*it)[0], (*it)[1]);
//			Point pt2((*it)[2], (*it)[3]);
//			cv::line(lineImage1, pt1, pt2, Scalar(0, 255, 0), 2);
//		}
//		for (vector<Vec4f>::const_iterator it = lines2.begin(); it != lines2.end(); ++it)
//		{
//			Point pt1((*it)[0], (*it)[1]);
//			Point pt2((*it)[2], (*it)[3]);
//			cv::line(lineImage2, pt1, pt2, Scalar(255, 255, 255), 2);
//		}
//		cv::imshow("Matched Lines 1", lineImage1);
//		cv::imshow("Matched Lines 2", lineImage2);
//	}
//	
//
//
//// 5. ��line1��line2�е�ÿ��ֱ�߶εĲ������������h��r, theta��
//	vector<Vec2f> h_lines1,h_lines2;
//	for (const auto& line : lines1) {
//		float r = sqrt(pow(line[0] - line[2], 2) + pow(line[1] - line[3], 2));
//		float theta = atan2(line[3] - line[1], line[2] - line[0]);
//		h_lines1.push_back(Vec2f(r, theta));
//	}
//	for (const auto& line : lines2) {
//		float r = sqrt(pow(line[0] - line[2], 2) + pow(line[1] - line[3], 2));
//		float theta = atan2(line[3] - line[1], line[2] - line[0]);
//		h_lines2.push_back(Vec2f(r, theta));
//	}
//
//	// �½��׵׿հ�ͼ��png
//	int png = 0, pdf = 0;
//	if(png)
//	{
//		int width = 1000;
//		int height = 1000;
//		cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
//		cairo_t* cr = cairo_create(surface);
//
//		// ����������ɫΪ��ɫ�����������ᣩ
//		cairo_set_source_rgb(cr, 0, 0, 0);
//
//		// ����x��
//		cairo_move_to(cr, 0, height / 2);
//		cairo_line_to(cr, width, height / 2);
//
//		// ����y��
//		cairo_move_to(cr, width / 2, 0);
//		cairo_line_to(cr, width / 2, height);
//
//		// ����������
//		cairo_stroke(cr);
//
//		// ʹ��Cairo�⽫ÿ����r, theta����Ӧ�ĵ㣨r*cos(theta), r*sin(theta)���루0, 0�����ӳ�ʸ��ֱ�߶�
//
//	// ����������ɫΪ��ɫ������h_lines1��
//		cairo_set_source_rgb(cr, 1, 0, 0);
//		for (const auto& h_line : h_lines1) {
//			float r = h_line[0];
//			float theta = h_line[1];
//			float x = r * cos(theta);
//			float y = r * sin(theta);
//
//			cairo_move_to(cr, width / 2, height / 2);
//			cairo_line_to(cr, x + width / 2, y + height / 2);
//		}
//		// ����ֱ��
//		cairo_stroke(cr);
//		cairo_surface_write_to_png(surface, "output1.png");
//
//		//// ����������ɫΪ��ɫ������h_lines2��
//		cairo_set_source_rgb(cr, 0, 1, 0);
//		for (const auto& h_line : h_lines2) {
//			float r = h_line[0];
//			float theta = h_line[1];
//			float x = r * cos(theta);
//			float y = r * sin(theta);
//
//			cairo_move_to(cr, width / 2, height / 2);
//			cairo_line_to(cr, x + width / 2, y + height / 2);
//		}
//		// ����ֱ��
//		cairo_stroke(cr);
//
//		// ����ͼ���ļ�
//		cairo_surface_write_to_png(surface, "output.png");
//
//		// �ͷ���Դ
//		cairo_destroy(cr);
//		cairo_surface_destroy(surface);
//	}
//
//	// �½��׵׿հ�ͼ��pdf
//	if(pdf){
//		int width = 1000;
//		int height = 1000;
//		cairo_surface_t* surface = cairo_pdf_surface_create("output.pdf", width, height);
//		cairo_t* cr = cairo_create(surface);
//
//		// ����������ɫΪ��ɫ�����������ᣩ
//		cairo_set_source_rgb(cr, 0, 0, 0);
//
//		// ����x��
//		cairo_move_to(cr, 0, height / 2);
//		cairo_line_to(cr, width, height / 2);
//
//		// ����y��
//		cairo_move_to(cr, width / 2, 0);
//		cairo_line_to(cr, width / 2, height);
//
//		// ����������
//		cairo_stroke(cr);
//
//		// ʹ��Cairo�⽫ÿ����r, theta����Ӧ�ĵ㣨r*cos(theta), r*sin(theta)���루0, 0�����ӳ�ʸ��ֱ�߶�
//
//		// ����������ɫΪ��ɫ������h_lines2��
//		cairo_set_source_rgb(cr, 0, 1, 0);
//		for (const auto& h_line : h_lines2) {
//			float r = h_line[0];
//			float theta = h_line[1];
//			float x = r * cos(theta);
//			float y = r * sin(theta);
//
//			cairo_move_to(cr, width / 2, height / 2);
//			cairo_line_to(cr, x + width / 2, y + height / 2);
//		}
//		// ����ֱ��
//		cairo_stroke(cr);
//
//		// ����������ɫΪ��ɫ������h_lines1��
//		cairo_set_source_rgb(cr, 1, 0, 0);
//		for (const auto& h_line : h_lines1) {
//			float r = h_line[0];
//			float theta = h_line[1];
//			float x = r * cos(theta);
//			float y = r * sin(theta);
//
//			cairo_move_to(cr, width / 2, height / 2);
//			cairo_line_to(cr, x + width / 2, y + height / 2);
//		}
//		// ����ֱ��
//		cairo_stroke(cr);
//
//		// ����PDFд��
//		cairo_show_page(cr);
//
//		// �ͷ���Դ
//		cairo_destroy(cr);
//		cairo_surface_destroy(surface);
//	}
//
//
//	//6. ����ֱ���������������ɣ�r,theta,d(0,1),d)���
//	int rows1 = static_cast<int>(h_lines1.size());
//	int rows2 = static_cast<int>(h_lines2.size());
//
//	Eigen::MatrixXf ehline1(rows1, 5), ehline2(rows2, 5);
//	for (int i = 0; i < rows1; ++i) {
//		ehline1(i, 0) = h_lines1[i][0];
//		ehline1(i, 1) = h_lines1[i][1];
//
//		// �����е�����
//		float mid_x = (lines1[i](0)+lines1[i](2))/2-img1_width/2;
//		float mid_y = (lines1[i](1) + lines1[i](3))/2 -img1_height/2;
//
//		// �ж����޲��洢���ȱ���
//		if (mid_x >= 0 && mid_y >= 0) { // ��һ����
//			ehline1(i, 2) = 1;
//			ehline1(i, 3) = 1;
//		}
//		else if (mid_x < 0 && mid_y >= 0) { // �ڶ�����
//			ehline1(i, 2) = -1;
//			ehline1(i, 3) = 1;
//		}
//		else if (mid_x < 0 && mid_y < 0) { // ��������
//			ehline1(i, 2) = -1;
//			ehline1(i, 3) = -1;
//		}
//		else { // ��������
//			ehline1(i, 2) = 1;
//			ehline1(i, 3) = -1;
//		}
//
//		// ������벢�洢
//		ehline1(i, 4) = sqrt(mid_x * mid_x + mid_y * mid_y);
//	}
//
//	for (int i = 0; i < rows2; ++i) {
//		ehline2(i, 0) = h_lines2[i][0];
//		ehline2(i, 1) = h_lines2[i][1];
//
//		// �����е�����
//		float mid_x = (lines2[i](0) + lines2[i](2))/2 - img2_width / 2;
//		float mid_y = (lines2[i](1) + lines2[i](3))/2 - img2_height / 2;
//		
//		// �ж����޲��洢���ȱ���
//		if (mid_x >= 0 && mid_y >= 0) { // ��һ����
//			ehline2(i, 2) = 1;
//			ehline2(i, 3) = 1;
//		}
//		else if (mid_x < 0 && mid_y >= 0) { // �ڶ�����
//			ehline2(i, 2) = -1;
//			ehline2(i, 3) = 1;
//		}
//		else if (mid_x < 0 && mid_y < 0) { // ��������
//			ehline2(i, 2) = -1;
//			ehline2(i, 3) = -1;
//		}
//		else { // ��������
//			ehline2(i, 2) = 1;
//			ehline2(i, 3) = -1;
//		}
//
//		// ������벢�洢
//		ehline2(i, 4) = sqrt(mid_x * mid_x + mid_y * mid_y);
//	}
//
//	//export to csv file 
//
//	std::ofstream file1("ehline1.csv");
//	file1 << "r,theta,x,y,d\n";
//	for (int i = 0; i < ehline1.rows(); ++i) {
//		for (int j = 0; j < ehline1.cols(); ++j) {
//			file1 << ehline1(i, j);
//			if (j < ehline1.cols() - 1) {
//				file1 << ",";
//			}
//		}
//		file1 << "\n";
//	}
//	file1.close();
//
//	std::ofstream file2("ehline2.csv");
//	file2 << "r,theta,x,y,d\n";
//	for (int i = 0; i < ehline2.rows(); ++i) {
//		for (int j = 0; j < ehline2.cols(); ++j) {
//			file2 << ehline2(i, j);
//			if (j < ehline2.cols() - 1) {
//				file2 << ",";
//			}
//		}
//		file2 << "\n";
//	}
//	file2.close();
//
//	/*std::cout << ehline1.size() << std::endl;
//	std::cout << ehline2.size() << std::endl;*/
//
//	//����r��theta�Լ��߶��е�ľ���Ĳ�ֵ�ж���ֱ���Ƿ�ƥ�䣬ͬʱ�ӽ���0���ʾ��ѡƥ���
//	std::vector<std::vector<int>> match_lines;
//	std::vector<int> match_line;
//	std::vector<std::vector<double>> det_theta_values(rows1, std::vector<double>(rows2, std::numeric_limits<double>::max()));
//	std::vector<std::vector<double>> det_r_values(rows1, std::vector<double>(rows2, 0));
//
//	std::vector<bool> matched(rows2, false);
//
//	for (int i = 0; i < rows1; i++) {
//		double min_det_theta = std::numeric_limits<double>::max();
//		int min_det_theta_index = -1;
//
//		for (int j = 0; j < rows2; j++) {
//			// �ж������Ƿ���ͬһ���ޣ������������һ��ѭ��
//			if (ehline1(i, 2) != ehline2(j, 2) || ehline1(i, 3) != ehline2(j, 3)) {
//				continue;
//			}
//
//			// �ж��߶��Ƿ��ѱ�ƥ�䣬����ѱ�ƥ�䣬�������һ��ѭ��
//			if (matched[j]) {
//				continue;
//			}
//
//			double det_r = std::abs(ehline1(i, 0) - ehline2(j, 0));
//			double det_theta = std::abs(ehline1(i, 1) - ehline2(j, 1));
//
//			det_theta_values[i][j] = det_theta;
//			det_r_values[i][j] = det_r;
//
//			if (det_theta < min_det_theta) {
//				min_det_theta = det_theta;
//				min_det_theta_index = j;
//			}
//		}
//
//		if (min_det_theta_index != -1/*&& det_r_values[i][min_det_theta_index]<50*/) {
//			match_line.push_back(i);
//			match_line.push_back(min_det_theta_index);
//			match_line.push_back(det_r_values[i][min_det_theta_index]);
//			match_lines.push_back(match_line);
//			match_line.clear();
//
//			// ��Ǹ��߶��ѱ�ƥ��
//			matched[min_det_theta_index] = true;
//		}
//	}
//
//
//
//	std::cout << "Matched lines:" << std::endl;
//	for (const auto& match : match_lines) {
//		std::cout << "Line1 index: " << match[0] << ", Line2 index: " << match[1] << ", det_r: " << match[2] << std::endl;
//	}
//
//
//	// 7. ʹ�� match_lines ��ǰ������Ϊ��������ȡ��Ӧ��ƥ���߶β��ֱ���ʾ����
//	Mat matchedLines1 = image1.clone();
//	Mat matchedLines2 = image2.clone();
//
//	// �� image1 �ϻ���ƥ���߶�
//	for (size_t i = 0; i < match_lines.size(); i++) {
//		int line1_index = match_lines[i][0];
//		Point pt1(lines1[line1_index][0], lines1[line1_index][1]);
//		Point pt2(lines1[line1_index][2], lines1[line1_index][3]);
//		cv::line(matchedLines1, pt1, pt2, Scalar(0, 0, 255), 2);
//
//		// ���߶��е�������ֱ�ǩ
//		Point midpoint((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2);
//		cv::putText(matchedLines1, std::to_string(i + 1), midpoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
//
//		//// ������ޱ�ǩ
//		//int quadrant = (ehline1(line1_index, 2) == 1 ? (ehline1(line1_index, 3) == 1 ? 1 : 4) : (ehline1(line1_index, 3) == 1 ? 2 : 3));
//		//cv::putText(matchedLines1, "Q" + std::to_string(quadrant), midpoint - Point(0, 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
//	}
//
//	// �� image2 �ϻ���ƥ���߶�
//	for (size_t i = 0; i < match_lines.size(); i++) {
//		int line2_index = match_lines[i][1];
//		Point pt1(lines2[line2_index][0], lines2[line2_index][1]);
//		Point pt2(lines2[line2_index][2], lines2[line2_index][3]);
//		cv::line(matchedLines2, pt1, pt2, Scalar(0, 255, 0), 2);
//
//		// ���߶��е�������ֱ�ǩ
//		Point midpoint((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2);
//		cv::putText(matchedLines2, std::to_string(i + 1), midpoint, FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
//
//		//// ������ޱ�ǩ
//		//int quadrant = (ehline2(line2_index, 2) == 1 ? (ehline2(line2_index, 3) == 1 ? 1 : 4) : (ehline2(line2_index, 3) == 1 ? 2 : 3));
//		//cv::putText(matchedLines2, "Q" + std::to_string(quadrant), midpoint - Point(0, 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
//	}
//
//	// ��ʾƥ����߶�
//	cv::imshow("Matched Lines 1", matchedLines1);
//	cv::imshow("Matched Lines 2", matchedLines2);
//
//
//	// �ȴ��û��������رմ���
//	cv::waitKey(0);
//	cv::destroyAllWindows();
//
//
//
//	return 0;
//}
