#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "particlefilter.h"

const double skin_R = 120.0, skin_G = 63.0, skin_B = 45.0;
//const double skin_R = 255.0, skin_G = 0.0, skin_B = 0.0;

const double sigma = 10.0;

// 色の類似度を尤度として計算する
double Likelihood(cv::Mat &image, int x, int y)
{
    cv::Vec3b *bgr = image.ptr<cv::Vec3b>(y);
    double B = bgr[x][0];
    double G = bgr[x][1];
    double R = bgr[x][2];
    
    double color_dist = sqrt((R - skin_R) * (R - skin_R)
			     + (G - skin_G) * (G - skin_G)
			     + (B - skin_B) * (B - skin_B));
    
    return 1.0 / (sqrt(2.0 * M_PI) * sigma)
	   * exp(-color_dist * color_dist / (2.0 * sigma * sigma));
}

cv::Scalar convertValueToRGB(double value)
{
    double h = 4.0 - value * 4.0;
    double s = 1.0;
    double v = 1.0;
    int i = static_cast<int>(h);
    double f = h - static_cast<double>(i);
    
    if (!(i & 1)) {
	f = 1.0 - f;
    }
    
    double m = v * (1.0 - s);
    double n = v * (1.0 - s * f);
    int r, g, b;
    switch (i) {
    case 0:
	r = cv::saturate_cast<uchar>(v * 255.0 + 0.5);
	g = cv::saturate_cast<uchar>(n * 255.0 + 0.5);
	b = cv::saturate_cast<uchar>(m * 255.0 + 0.5);
	break;
    case 1:
	r = cv::saturate_cast<uchar>(n * 255.0 + 0.5);
	g = cv::saturate_cast<uchar>(v * 255.0 + 0.5);
	b = cv::saturate_cast<uchar>(m * 255.0 + 0.5);
	break;
    case 2:
	r = cv::saturate_cast<uchar>(m * 255.0 + 0.5);
	g = cv::saturate_cast<uchar>(v * 255.0 + 0.5);
	b = cv::saturate_cast<uchar>(n * 255.0 + 0.5);
	break;
    case 3:
	r = cv::saturate_cast<uchar>(m * 255.0 + 0.5);
	g = cv::saturate_cast<uchar>(n * 255.0 + 0.5);
	b = cv::saturate_cast<uchar>(v * 255.0 + 0.5);
	break;
    case 4:
	r = cv::saturate_cast<uchar>(n * 255.0 + 0.5);
	g = cv::saturate_cast<uchar>(m * 255.0 + 0.5);
	b = cv::saturate_cast<uchar>(v * 255.0 + 0.5);
	break;
    case 5:
	r = cv::saturate_cast<uchar>(v * 255.0 + 0.5);
	g = cv::saturate_cast<uchar>(m * 255.0 + 0.5);
	b = cv::saturate_cast<uchar>(n * 255.0 + 0.5);
	break;
    }
    return cv::Scalar(b, g, r);
}

int main(int argc, char *argv[])
{
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
	std::cerr << "Fail: Camera cannot open." << std::endl;
	return -1;
    }
    
    cv::Mat frame;
    capture >> frame;
    
    cv::namedWindow("Capture", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
    
    int particleNum = 1000;
    
    StateModel *model = new RandomModel(5.0);
    //StateModel *model = new LinearModel(3.0);
    ParticleFilter filter(particleNum, model);
    double **boundary = new double*[model->dimension()];
    for (int i = 0; i < model->dimension(); ++i) {
	boundary[i] = new double[2];
    }
    boundary[0][0] = 0.0;
    boundary[0][1] = frame.cols - 1;
    boundary[1][0] = 0.0;
    boundary[1][1] = frame.rows - 1;
    /*boundary[2][0] = -3.0;
    boundary[2][1] = 3.0;
    boundary[3][0] = -3.0;
    boundary[3][1] = 3.0;*/
    
    // パーティクルの初期化
    filter.initParticles(boundary);
    
    std::vector<cv::Point> pos;
    while (cv::waitKey(1) != 'q') {
	capture >> frame;
	
	// 予測
	filter.predict();
	
	// 尤度推定
	for (int i = 0; i < particleNum; ++i) {
	    int x = static_cast<int>(filter.particles(i, 0));
	    int y = static_cast<int>(filter.particles(i, 1));
	    if (x < 0 || x >= frame.cols || y < 0 || y >= frame.rows) {
		filter.setLikelihood(i, 0.0);
	    } else {
		filter.setLikelihood(i, Likelihood(frame, x, y));
	    }
	}
	
	// 尤度正規化
	filter.normalizeLikelihood();
	
	// パーティクルをプロット
	for (int i = 0; i < particleNum; ++i) {
	    int x = static_cast<int>(filter.particles(i, 0));
	    int y = static_cast<int>(filter.particles(i, 1));
	    cv::circle(frame, cv::Point(x, y), 1, convertValueToRGB(filter.likelihood(i) * 100.0), -1, CV_AA);
	}
	
	// 追跡結果を推定・プロット
	filter.estimate();
	pos.push_back(cv::Point(static_cast<int>(filter.estimateResult(0)), static_cast<int>(filter.estimateResult(1))));
	for (size_t i = 0; i < pos.size() - 1; ++i) {
	    cv::line(frame, pos[i], pos[i + 1], cv::Scalar(0, 0, 255), 0, CV_AA);
	}
	
	cv::imshow("Capture", frame);
	
#ifdef _DEBUG
	if (cv::waitKey(0) == 'q') {
	    break;
	}
#endif
	// フィルタリング・リサンプリング
	filter.filterParticles();
    }
    
    for (int i = 0; i < model->dimension(); ++i) {
	delete[] boundary[i];
    }
    delete[] boundary;
    delete model;
    
    return 0;
}
