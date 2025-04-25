#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

// Compute the median of a vector<double>
static double median(std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    size_t n = v.size()/2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    double med = v[n];
    if (v.size()%2==0) {
        std::nth_element(v.begin(), v.begin()+n-1, v.end());
        med = 0.5*(med + v[n-1]);
    }
    return med;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " video1 [video2 ...]\n";
        return 1;
    }

    for (int idx = 1; idx < argc; ++idx) {
        std::string path = argv[idx];
        cv::VideoCapture cap(path);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video: " << path << "\n";
            continue;
        }

        cv::Mat prev, prevGray;
        if (!cap.read(prev)) {
            std::cerr << "Empty video: " << path << "\n";
            continue;
        }
        cv::cvtColor(prev, prevGray, cv::COLOR_BGR2GRAY);

        std::vector<double> frameMeds;

        cv::Mat frame, gray;
        while (cap.read(frame)) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            cv::Mat flow;
            cv::calcOpticalFlowFarneback(prevGray, gray, flow,
                                         0.5, 3, 15, 3, 5, 1.2, 0);

            std::vector<cv::Mat> flows(2);
            cv::split(flow, flows);
            cv::Mat mag;
            cv::magnitude(flows[0], flows[1], mag);

            // flatten to 1‑D vector<float>
            mag = mag.reshape(1,1);
            std::vector<float> mf; mag.copyTo(mf);
            if (!mf.empty()) {
                size_t n = mf.size()/2;
                std::nth_element(mf.begin(), mf.begin()+n, mf.end());
                frameMeds.push_back(mf[n]);
            }

            prevGray = gray;
        }

        double medJitter = median(frameMeds);
        std::cout << path << "\tmedian_jitter_px=" << medJitter << "\n";
    }

    return 0;
}
