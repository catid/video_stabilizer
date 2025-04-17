#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <deque>

#include "alignment.hpp"
#include "smoother.hpp"

struct VideoStabilizerParams {
    VideoAlignerParams aligner;

    // Lag: number of frames to delay before smoothing
    // Tuned defaults (via grid search):
    int lag = 10;                // frames of latency (lagBehind)
    int smoother_memory = 30;    // additional history (lagAhead) used by smoother
    double lambda = 2.0;         // smoothness strength

    // Enable smoother: if false, just use the aligner
    bool enable_smoother = true;

    // Crop pixels: if > 0, crop the output image by this amount
    int crop_pixels = 16;        // auto‑cropping margin

    // Displacement thresholds for decay and full reset
    double min_disp = 48.0, max_disp = 160.0; // pixels
    double min_decay = 0.90, max_decay = 0.40;
};

class VideoStabilizer {
public:
    VideoStabilizer(const VideoStabilizerParams& params = VideoStabilizerParams());
    /**
     * Process one frame and return the stabilized frame if available;
     * otherwise return an empty cv::Mat until enough frames have arrived.
     */
    cv::Mat processFrame(const cv::Mat& inputFrame);

protected:
    VideoStabilizerParams m_params;
    VideoAligner aligner;
    int m_frameIndex = 0;
    int alignFailures = 0;
    L1SmootherCenter m_l1Smoother;

    // For each frame i, we store a measured transform from frame i-1 to i
    // (i.e. T_i) once alignment is done.
    std::deque<SimilarityTransform> m_measurementBuffer;

    // We also keep around all input frames until it's time to pop them
    std::deque<cv::Mat> m_frameBuffer;

    SimilarityTransform m_accum;
};
