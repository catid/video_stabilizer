#pragma once

#include "imgproc.hpp"

struct VideoAlignerParams {
    /*
        Enable initialization from phase correlation.
        This can be useful for handling fast camera pans.
        Otherwise it's unlikely to be useful.
    */
    bool phase_correlate = false;
    double phase_correlate_threshold = 0.5;

    /*
        There is a sweet spot for this threshold.
        Too low: Will iterate too many times, accumulating errors until it diverges.
        Too high: Will iterate too few times, creating visual errors and/or diverging more.
    */
    double threshold = 0.03;

    /*
        The smallest fraction of the image to use for the sparse set.
        Too small: Will not find enough keypoints to do a good alignment.
        Too large: Will accept too many false positives, and may not converge.
    */
    float smallest_fraction = 0.5f;

    /*
        The maximum number of iterations to run before giving up at each level.
    */
    int max_iters = 64;

    /*
        The minimum width or height of the image pyramid (smallest layer size).
    */
    int pyramid_min_width = 20;
    int pyramid_min_height = 20;
};

/*
    (1) Keyframe every other frame
    (2) Use cv::phaseCorrelate to find the initial x/y shift guess
    (3) Compute image pyramids
    (4) Precompute x/y gradient images and Hessians
    (5) Lucas-Kanade iterations
    (6) Flip sign based on keyframe ahead or behind and return result
*/
class VideoAligner
{
public:
    // Returns false if track is lost or kernel fails
    bool AlignNextFrame(
        const cv::Mat& frame,
        SimilarityTransform& transform,
        const VideoAlignerParams& params = VideoAlignerParams());

protected:
    // Alternating frame indexing
    int CurrFrameIndex = 0;
    int PrevFrameIndex = 1;
    int FramesAccumulated = 0;
    const int KeyframeIndex = 1;
    const int NonKeyframeIndex = 0;

    int PyramidLevels = -1;
    const int PhaseLevel = 2;

    int LastWidth = -1, LastHeight = -1;
    cv::Mat GrayInput[2];
    std::vector<Halide::Runtime::Buffer<uint8_t>> ScalePyramid[2];

    std::vector<Halide::Runtime::Buffer<float>> KeyframeGradX, KeyframeGradY;

    std::vector<Halide::Runtime::Buffer<uint16_t>> KeyframeArgMaxX, KeyframeArgMaxY;
    std::vector<int> KeyframeTileSize;

    std::vector<Halide::Runtime::Buffer<float>> KeyframeJacobianX, KeyframeJacobianY;

    cv::Mat PhaseImage[2];

    struct DeltaPixel {
        uint16_t abs_delta, tile_x, tile_y;
    };
    std::vector<DeltaPixel> DeltaPixelsX, DeltaPixelsY;

    std::vector<Halide::Runtime::Buffer<uint16_t>> SelectedPixelsX, SelectedPixelsY;
    std::vector<Halide::Runtime::Buffer<float>> SelectedJacobianX, SelectedJacobianY;

    std::vector<Halide::Runtime::Buffer<uint16_t>> WarpDiffX, WarpDiffY;

    Halide::Runtime::Buffer<double> IcaResult;

    // Returns false if this is the first frame
    bool ComputePyramid(const cv::Mat& inputFrame, const VideoAlignerParams& params);
    bool ComputeKeyFrame();
};
