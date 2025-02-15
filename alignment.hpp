#pragma once

#include "imgproc.hpp"

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
        bool phase_correlate = false);

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
    bool ComputePyramid(const cv::Mat& inputFrame);
    bool ComputeKeyFrame();
};
