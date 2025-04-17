#pragma once

#include "imgproc.hpp" // Include imgproc for SimilarityTransform, etc.
#include <vector>
#include <string>
#include <HalideBuffer.h> // Use HalideBuffer.h for Buffer<>
#include <opencv2/opencv.hpp> // For cv::Mat

// Forward declare if necessary
// struct SimilarityTransform;

struct VideoAlignerParams {
    /*
        Enable initialization from phase correlation.
        This can be useful for handling fast camera pans.
        Otherwise it's unlikely to be useful.
    */
    bool phase_correlate = true;           // Seed IC with phase correlation
    double phase_correlate_threshold = 0.55; // PhaseCorr response threshold

    /*
        ICA iteration termination threshold (max corner displacement in pixels).
        There is a sweet spot for this threshold.
        Too low: Will iterate too many times, potentially accumulating errors.
        Too high: Will iterate too few times, potentially causing visual errors or divergence.
    */
    double threshold = 0.18;                // ICA convergence (px)

    /*
        The fraction of gradient maxima points (per tile) with the smallest
        warp difference to select for the Inverse Compositional Algorithm (ICA).
        Too small: May not have enough constraints for stable alignment.
        Too large: May include outlier points, hindering convergence.
    */
    float smallest_fraction = 0.25f;        // keep best‑25 % pixels

    /*
        The maximum number of ICA iterations per pyramid level before giving up.
    */
    int max_iters = 32;

    /*
        The minimum width or height for the smallest level in the image pyramid.
        Stops generating pyramid levels smaller than this.
    */
    int pyramid_min_width = 20;
    int pyramid_min_height = 20;

    /*
        Maximum allowed displacement of any image corner after convergence at a
        pyramid level (measured in pixels at that level's resolution).
        Helps reject unstable solutions or large, sudden jumps.
    */
    double max_displacement = 12.0;
};

/*
    Video Aligner Class using Sparse Inverse Compositional method.

    Algorithm Overview:
    (1) Maintain alternating keyframe/non-keyframe buffers.
    (2) Optionally use phase correlation for initial translation guess.
    (3) Compute image pyramids for both frames.
    (4) Precompute gradients (GradXY) and gradient maxima locations (GradArgMax)
        within tiles for the keyframe pyramid levels.
    (5) Precompute sparse Jacobians (SparseJacobian) for the keyframe based on maxima locations.
    (6) Iterate through pyramid levels (coarse to fine):
        a. Calculate warp difference (SparseWarpDiff) between template and keyframe
           at gradient maxima locations using the current transform estimate.
        b. Select a subset of points with the smallest difference.
        c. Compute Hessian matrix (sum JᵀJ) from the selected Jacobians.
        d. Perform Inverse Compositional Algorithm (ICA) iterations:
            i. Calculate warp update parameters using SparseICA (Sum Jᵀ * residual).
            ii. Solve for delta transform: dt = H⁻¹ * Sum(Jᵀ * residual).
            iii. Update overall transform: T = compose(delta_T, T).
            iv. Check for convergence (small displacement).
        e. Check total displacement at level end against max_displacement.
        f. Scale translation parameters (TX, TY) when moving to finer level.
    (7) If necessary, invert the final transform.
    (8) Return the final center-based SimilarityTransform.
*/
class VideoAligner
{
public:
    // Aligns the input frame to the previous one (or keyframe).
    // Returns false if alignment fails (tracking lost, kernel error, divergence).
    // The computed 'transform' maps points from the *previous* frame's coordinate system
    // to the *current* frame's coordinate system, using center-based rotation.
    bool AlignNextFrame(
        const cv::Mat& frame,            // Input frame (e.g., BGR)
        SimilarityTransform& transform,  // Output: computed transform
        const VideoAlignerParams& params = VideoAlignerParams()); // Algorithm parameters

protected:
    // Frame Buffering & Indexing
    int CurrFrameIndex = 0;        // Index (0 or 1) of the buffer holding the current frame
    int PrevFrameIndex = 1;        // Index (0 or 1) of the buffer holding the previous frame
    int FramesAccumulated = 0;     // Counter for initial frames
    const int KeyframeIndex = 1;     // Designated index for the keyframe buffer
    const int NonKeyframeIndex = 0; // Designated index for the non-keyframe buffer

    // Pyramid Properties
    int PyramidLevels = -1;        // Number of levels computed
    const int PhaseLevel = 2;      // Pyramid level used for phase correlation (if enabled)

    // Image Dimensions & Buffers
    int LastWidth = -1, LastHeight = -1; // Dimensions of the base level images
    cv::Mat GrayInput[2];                // Grayscale input images (base level)
    // Image pyramid buffers (ScalePyramid[frame_idx][level])
    std::vector<Halide::Runtime::Buffer<uint8_t>> ScalePyramid[2];

    // Keyframe Precomputed Data (per pyramid level)
    std::vector<Halide::Runtime::Buffer<float>> KeyframeGradX; // Gradient X images
    std::vector<Halide::Runtime::Buffer<float>> KeyframeGradY; // Gradient Y images
    std::vector<Halide::Runtime::Buffer<uint16_t>> KeyframeArgMaxX; // Max grad X locations (tile_x, tile_y, coord)
    std::vector<Halide::Runtime::Buffer<uint16_t>> KeyframeArgMaxY; // Max grad Y locations (tile_x, tile_y, coord)
    std::vector<int> KeyframeTileSize; // Tile size used for ArgMax at each level
    std::vector<Halide::Runtime::Buffer<float>> KeyframeJacobianX; // Jacobian components (tile_x, tile_y, param) for X grads
    std::vector<Halide::Runtime::Buffer<float>> KeyframeJacobianY; // Jacobian components (tile_x, tile_y, param) for Y grads

    // Phase Correlation Data
    cv::Mat PhaseImage[2]; // Images for phase correlation (CV_32F)

    // Data Structures for Point Selection (per pyramid level, reused)
    struct DeltaPixel {
        uint16_t abs_delta; // Absolute difference value
        uint16_t tile_x;    // Tile index X
        uint16_t tile_y;    // Tile index Y
        // No need to store pixel coords here, can retrieve from KeyframeArgMax

        // Comparison operator for sorting/nth_element based on delta
        bool operator<(const DeltaPixel& other) const {
            return abs_delta < other.abs_delta;
        }
    };
    std::vector<DeltaPixel> DeltaPixelsX; // Differences calculated using KeyframeArgMaxX points
    std::vector<DeltaPixel> DeltaPixelsY; // Differences calculated using KeyframeArgMaxY points

    // Buffers for Selected Points and Jacobians (per pyramid level, resized as needed)
    std::vector<Halide::Runtime::Buffer<uint16_t>> SelectedPixelsX; // Coords of selected X points (N_x, coord_xy)
    std::vector<Halide::Runtime::Buffer<uint16_t>> SelectedPixelsY; // Coords of selected Y points (N_y, coord_xy)
    std::vector<Halide::Runtime::Buffer<float>> SelectedJacobianX; // Jacobians of selected X points (N_x, param_idx)
    std::vector<Halide::Runtime::Buffer<float>> SelectedJacobianY; // Jacobians of selected Y points (N_y, param_idx)

    // Buffers for Warp Difference Calculation (per pyramid level)
    std::vector<Halide::Runtime::Buffer<uint16_t>> WarpDiffX; // Differences at ArgMaxX locations (tile_x, tile_y)
    std::vector<Halide::Runtime::Buffer<uint16_t>> WarpDiffY; // Differences at ArgMaxY locations (tile_x, tile_y)

    // Buffer for ICA Result (reused)
    Halide::Runtime::Buffer<double> IcaResult; // 4-element vector from SparseICA

    // --- Internal Helper Methods ---

    // Computes image pyramid and converts base level to grayscale.
    // Returns false if it's the first frame (needs one more).
    bool ComputePyramid(const cv::Mat& inputFrame, const VideoAlignerParams& params);

    // Computes gradients, gradient maxima, and Jacobians for the keyframe pyramid.
    // Returns false on failure.
    bool ComputeKeyFrame();
};
