#include "alignment.hpp"
#include "tools.hpp" // Assuming this exists for helper functions if needed
#include "imgproc.hpp" // Make sure imgproc is included for its functions/structs

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>     // For std::abs, std::min, std::max
#include <numeric>   // For std::iota potentially
#include <algorithm> // For std::nth_element, std::min, std::max


// ---- Performance Metrics ---- (Optional, kept from original)
#define ENABLE_PERFORMANCE_METRICS // Comment out to disable

#ifdef ENABLE_PERFORMANCE_METRICS

// Performance measurement utilities
class PerformanceMetrics {
public:
    // Singleton instance getter
    static PerformanceMetrics& getInstance() {
        static PerformanceMetrics instance;
        return instance;
    }

    // Start a timer with a given label
    void startTimer(const std::string& label) {
        timers[label] = std::chrono::high_resolution_clock::now();
    }

    // End a timer, record duration, and update statistics
    double endTimer(const std::string& label) {
        auto end = std::chrono::high_resolution_clock::now();
        auto it = timers.find(label);
        if (it == timers.end()) {
            std::cerr << "Timer '" << label << "' was never started!" << std::endl;
            return 0.0;
        }

        double duration = std::chrono::duration<double, std::milli>(end - it->second).count();

        // Update time-based metrics
        auto& metric = metrics[label]; // Creates if not exists
        if (metric.callCount == 0) { // First call
             metric = {duration, 1, duration, duration};
        } else {
            metric.totalTime += duration;
            metric.callCount++;
            metric.minTime = std::min(metric.minTime, duration);
            metric.maxTime = std::max(metric.maxTime, duration);
        }

        return duration;
    }

    // Log a custom metric (e.g., number of points, condition number)
    void logMetric(const std::string& label, double value) {
         auto& metric = customMetrics[label]; // Creates if not exists
         if (metric.callCount == 0) { // First data point
             metric = {value, 1, value, value}; // Store value in 'totalTime' field for aggregation
         } else {
            metric.totalTime += value; // Aggregate sum
            metric.callCount++;
            metric.minTime = std::min(metric.minTime, value);
            metric.maxTime = std::max(metric.maxTime, value);
         }
    }

    // Print all collected metrics to the console
    void printAllMetrics() {
        std::cout << "\n==== PERFORMANCE METRICS ====\n";
        std::cout << std::left << std::setw(40) << "Function/Label"
                  << std::setw(15) << "Avg Time/Val" // Combined label
                  << std::setw(15) << "Total Time/Sum"
                  << std::setw(15) << "Calls/Count"
                  << std::setw(15) << "Min (ms/Val)"
                  << std::setw(15) << "Max (ms/Val)"
                  << std::endl;
        std::cout << std::string(115, '-') << std::endl;

        // Print function timing metrics
        for (const auto& pair : metrics) {
            const auto& metric = pair.second;
            if (metric.callCount == 0) continue;
            double avgTime = metric.totalTime / metric.callCount;

            std::cout << std::left << std::setw(40) << pair.first
                      << std::fixed << std::setprecision(3)
                      << std::setw(15) << avgTime
                      << std::setw(15) << metric.totalTime
                      << std::setw(15) << metric.callCount
                      << std::setw(15) << metric.minTime
                      << std::setw(15) << metric.maxTime
                      << std::endl;
        }

        // Print custom metrics
        if (!customMetrics.empty()) {
            std::cout << "\n==== CUSTOM METRICS ====\n";
             std::cout << std::left << std::setw(40) << "Label"
                  << std::setw(15) << "Avg Value"
                  << std::setw(15) << "Total Sum"
                  << std::setw(15) << "Count"
                  << std::setw(15) << "Min Value"
                  << std::setw(15) << "Max Value"
                  << std::endl;
            std::cout << std::string(115, '-') << std::endl;
            for (const auto& pair : customMetrics) {
                const auto& metric = pair.second;
                 if (metric.callCount == 0) continue;
                double avgValue = metric.totalTime / metric.callCount; // totalTime holds the sum here

                std::cout << std::left << std::setw(40) << pair.first
                          << std::fixed << std::setprecision(3)
                          << std::setw(15) << avgValue
                          << std::setw(15) << metric.totalTime // Print sum
                          << std::setw(15) << metric.callCount
                          << std::setw(15) << metric.minTime
                          << std::setw(15) << metric.maxTime
                          << std::endl;
            }
        }
         std::cout << "===========================\n" << std::endl;
    }

    // Reset all metrics
    void reset() {
        timers.clear();
        metrics.clear();
        customMetrics.clear();
    }

private:
    // Structure to hold metric statistics
    struct Metric {
        double totalTime = 0.0; // Also used for sum of custom values
        int callCount = 0;
        double minTime = 0.0;   // Also used for min custom value
        double maxTime = 0.0;   // Also used for max custom value
    };

    // Storage for timers and metrics
    std::unordered_map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> timers;
    std::unordered_map<std::string, Metric> metrics;       // For time measurements
    std::unordered_map<std::string, Metric> customMetrics; // For custom values

    // Private constructor/destructor for singleton
    PerformanceMetrics() = default;
    ~PerformanceMetrics() = default;
    // Prevent copying
    PerformanceMetrics(const PerformanceMetrics&) = delete;
    PerformanceMetrics& operator=(const PerformanceMetrics&) = delete;
};

// RAII timer guard for easy function timing
class TimerGuard {
public:
    explicit TimerGuard(const std::string& label) : label_(label), active_(true) {
        PerformanceMetrics::getInstance().startTimer(label_);
    }

    // Non-copyable
    TimerGuard(const TimerGuard&) = delete;
    TimerGuard& operator=(const TimerGuard&) = delete;

     // Move constructor
    TimerGuard(TimerGuard&& other) noexcept : label_(std::move(other.label_)), active_(other.active_) {
        other.active_ = false; // Prevent double-stopping
    }
     // No move assignment needed usually for simple guards

    ~TimerGuard() {
        if (active_) {
             PerformanceMetrics::getInstance().endTimer(label_);
        }
    }

private:
    std::string label_;
    bool active_; // To handle potential moves
};

// Macro to easily time a scope
#define TIME_FUNCTION(label) \
    TimerGuard timer_guard_##__COUNTER__(label) // Use __COUNTER__ for unique names

#else // Performance metrics disabled
#define TIME_FUNCTION(label) (void)0 // No-op macro
#endif // ENABLE_PERFORMANCE_METRICS


// ---- VideoAligner Implementation ----

// Computes image pyramid, handling initialization and buffer allocation.
bool VideoAligner::ComputePyramid(const cv::Mat& inputFrame, const VideoAlignerParams& params) {
    TIME_FUNCTION("ComputePyramid");

    int width = inputFrame.cols;
    int height = inputFrame.rows;

    // Initialize or re-initialize if dimensions change
    if (ScalePyramid[0].empty() || width != LastWidth || height != LastHeight || PyramidLevels < 1) {
        TIME_FUNCTION("ComputePyramid_Setup");

        // Start with the *very first* image being stored in the key‑frame slot
        // so that subsequent frames are always compared against this reference
        // image.  Using the opposite ordering (current=0, keyframe=1) caused
        // the roles of template and key‑frame to be swapped on the first
        // alignment pair which led to an ill‑conditioned Hessian and early
        // divergence.
        CurrFrameIndex = KeyframeIndex;      // 1
        PrevFrameIndex = NonKeyframeIndex;   // 0
        FramesAccumulated = 0;
        LastWidth = width;
        LastHeight = height;

        // Determine number of pyramid levels
        PyramidLevels = 0;
        int current_width = width;
        int current_height = height;
        do {
            PyramidLevels++;
            current_width /= 2;
            current_height /= 2;
        } while (current_width >= params.pyramid_min_width && current_height >= params.pyramid_min_height && PyramidLevels < 10); // Added safety limit

         if (PyramidLevels == 0) {
             std::cerr << "Error: Input image too small for pyramid generation based on min dimensions." << std::endl;
             return false; // Cannot proceed
         }

        // Clear and resize pyramid buffers
        for(int i=0; i<2; ++i) {
            ScalePyramid[i].clear();
            ScalePyramid[i].resize(PyramidLevels);
        }

        // Resize keyframe-specific buffers
        KeyframeGradX.resize(PyramidLevels);
        KeyframeGradY.resize(PyramidLevels);
        KeyframeArgMaxX.resize(PyramidLevels);
        KeyframeArgMaxY.resize(PyramidLevels);
        KeyframeTileSize.resize(PyramidLevels);
        KeyframeJacobianX.resize(PyramidLevels);
        KeyframeJacobianY.resize(PyramidLevels);

        // Resize selection and difference buffers
        SelectedPixelsX.resize(PyramidLevels);
        SelectedPixelsY.resize(PyramidLevels);
        SelectedJacobianX.resize(PyramidLevels);
        SelectedJacobianY.resize(PyramidLevels);
        WarpDiffX.resize(PyramidLevels);
        WarpDiffY.resize(PyramidLevels);

        // Note: Actual Halide buffer allocation happens lazily or when needed later.
        // We don't allocate all buffers here to save memory initially.
        std::cout << "Pyramid setup: " << PyramidLevels << " levels for " << width << "x" << height << std::endl;
    } else {
        // Swap frame indices for the next frame
        PrevFrameIndex = CurrFrameIndex;
        CurrFrameIndex = (CurrFrameIndex + 1) % 2;
    }

    // Convert input frame to grayscale for the current index
    {
        TIME_FUNCTION("ConvertToGray"); // Changed from BGR
        // Handle different input types gracefully
        if (inputFrame.channels() == 3) {
            cv::cvtColor(inputFrame, GrayInput[CurrFrameIndex], cv::COLOR_BGR2GRAY);
        } else if (inputFrame.channels() == 4) {
             cv::cvtColor(inputFrame, GrayInput[CurrFrameIndex], cv::COLOR_BGRA2GRAY);
        } else if (inputFrame.channels() == 1) {
            GrayInput[CurrFrameIndex] = inputFrame; // Already grayscale
        } else {
             std::cerr << "Error: Unsupported number of channels in input frame: " << inputFrame.channels() << std::endl;
             return false;
        }

        // --- Brightness / Gain Normalisation ---------------------------------
        //  Apply CLAHE (contrast‑limited adaptive histogram equalisation) which
        //  equalises local contrast and greatly reduces the impact of global
        //  exposure changes on gradient magnitudes.  This is lightweight and
        //  avoids negative values that would arise from mean‑subtraction.
        static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
        cv::Mat normalised;
        clahe->apply(GrayInput[CurrFrameIndex], normalised);
        GrayInput[CurrFrameIndex] = normalised;
    }

    // Convert the base level grayscale cv::Mat to a Halide buffer
    {
        TIME_FUNCTION("MatToHalide_L0");
        try {
            ScalePyramid[CurrFrameIndex][0] = mat_to_halide_buffer_u8(GrayInput[CurrFrameIndex]);
        } catch (const std::runtime_error& e) {
            std::cerr << "Error converting Mat to Halide buffer: " << e.what() << std::endl;
            LastWidth = -1; // Force reinitialization on next valid frame
            return false;
        }
    }

    // Build the pyramid by downsampling
    for (int i = 1; i < PyramidLevels; i++) {
        TIME_FUNCTION("PyrDown_" + std::to_string(i));
        // Output buffer allocation happens inside PyrDown wrapper if needed
        if (!PyrDown(ScalePyramid[CurrFrameIndex][i-1], ScalePyramid[CurrFrameIndex][i])) {
             std::cerr << "Error computing pyramid level " << i << std::endl;
             LastWidth = -1; // Force reinit
             return false;
        }
    }

    // Prepare image for phase correlation (if enabled)
    if (params.phase_correlate && PhaseLevel < PyramidLevels) {
         TIME_FUNCTION("PhaseLayerConversion");
        try {
            // Convert the Halide buffer at PhaseLevel to cv::Mat
            cv::Mat phase_layer_mat = halide_buffer_to_mat(ScalePyramid[CurrFrameIndex][PhaseLevel]);
            // Convert to 32-bit float for phaseCorrelate
            phase_layer_mat.convertTo(PhaseImage[CurrFrameIndex], CV_32F);
         } catch (const std::runtime_error& e) {
             std::cerr << "Error preparing phase correlation image: " << e.what() << std::endl;
             // Continue without phase correlation? Or return false? Decide based on requirements.
             // For now, just log and continue.
         }
    }

    // Increment frame counter and check if we have enough frames to start alignment
    FramesAccumulated++;
    return FramesAccumulated >= 2; // Need at least two frames (previous and current)
}

// Precomputes keyframe data (gradients, maxima, Jacobians) for all pyramid levels.
bool VideoAligner::ComputeKeyFrame() {
    TIME_FUNCTION("ComputeKeyFrame_Total"); // Renamed timer label

    // The keyframe data is always stored at KeyframeIndex (e.g., index 1)
    const int kf_idx = KeyframeIndex;

    for (int i = 0; i < PyramidLevels; i++) {
        auto& keyframe_level = ScalePyramid[kf_idx][i]; // Image at this level
        auto& grad_x = KeyframeGradX[i];
        auto& grad_y = KeyframeGradY[i];

        // 1. Compute Gradients
        {
            TIME_FUNCTION("GradXY_" + std::to_string(i));
            // Allocation happens inside GradXY wrapper if needed
            if (!GradXY(keyframe_level, grad_x, grad_y)) {
                std::cerr << "Failed to compute gradient images for keyframe at level " << i << std::endl;
                return false;
            }
        }

        // 2. Compute Gradient ArgMax within tiles
        auto& argmax_x = KeyframeArgMaxX[i];
        auto& argmax_y = KeyframeArgMaxY[i];
        int& tile_size = KeyframeTileSize[i]; // Get reference to store tile size

        {
            TIME_FUNCTION("GradArgMax_" + std::to_string(i));
            // GradArgMax determines tile_size and allocates buffers if needed
            if (!GradArgMax(grad_x, grad_y, tile_size, argmax_x, argmax_y)) {
                std::cerr << "Failed to compute gradient argmax for keyframe at level " << i << std::endl;
                return false;
            }
            #ifdef ENABLE_PERFORMANCE_METRICS
                PerformanceMetrics::getInstance().logMetric("TileSize_L" + std::to_string(i), tile_size);
                PerformanceMetrics::getInstance().logMetric("TileCount_L" + std::to_string(i), (double)argmax_x.dim(0).extent() * argmax_x.dim(1).extent());
            #endif
        }

        // 3. Compute Sparse Jacobian
        auto& jacobian_x = KeyframeJacobianX[i];
        auto& jacobian_y = KeyframeJacobianY[i];
        int current_width = grad_x.width(); // Get dimensions for Jacobian calculation
        int current_height = grad_x.height();

        {
            TIME_FUNCTION("SparseJacobian_" + std::to_string(i));
             // Allocation happens inside SparseJacobian wrapper if needed
            if (!SparseJacobian(grad_x, grad_y, argmax_x, argmax_y, current_width, current_height, jacobian_x, jacobian_y)) {
                std::cerr << "Failed to compute sparse Jacobian for keyframe at level " << i << std::endl;
                return false;
            }
        }
    }

    return true;
}


// Computes Hessian matrix H = Sum( JᵀJ ) from selected Jacobians.
// Assumes H is 4x4.
static cv::Mat ComputeHessianFromSelected(
    const Halide::Runtime::Buffer<float> &selected_jacobian_x, // Shape: (N_selected_x, 4)
    const Halide::Runtime::Buffer<float> &selected_jacobian_y) // Shape: (N_selected_y, 4)
{
    TIME_FUNCTION("ComputeHessianFromSelected");

    cv::Mat H = cv::Mat::zeros(4, 4, CV_64F); // Use double precision for accumulation
    double* H_data = H.ptr<double>(0); // Direct pointer access

    // Helper lambda robust to arbitrary Halide Buffer strides
    auto accumulate_hessian = [&H_data](const Halide::Runtime::Buffer<float> &jacobian) {
        const int extent_x = jacobian.dim(0).extent();   // tile count X
        const int extent_y = jacobian.dim(1).extent();   // tile count Y

        // Iterate over every selected point (x,y)
        for (int y = 0; y < extent_y; ++y) {
            for (int x = 0; x < extent_x; ++x) {
                // Fetch the 4‑component Jacobian vector (cast to double)
                double j[4] = {
                    static_cast<double>(jacobian(x, y, 0)),
                    static_cast<double>(jacobian(x, y, 1)),
                    static_cast<double>(jacobian(x, y, 2)),
                    static_cast<double>(jacobian(x, y, 3))
                };

                // Accumulate outer product into upper triangle of H
                double *H_row_ptr = H_data;
                for (int r = 0; r < 4; ++r) {
                    for (int c = r; c < 4; ++c) {
                        H_row_ptr[c] += j[r] * j[c];
                    }
                    H_row_ptr += 4; // next row
                }
            }
        }
    };

    // Process Jacobians from both X and Y gradient selections
    {
        TIME_FUNCTION("AccumulateHessianX");
        accumulate_hessian(selected_jacobian_x);
    }
    {
        TIME_FUNCTION("AccumulateHessianY");
        accumulate_hessian(selected_jacobian_y);
    }

    // Fill the lower triangle of H using symmetry H(c, r) = H(r, c)
    for (int r = 1; r < 4; ++r) {
        for (int c = 0; c < r; ++c) {
            H.at<double>(r, c) = H.at<double>(c, r);
        }
    }

    return H;
}


// Main alignment function for processing the next frame.
bool VideoAligner::AlignNextFrame(
    const cv::Mat& inputFrame,
    SimilarityTransform& transform, // Output parameter
    const VideoAlignerParams& params)
{
#ifdef ENABLE_PERFORMANCE_METRICS
    // Start top-level timer for the whole frame processing
    PerformanceMetrics::getInstance().startTimer("TotalFrameTime");
    // Use a guard for the main function scope as well
    TIME_FUNCTION("AlignNextFrame");
#endif

    // Initialize output transform to identity
    transform = SimilarityTransform();

    // 1. Compute Image Pyramid for the current frame
    {
        TIME_FUNCTION("ComputePyramidCall");
        if (!ComputePyramid(inputFrame, params)) {
            // Not enough frames yet, or error occurred
            #ifdef ENABLE_PERFORMANCE_METRICS
            PerformanceMetrics::getInstance().endTimer("TotalFrameTime"); // Stop timer before returning
            PerformanceMetrics::getInstance().logMetric("FrameResult", 0.0); // 0.0 = Incomplete/Error
            #endif
            return false;
        }
    }

     // 2. Compute Keyframe Data if this is the designated keyframe index
     //    This only needs to be done once per keyframe.
    if (FramesAccumulated == 2 || CurrFrameIndex == KeyframeIndex) { // Compute on first alignment pair or when keyframe buffer is updated
        // Check if keyframe data is already computed for these dimensions
        if (KeyframeGradX[0].data() == nullptr) { // Check if base level gradient exists
             TIME_FUNCTION("ComputeKeyFrameCall");
             if (!ComputeKeyFrame()) {
                std::cerr << "Keyframe computation failed." << std::endl;
                LastWidth = -1; // Force re-initialization on next frame
                #ifdef ENABLE_PERFORMANCE_METRICS
                PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
                PerformanceMetrics::getInstance().logMetric("FrameResult", 0.0); // Error
                #endif
                return false;
            }
        }
    }

    // 3. Optional: Phase Correlation for Initial Translation Guess
    if (params.phase_correlate && PhaseLevel < PyramidLevels && FramesAccumulated >= 2) {
        TIME_FUNCTION("PhaseCorrelation");
        try {
            // Ensure Phase images are valid
             if (PhaseImage[PrevFrameIndex].empty() || PhaseImage[CurrFrameIndex].empty()) {
                 std::cerr << "Warning: Phase correlation enabled but images are missing for level " << PhaseLevel << std::endl;
             } else {
                cv::Point2d detected_shift;
                double response = 0.0;
                // Use windowing (e.g., Hanning) to reduce edge artifacts if needed
                // cv::Mat window; cv::createHanningWindow(window, PhaseImage[PrevFrameIndex].size(), CV_32F);
                // detected_shift = cv::phaseCorrelate(PhaseImage[PrevFrameIndex], PhaseImage[CurrFrameIndex], window, &response);
                detected_shift = cv::phaseCorrelate(PhaseImage[PrevFrameIndex], PhaseImage[CurrFrameIndex], cv::noArray(), &response);

                #ifdef ENABLE_PERFORMANCE_METRICS
                PerformanceMetrics::getInstance().logMetric("PhaseCorrelateResponse", response);
                #endif

                if (response > params.phase_correlate_threshold) {
                    // Scale the shift from PhaseLevel resolution back to base level (Level 0)
                    double scale_factor = static_cast<double>(1 << PhaseLevel); // 2^PhaseLevel
                    transform.TX = detected_shift.x * scale_factor;
                    transform.TY = detected_shift.y * scale_factor;
                     std::cout << "Phase Corr Shift (L" << PhaseLevel << "): (" << detected_shift.x << "," << detected_shift.y << ") Resp: " << response << " -> Initial T: (" << transform.TX << "," << transform.TY << ")" << std::endl;
                    // Note: Phase correlation only gives translation (A=0, B=0).
                } else {
                     std::cout << "Phase Corr response below threshold: " << response << std::endl;
                }
            }
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV Error during Phase Correlation: " << e.what() << std::endl;
            // Continue without phase correlation result
        }
    }


    // 4. Iterative Alignment across Pyramid Levels (Coarse to Fine)
    for (int i = PyramidLevels - 1; i >= 0; --i) {
        std::string level_label = "PyramidLevel_" + std::to_string(i);
        TIME_FUNCTION(level_label);

        // References to data for the current level 'i'
        auto& template_image = ScalePyramid[NonKeyframeIndex][i]; // Current frame (if not keyframe) acts as template
        auto& keyframe_image = ScalePyramid[KeyframeIndex][i];   // Keyframe acts as image to be warped

        // Image dimensions at this level
        const int image_width = keyframe_image.width();
        const int image_height = keyframe_image.height();
        if (image_width <= 0 || image_height <= 0) {
             std::cerr << "Error: Invalid image dimensions at level " << i << std::endl;
             #ifdef ENABLE_PERFORMANCE_METRICS
             PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
             PerformanceMetrics::getInstance().logMetric("FrameResult", 0.0); // Error
             #endif
             return false;
        }
        const double Cx = static_cast<double>(image_width) / 2.0;
        const double Cy = static_cast<double>(image_height) / 2.0;


        // Keyframe gradient maxima locations for this level
        auto& grad_argmax_x = KeyframeArgMaxX[i];
        auto& grad_argmax_y = KeyframeArgMaxY[i];
        if (grad_argmax_x.data() == nullptr || grad_argmax_y.data() == nullptr) {
             std::cerr << "Error: Keyframe ArgMax data missing for level " << i << std::endl;
             #ifdef ENABLE_PERFORMANCE_METRICS
             PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
             PerformanceMetrics::getInstance().logMetric("FrameResult", 0.0); // Error
             #endif
             return false;
        }


        // Buffers for warp difference calculation
        auto& warpdiff_x = WarpDiffX[i];
        auto& warpdiff_y = WarpDiffY[i];

        // --- Point Selection based on Warp Difference ---

        // a. Calculate warp difference at all gradient maxima points
        {
            TIME_FUNCTION("SparseWarpDiff_X_" + std::to_string(i));
            // SparseWarpDiff calculates | Template(argmax_loc) - Warp(Keyframe)(argmax_loc) |
            // Allocation happens inside wrapper if needed.
            if (!SparseWarpDiff(template_image, keyframe_image, grad_argmax_x, transform, image_width, image_height, warpdiff_x)) {
                std::cerr << "Failed to compute warp diff X at level " << i << std::endl;
                #ifdef ENABLE_PERFORMANCE_METRICS
                PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
                PerformanceMetrics::getInstance().logMetric("FrameResult", 0.0); // Error
                #endif
                return false;
            }
        }
        {
            TIME_FUNCTION("SparseWarpDiff_Y_" + std::to_string(i));
             if (!SparseWarpDiff(template_image, keyframe_image, grad_argmax_y, transform, image_width, image_height, warpdiff_y)) {
                std::cerr << "Failed to compute warp diff Y at level " << i << std::endl;
                 #ifdef ENABLE_PERFORMANCE_METRICS
                PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
                PerformanceMetrics::getInstance().logMetric("FrameResult", 0.0); // Error
                #endif
                return false;
            }
        }

        // b. Collect differences and tile indices into vectors
        {
            TIME_FUNCTION("DeltaPixelsSetup_" + std::to_string(i));
            int tiles_w = grad_argmax_x.dim(0).extent();
            int tiles_h = grad_argmax_x.dim(1).extent();
            size_t total_tiles = static_cast<size_t>(tiles_w) * tiles_h;

            DeltaPixelsX.clear();
            DeltaPixelsX.reserve(total_tiles);
            warpdiff_x.copy_to_host(); // Ensure data is accessible
            for (int ty = 0; ty < tiles_h; ++ty) {
                for (int tx = 0; tx < tiles_w; ++tx) {
                    DeltaPixelsX.push_back({warpdiff_x(tx, ty), static_cast<uint16_t>(tx), static_cast<uint16_t>(ty)});
                }
            }

            tiles_w = grad_argmax_y.dim(0).extent(); // Re-read in case they differ (shouldn't)
            tiles_h = grad_argmax_y.dim(1).extent();
            total_tiles = static_cast<size_t>(tiles_w) * tiles_h;
            DeltaPixelsY.clear();
            DeltaPixelsY.reserve(total_tiles);
            warpdiff_y.copy_to_host(); // Ensure data is accessible
             for (int ty = 0; ty < tiles_h; ++ty) {
                for (int tx = 0; tx < tiles_w; ++tx) {
                    DeltaPixelsY.push_back({warpdiff_y(tx, ty), static_cast<uint16_t>(tx), static_cast<uint16_t>(ty)});
                }
            }
        }

        // c. Select subset with smallest differences using nth_element
        size_t selected_count_x = 0;
        size_t selected_count_y = 0;
        {
            TIME_FUNCTION("NthElement_" + std::to_string(i));

            selected_count_x = static_cast<size_t>(DeltaPixelsX.size() * params.smallest_fraction);
            if (selected_count_x < 10) selected_count_x = std::min(DeltaPixelsX.size(), (size_t)10); // Ensure minimum points
            if (!DeltaPixelsX.empty()) {
                std::nth_element(
                    DeltaPixelsX.begin(),
                    DeltaPixelsX.begin() + selected_count_x,
                    DeltaPixelsX.end()); // std::vector::end is correct
                DeltaPixelsX.resize(selected_count_x); // Keep only the smallest N
            } else {
                selected_count_x = 0; // Handle empty case
            }


            selected_count_y = static_cast<size_t>(DeltaPixelsY.size() * params.smallest_fraction);
             if (selected_count_y < 10) selected_count_y = std::min(DeltaPixelsY.size(), (size_t)10); // Ensure minimum points
            if (!DeltaPixelsY.empty()) {
                std::nth_element(
                    DeltaPixelsY.begin(),
                    DeltaPixelsY.begin() + selected_count_y,
                    DeltaPixelsY.end());
                DeltaPixelsY.resize(selected_count_y);
             } else {
                selected_count_y = 0; // Handle empty case
            }

            #ifdef ENABLE_PERFORMANCE_METRICS
            PerformanceMetrics::getInstance().logMetric("SelectedPointsX_L" + std::to_string(i), selected_count_x);
            PerformanceMetrics::getInstance().logMetric("SelectedPointsY_L" + std::to_string(i), selected_count_y);
            #endif

             // Check if enough points were selected
             if (selected_count_x < 4 || selected_count_y < 4) { // Need at least 4 parameters
                 std::cerr << "Warning: Insufficient points selected at level " << i << " (X:" << selected_count_x << ", Y:" << selected_count_y << "). Skipping level or failing." << std::endl;
                 // Option 1: Skip level (might be okay if coarse levels already aligned)
                 // continue;
                 // Option 2: Fail alignment
                 #ifdef ENABLE_PERFORMANCE_METRICS
                 PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
                 PerformanceMetrics::getInstance().logMetric("FrameResult", 0.0); // Error
                 #endif
                 return false;
             }
        }

        // d. Prepare buffers for selected points and their Jacobians
        auto& selected_pixels_x = SelectedPixelsX[i];
        auto& selected_pixels_y = SelectedPixelsY[i];
        auto& selected_jacobian_x = SelectedJacobianX[i];
        auto& selected_jacobian_y = SelectedJacobianY[i];
        auto& keyframe_jacobian_x = KeyframeJacobianX[i]; // Source Jacobians
        auto& keyframe_jacobian_y = KeyframeJacobianY[i];

        {
            TIME_FUNCTION("BufferSetup_" + std::to_string(i));

            // Resize Halide buffers if needed (allocates memory)
            // Selected Pixels: (N_selected, 2 for xy coord)
            if (selected_pixels_x.dimensions() != 2 || selected_pixels_x.dim(0).extent() != (int)selected_count_x || selected_pixels_x.dim(1).extent() != 2) {
                selected_pixels_x = Halide::Runtime::Buffer<uint16_t>(selected_count_x, 2);
            }
             if (selected_pixels_y.dimensions() != 2 || selected_pixels_y.dim(0).extent() != (int)selected_count_y || selected_pixels_y.dim(1).extent() != 2) {
                selected_pixels_y = Halide::Runtime::Buffer<uint16_t>(selected_count_y, 2);
            }
            // Selected Jacobians: (N_selected, 4 for params A, B, TX, TY)
             if (selected_jacobian_x.dimensions() != 2 || selected_jacobian_x.dim(0).extent() != (int)selected_count_x || selected_jacobian_x.dim(1).extent() != 4) {
                selected_jacobian_x = Halide::Runtime::Buffer<float>(selected_count_x, 4);
            }
             if (selected_jacobian_y.dimensions() != 2 || selected_jacobian_y.dim(0).extent() != (int)selected_count_y || selected_jacobian_y.dim(1).extent() != 4) {
                selected_jacobian_y = Halide::Runtime::Buffer<float>(selected_count_y, 4);
            }
        }

        // e. Copy selected pixel coordinates and Jacobians into the Halide buffers
        {
            TIME_FUNCTION("JacobianCopy_" + std::to_string(i)); // Renamed timer

            // Ensure source data is on host
            grad_argmax_x.copy_to_host();
            grad_argmax_y.copy_to_host();
            keyframe_jacobian_x.copy_to_host();
            keyframe_jacobian_y.copy_to_host();

            // Iterate through the selected DeltaPixels and copy data
            for (size_t j = 0; j < selected_count_x; ++j) {
                int tile_x = DeltaPixelsX[j].tile_x;
                int tile_y = DeltaPixelsX[j].tile_y;

                // Copy pixel coordinates from grad_argmax_x buffer
                selected_pixels_x(j, 0) = grad_argmax_x(tile_x, tile_y, 0); // Pixel X coord
                selected_pixels_x(j, 1) = grad_argmax_x(tile_x, tile_y, 1); // Pixel Y coord

                // Copy Jacobian vector from keyframe_jacobian_x buffer
                for (int k = 0; k < 4; ++k) { // k corresponds to parameter index (A, B, TX, TY)
                    selected_jacobian_x(j, k) = keyframe_jacobian_x(tile_x, tile_y, k);
                }
            }
             for (size_t j = 0; j < selected_count_y; ++j) {
                int tile_x = DeltaPixelsY[j].tile_x;
                int tile_y = DeltaPixelsY[j].tile_y;

                selected_pixels_y(j, 0) = grad_argmax_y(tile_x, tile_y, 0);
                selected_pixels_y(j, 1) = grad_argmax_y(tile_x, tile_y, 1);

                for (int k = 0; k < 4; ++k) {
                    selected_jacobian_y(j, k) = keyframe_jacobian_y(tile_x, tile_y, k);
                }
            }
            // Mark buffers as dirty on host if needed (usually needed after manual copy)
            selected_pixels_x.set_host_dirty();
            selected_pixels_y.set_host_dirty();
            selected_jacobian_x.set_host_dirty();
            selected_jacobian_y.set_host_dirty();
        }

        // f. Compute Hessian Matrix H = Sum(JᵀJ)
        cv::Mat H;
        {
            TIME_FUNCTION("ComputeHessian_" + std::to_string(i));
            // Function now takes both sets of selected Jacobians
            H = ComputeHessianFromSelected(selected_jacobian_x, selected_jacobian_y);
        }

        // g. Check Hessian condition number and apply regularization if needed
        cv::Mat Hinv;
        {
            TIME_FUNCTION("ConditionCheckAndInvert_" + std::to_string(i));
            cv::SVD svd(H, cv::SVD::NO_UV); // Compute singular values only
            double min_sv = svd.w.at<double>(svd.w.rows - 1);
            double max_sv = svd.w.at<double>(0);

            const double sv_epsilon = 1e-12; // absolute tiny value
            double condition_number;
            if (max_sv < sv_epsilon) {
                // Degenerate zero matrix – no information
                condition_number = std::numeric_limits<double>::infinity();
            } else if (min_sv < sv_epsilon) {
                condition_number = std::numeric_limits<double>::infinity();
            } else {
                condition_number = max_sv / min_sv;
            }

            #ifdef ENABLE_PERFORMANCE_METRICS
            PerformanceMetrics::getInstance().logMetric("ConditionNumber_L" + std::to_string(i), condition_number);
            #endif

            const double condition_threshold = 1e6; // Threshold for ill-conditioning
            if (condition_number > condition_threshold) {
                std::cout << "Warning: Hessian ill-conditioned at level " << i << " (Cond #: " << condition_number << "). Applying Tikhonov regularization." << std::endl;
                // Apply Tikhonov regularization: H = H + lambda * I
                double lambda = (max_sv < sv_epsilon ? 1e-3 : 1e-6 * max_sv);
                H += cv::Mat::eye(H.rows, H.cols, CV_64F) * lambda;
                #ifdef ENABLE_PERFORMANCE_METRICS
                PerformanceMetrics::getInstance().logMetric("RegularizationLambda_L" + std::to_string(i), lambda);
                #endif
            }

            // Invert the (potentially regularized) Hessian
            // Use SVD-based inversion for robustness against near-singularity
            Hinv = H.inv(cv::DECOMP_SVD);
        }


        // --- Inverse Compositional Algorithm (ICA) Iterations ---
        int iterations_performed = 0;

        // Store initial corner locations (warped by current transform estimate)
        Point ul0_pt{0.0, 0.0}, ur0_pt{(double)image_width - 1.0, 0.0};
        Point ll0_pt{0.0, (double)image_height - 1.0}, lr0_pt{(double)image_width - 1.0, (double)image_height - 1.0};
        Point ul1 = transform.warp(ul0_pt, Cx, Cy);
        Point ur1 = transform.warp(ur0_pt, Cx, Cy);
        Point ll1 = transform.warp(ll0_pt, Cx, Cy);
        Point lr1 = transform.warp(lr0_pt, Cx, Cy);

        for (int iter = 0; iter < params.max_iters; ++iter) {
            TIME_FUNCTION("ICAIteration_" + std::to_string(i) + "_" + std::to_string(iter));
            iterations_performed++;

            // i. Compute Sum( Jᵀ * (Template - Warped_Keyframe) ) using SparseICA Halide kernel
            if (!SparseICA(
                    template_image,
                    keyframe_image,
                    selected_pixels_x,
                    selected_pixels_y,
                    selected_jacobian_x, // Pass selected Jacobians
                    selected_jacobian_y,
                    transform,          // Current transform estimate T
                    image_width,
                    image_height,
                    IcaResult))         // Output buffer for the sum vector
            {
                std::cerr << "Failed to compute ICA result at level " << i << ", iter " << iter << std::endl;
                 #ifdef ENABLE_PERFORMANCE_METRICS
                 PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
                 PerformanceMetrics::getInstance().logMetric("FrameResult", 0.0); // Error
                 #endif
                return false;
            }

            // ii. Solve for delta parameters: dt = H⁻¹ * IcaResult
            cv::Mat dt_mat; // Delta Transform parameters as a 4x1 matrix
            {
                TIME_FUNCTION("MatrixMultiply_" + std::to_string(i) + "_" + std::to_string(iter));
                IcaResult.copy_to_host(); // Ensure result is on host
                cv::Mat ica_vec = halide_vec4_to_mat(IcaResult); // Convert Halide vec to cv::Mat
                dt_mat = Hinv * ica_vec; // dt = H⁻¹ * Sum(Jᵀ * res)
            }

            // iii. Create delta transform and update overall transform
            SimilarityTransform delta_transform;
            // Note: The 'scale' factor used previously in Jacobian is handled implicitly
            //       by how A and B affect coordinates relative to the center.
            //       The delta values directly apply to A, B, TX, TY.
            delta_transform.A = dt_mat.at<double>(0, 0);
            delta_transform.B = dt_mat.at<double>(1, 0);
            delta_transform.TX = dt_mat.at<double>(2, 0);
            delta_transform.TY = dt_mat.at<double>(3, 0);

            {
                TIME_FUNCTION("TransformCompose_" + std::to_string(i) + "_" + std::to_string(iter));
                // Update rule for ICA: T_new = T_old * inv(delta_T)
                // However, the way Jacobians/ICA are set up often computes the update
                // such that T_new = delta_T * T_old (or compose(delta_T, T_old)).
                // Verify based on the specific ICA formulation and Jacobian definition.
                // Assuming T_new = compose(delta_T, T_old) based on typical LK/ICA structure.
                // If divergence occurs, try T_new = compose(T_old, delta_T.inverse(Cx, Cy))
                transform = delta_transform.compose(transform);
            }

            // iv. Check for convergence based on corner displacement
            Point ul2 = transform.warp(ul0_pt, Cx, Cy);
            Point ur2 = transform.warp(ur0_pt, Cx, Cy);
            Point ll2 = transform.warp(ll0_pt, Cx, Cy);
            Point lr2 = transform.warp(lr0_pt, Cx, Cy);

            // Max distance any corner moved in this iteration
            double max_dx = std::max(ul1.distance(ul2), ur1.distance(ur2));
            double max_dy = std::max(ll1.distance(ll2), lr1.distance(lr2));
            double displacement_iter = std::max(max_dx, max_dy);

            // Update corner positions for next iteration's check
            ul1 = ul2; ur1 = ur2; ll1 = ll2; lr1 = lr2;

            #ifdef ENABLE_PERFORMANCE_METRICS
            PerformanceMetrics::getInstance().logMetric("Displacement_L" + std::to_string(i) + "_Iter" + std::to_string(iter), displacement_iter);
            #endif

            // Check convergence threshold
            if (displacement_iter < params.threshold) {
                // std::cout << "Level " << i << " converged in " << iterations_performed << " iterations." << std::endl;
                break; // Exit iteration loop for this level
            }

            // Check if max iterations exceeded for this level
            if (iter >= params.max_iters - 1) {
                std::cerr << "Warning: Max iterations (" << params.max_iters << ") exceeded at level " << i << ". Displacement: " << displacement_iter << std::endl;
                #ifdef ENABLE_PERFORMANCE_METRICS
                 PerformanceMetrics::getInstance().logMetric("MaxIterationsExceeded_L" + std::to_string(i), 1.0);
                 PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
                 PerformanceMetrics::getInstance().logMetric("FrameResult", 0.0); // Error/Diverged
                #endif
                return false; // Alignment failed to converge
            }
        } // End ICA iteration loop

        #ifdef ENABLE_PERFORMANCE_METRICS
        PerformanceMetrics::getInstance().logMetric("IterationsPerformed_L" + std::to_string(i), iterations_performed);
        #endif

        // --- Post-Iteration Checks for the Level ---

        // Check total displacement from start of level against threshold
        double total_disp_level = transform.maxCornerDisplacement(image_width, image_height); // Uses center-warp internally
        #ifdef ENABLE_PERFORMANCE_METRICS
         PerformanceMetrics::getInstance().logMetric("TotalDisplacement_L" + std::to_string(i), total_disp_level);
        #endif

        if (total_disp_level > params.max_displacement) {
            std::cerr << "Warning: Max displacement exceeded at level " << i << " (" << total_disp_level << " > " << params.max_displacement << "). Alignment potentially unstable." << std::endl;
            #ifdef ENABLE_PERFORMANCE_METRICS
            PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
            PerformanceMetrics::getInstance().logMetric("FrameResult", 0.0); // Error/Unstable
            #endif
            return false; // Reject unstable result
        }


        // 5. Propagate Transform to Next Finer Level (if not at base level)
        if (i > 0) {
            // Scale translation components by 2 when moving up the pyramid
            // Rotation (A) and scale/shear (B) are scale-invariant in this formulation.
            transform.TX *= 2.0;
            transform.TY *= 2.0;
        }

    } // End pyramid level loop


    // 6. Final Transform Inversion (if necessary)
    // The loop computes the transform T mapping points from NonKeyframeIndex -> KeyframeIndex.
    // If the *current* frame processed is the NonKeyframeIndex, we need the inverse transform
    // to represent the motion from the Previous (Keyframe) to the Current (NonKeyframe).
    if (CurrFrameIndex == NonKeyframeIndex) {
        TIME_FUNCTION("TransformInverse");
        // Calculate center of the base level image (Level 0)
        double base_width = static_cast<double>(ScalePyramid[KeyframeIndex][0].width());
        double base_height = static_cast<double>(ScalePyramid[KeyframeIndex][0].height());
        double base_Cx = base_width / 2.0;
        double base_Cy = base_height / 2.0;
        transform = transform.inverse(base_Cx, base_Cy);
    }
    // If CurrFrameIndex == KeyframeIndex, the computed transform already maps Previous -> Current.


#ifdef ENABLE_PERFORMANCE_METRICS
    // Log successful frame and print metrics periodically
    double totalTime = PerformanceMetrics::getInstance().endTimer("TotalFrameTime");
    PerformanceMetrics::getInstance().logMetric("FrameResult", 1.0); // 1.0 = Success
    PerformanceMetrics::getInstance().logMetric("SuccessfulFrameTime", totalTime); // Log time only for successful frames

    static int frameCountSuccess = 0;
    const int print_interval = 100; // Print every 100 successful frames
    if (++frameCountSuccess % print_interval == 0) {
        PerformanceMetrics::getInstance().printAllMetrics();
         // Optional: Reset metrics after printing
         // PerformanceMetrics::getInstance().reset();
    }
#endif

    return true; // Alignment successful
}
