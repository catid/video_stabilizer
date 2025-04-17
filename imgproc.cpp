#include "imgproc.hpp"

#include <sstream>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <cmath> // For std::abs, std::sqrt

// Include Halide runtime headers for Buffer<> and generated function headers
#include "HalideBuffer.h"
#include "sparse_jac.h"
#include "pyr_down.h"
#include "image_warp.h"
#include "grad_xy.h"
#include "grad_argmax_20.h" // Assuming these exist based on original code
#include "grad_argmax_18.h"
#include "grad_argmax_16.h"
#include "grad_argmax_14.h"
#include "grad_argmax_12.h"
#include "grad_argmax_10.h"
#include "grad_argmax_8.h"
#include "grad_argmax_6.h"
#include "grad_argmax_4.h"
#include "grad_argmax_2.h"
#include "sparse_ica.h"
#include "sparse_warpdiff.h"


// --- Halide Function Wrappers ---

bool SparseJacobian(
    Halide::Runtime::Buffer<float>& grad_x,
    Halide::Runtime::Buffer<float>& grad_y,
    Halide::Runtime::Buffer<uint16_t>& local_max_x,
    Halide::Runtime::Buffer<uint16_t>& local_max_y,
    int image_width,
    int image_height,
    Halide::Runtime::Buffer<float>& output_x,
    Halide::Runtime::Buffer<float>& output_y)
{
    // Ensure output buffers have the correct dimensions based on local_max inputs
    // Dimensions: (tile_x_count, tile_y_count, 4)
    if (output_x.dimensions() != 3 ||
        output_x.dim(0).extent() != local_max_x.dim(0).extent() ||
        output_x.dim(1).extent() != local_max_x.dim(1).extent() ||
        output_x.dim(2).extent() != 4) // Dimension for parameters (A, B, TX, TY)
    {
        // std::cout << "Resizing Jacobian buffers: " << local_max_x.dim(0).extent() << "x" << local_max_x.dim(1).extent() << std::endl;
        output_x = Halide::Runtime::Buffer<float>(local_max_x.dim(0).extent(), local_max_x.dim(1).extent(), 4);
    }
     if (output_y.dimensions() != 3 ||
        output_y.dim(0).extent() != local_max_y.dim(0).extent() || // Use local_max_y dims
        output_y.dim(1).extent() != local_max_y.dim(1).extent() ||
        output_y.dim(2).extent() != 4)
    {
        output_y = Halide::Runtime::Buffer<float>(local_max_y.dim(0).extent(), local_max_y.dim(1).extent(), 4);
    }


    // Call the Halide-generated sparse_jac function
    int r = sparse_jac(grad_x, grad_y, local_max_x, local_max_y, image_width, image_height, output_x, output_y);
    if (r != 0) {
        std::cerr << "Halide sparse_jac call failed with error code: " << r << std::endl;
    }
    return r == 0;
}

bool SparseICA(
    Halide::Runtime::Buffer<uint8_t>& input_template,
    Halide::Runtime::Buffer<uint8_t>& input_keyframe,
    Halide::Runtime::Buffer<uint16_t>& selected_pixels_x,
    Halide::Runtime::Buffer<uint16_t>& selected_pixels_y,
    Halide::Runtime::Buffer<float>& selected_jacobians_x,
    Halide::Runtime::Buffer<float>& selected_jacobians_y,
    const SimilarityTransform& transform,
    int image_width,
    int image_height,
    Halide::Runtime::Buffer<double>& output) // Output buffer for the 4 update parameters
{
    // Ensure the output buffer has the correct size (1 dimension, extent 4)
    if (output.dimensions() != 1 || output.dim(0).extent() != 4)
    {
        // std::cout << "Resizing ICA result buffer" << std::endl;
        output = Halide::Runtime::Buffer<double>(4);
    }

    // Call the Halide-generated sparse_ica function
    int r = sparse_ica(
        input_template,
        input_keyframe,
        selected_pixels_x,
        selected_pixels_y,
        selected_jacobians_x,
        selected_jacobians_y,
        static_cast<float>(transform.A), // Pass transform parameters
        static_cast<float>(transform.B),
        static_cast<float>(transform.TX),
        static_cast<float>(transform.TY),
        image_width,                      // Pass image dimensions
        image_height,
        output);                          // Pass the output buffer

    if (r != 0) {
        std::cerr << "Halide sparse_ica call failed with error code: " << r << std::endl;
    }
    return r == 0;
}

bool SparseWarpDiff(
    Halide::Runtime::Buffer<uint8_t>& input_template,
    Halide::Runtime::Buffer<uint8_t>& input_keyframe,
    Halide::Runtime::Buffer<uint16_t>& local_max, // Input specifying pixel locations
    const SimilarityTransform& transform,
    int image_width,
    int image_height,
    Halide::Runtime::Buffer<uint16_t>& output) // Output buffer for differences
{
    // Ensure output buffer dimensions match the tile dimensions from local_max
    // Dimensions: (tile_x_count, tile_y_count)
    if (output.dimensions() != 2 ||
        output.dim(0).extent() != local_max.dim(0).extent() ||
        output.dim(1).extent() != local_max.dim(1).extent())
    {
        // std::cout << "Resizing WarpDiff buffer: " << local_max.dim(0).extent() << "x" << local_max.dim(1).extent() << std::endl;
        output = Halide::Runtime::Buffer<uint16_t>(local_max.dim(0).extent(), local_max.dim(1).extent());
    }

    // Call the Halide-generated sparse_warpdiff function
    int r = sparse_warpdiff(
        input_template,
        input_keyframe,
        local_max,
        static_cast<float>(transform.A), // Pass transform parameters
        static_cast<float>(transform.B),
        static_cast<float>(transform.TX),
        static_cast<float>(transform.TY),
        image_width,                      // Pass image dimensions
        image_height,
        output);                          // Pass the output buffer

    if (r != 0) {
        std::cerr << "Halide sparse_warpdiff call failed with error code: " << r << std::endl;
    }
    return r == 0;
}


bool PyrDown(
    Halide::Runtime::Buffer<uint8_t>& input,
    Halide::Runtime::Buffer<uint8_t>& output)
{
    // Ensure output buffer has correct dimensions (half of input)
    int out_width = input.width() / 2;
    int out_height = input.height() / 2;
     if (output.dimensions() != 2 || output.width() != out_width || output.height() != out_height) {
        output = Halide::Runtime::Buffer<uint8_t>(out_width, out_height);
    }

    int r = pyr_down(input, output);
     if (r != 0) {
        std::cerr << "Halide pyr_down call failed with error code: " << r << std::endl;
    }
    return r == 0;
}

bool ImageWarp(
    Halide::Runtime::Buffer<uint8_t>& input,
    const SimilarityTransform& transform,
    int image_width,
    int image_height,
    Halide::Runtime::Buffer<float>& output)
{
    // Ensure output buffer has the same dimensions as the input
    if (output.dimensions() != 2 || output.width() != input.width() || output.height() != input.height()) {
        output = Halide::Runtime::Buffer<float>(input.width(), input.height());
    }

    int r = image_warp(
        input,
        static_cast<float>(transform.A),
        static_cast<float>(transform.B),
        static_cast<float>(transform.TX),
        static_cast<float>(transform.TY),
        image_width,
        image_height,
        output);

    if (r != 0) {
        std::cerr << "Halide image_warp call failed with error code: " << r << std::endl;
    }
    return r == 0;
}


bool GradXY(
    Halide::Runtime::Buffer<uint8_t>& input,
    Halide::Runtime::Buffer<float>& output_x,
    Halide::Runtime::Buffer<float>& output_y)
{
    // Ensure output buffers have the same dimensions as the input
     if (output_x.dimensions() != 2 || output_x.width() != input.width() || output_x.height() != input.height()) {
        output_x = Halide::Runtime::Buffer<float>(input.width(), input.height());
    }
    if (output_y.dimensions() != 2 || output_y.width() != input.width() || output_y.height() != input.height()) {
        output_y = Halide::Runtime::Buffer<float>(input.width(), input.height());
    }

    int r = grad_xy(input, output_x, output_y);
     if (r != 0) {
        std::cerr << "Halide grad_xy call failed with error code: " << r << std::endl;
    }
    return r == 0;
}

bool GradArgMax(
    Halide::Runtime::Buffer<float>& grad_x,
    Halide::Runtime::Buffer<float>& grad_y,
    int& tile_size, // Output parameter
    Halide::Runtime::Buffer<uint16_t>& local_max_x, // Output buffer
    Halide::Runtime::Buffer<uint16_t>& local_max_y) // Output buffer
{
    const int min_tiles_threshold = 100; // Minimum number of tiles desired
    const int max_allowed_tile_size = 20; // Max tile size supported by generators
    const int min_allowed_tile_size = 2;  // Min tile size

    // Determine optimal tile size dynamically
    tile_size = min_allowed_tile_size; // Start with smallest
    for (int current_ts = min_allowed_tile_size + 2; current_ts <= max_allowed_tile_size; current_ts += 2) {
        int tiles_x = grad_x.width() / current_ts;
        int tiles_y = grad_y.height() / current_ts; // Use grad_y height assuming it's same as grad_x
        if (tiles_x * tiles_y < min_tiles_threshold) {
            break; // Stop increasing tile size if tile count drops too low
        }
        tile_size = current_ts; // Accept this tile size
    }

    // Calculate the number of tiles based on the final tile_size
    int width_tiles = grad_x.width() / tile_size;
    int height_tiles = grad_y.height() / tile_size; // Use grad_y height

    // Ensure output buffers have the correct dimensions: (width_tiles, height_tiles, 2)
    if (local_max_x.dimensions() != 3 ||
        local_max_x.dim(0).extent() != width_tiles ||
        local_max_x.dim(1).extent() != height_tiles ||
        local_max_x.dim(2).extent() != 2)
    {
        // std::cout << "Resizing ArgMax buffers: " << width_tiles << "x" << height_tiles << " (Tile size: " << tile_size << ")" << std::endl;
        local_max_x = Halide::Runtime::Buffer<uint16_t>(width_tiles, height_tiles, 2);
    }
     if (local_max_y.dimensions() != 3 ||
        local_max_y.dim(0).extent() != width_tiles ||
        local_max_y.dim(1).extent() != height_tiles ||
        local_max_y.dim(2).extent() != 2)
    {
        local_max_y = Halide::Runtime::Buffer<uint16_t>(width_tiles, height_tiles, 2);
    }


    // Call the appropriate Halide generator based on tile_size
    int r = -1;
    // Ensure tile_size matches the generator parameter exactly
    if (tile_size >= 19) { // Covers 19, 20
        r = grad_argmax_20(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 17) { // Covers 17, 18
        r = grad_argmax_18(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 15) { // Covers 15, 16
        r = grad_argmax_16(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 13) { // Covers 13, 14
        r = grad_argmax_14(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 11) { // Covers 11, 12
        r = grad_argmax_12(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 9) { // Covers 9, 10
        r = grad_argmax_10(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 7) { // Covers 7, 8
        r = grad_argmax_8(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 5) { // Covers 5, 6
        r = grad_argmax_6(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 3) { // Covers 3, 4
        r = grad_argmax_4(grad_x, grad_y, local_max_x, local_max_y);
    } else { // Covers 1, 2
        r = grad_argmax_2(grad_x, grad_y, local_max_x, local_max_y);
    }

    if (r != 0) {
         std::cerr << "Halide grad_argmax_" << tile_size << " call failed with error code: " << r << std::endl;
        return false; // The underlying grad_argmax_xx(...) call failed
    }

    return true;
}


// --- Point and SimilarityTransform Implementations ---

double Point::distance(const Point& p) const {
    double dx = x - p.x;
    double dy = y - p.y;
    return std::sqrt(dx * dx + dy * dy);
}

std::string SimilarityTransform::toString() const {
    std::stringstream ss;
    ss << "A=" << A << ", B=" << B << ", TX=" << TX << ", TY=" << TY;
    return ss.str();
}

// Applies the center-based warp
Point SimilarityTransform::warp(Point p, double Cx, double Cy) const {
    Point W;
    double px_c = p.x - Cx; // Translate point relative to center
    double py_c = p.y - Cy;
    // Apply rotation/scale around origin (which is now the center)
    W.x = (1.0 + A) * px_c - B * py_c + Cx + TX; // Translate back and add global T
    W.y = B * px_c + (1.0 + A) * py_c + Cy + TY;
    return W;
}

// Calculates the inverse of the center-based transform
SimilarityTransform SimilarityTransform::inverse(double Cx, double Cy) const
{
    SimilarityTransform Tinv;
    double p = 1.0 + A;
    double q = B;
    double denom = p * p + q * q;

    if (std::abs(denom) < 1e-10) {
         // Degenerate case: scale is zero. Return identity or handle appropriately.
         std::cerr << "Warning: Attempting to invert a degenerate SimilarityTransform (zero scale)." << std::endl;
         return SimilarityTransform(); // Return identity
    }

    // Inverse rotation/scale matrix R_inv parameters
    double p_inv = p / denom;  // 1 + A_inv
    double q_inv = -q / denom; // B_inv
    Tinv.A = p_inv - 1.0;
    Tinv.B = q_inv;

    // Inverse translation T_inv = C - R_inv*C - R_inv*T
    // R_inv * C
    double r_inv_cx = p_inv * Cx - q_inv * Cy;
    double r_inv_cy = q_inv * Cx + p_inv * Cy;
    // R_inv * T
    double r_inv_tx = p_inv * TX - q_inv * TY;
    double r_inv_ty = q_inv * TX + p_inv * TY;

    Tinv.TX = Cx - r_inv_cx - r_inv_tx;
    Tinv.TY = Cy - r_inv_cy - r_inv_ty;

    return Tinv;
}


// Composes two center-based transforms: T3(p) = w2( this(p) )
SimilarityTransform SimilarityTransform::compose(const SimilarityTransform &w2) const
{
    // "this" is T1 (A, B, TX, TY)
    // w2 is T2 (w2.A, w2.B, w2.TX, w2.TY)
    SimilarityTransform T3;

    double p1 = 1.0 + A;
    double q1 = B;
    double p2 = 1.0 + w2.A;
    double q2 = w2.B;

    // Composition of rotation/scale part: R3 = R2 * R1
    double p3 = p2 * p1 - q2 * q1; // 1 + A3
    double q3 = p2 * q1 + q2 * p1; // B3
    T3.A = p3 - 1.0;
    T3.B = q3;

    // Composition of translation part: T3 = T2 + R2 * T1
    // Apply R2 to vector T1 = (TX, TY)
    double r2_tx1 = p2 * TX - q2 * TY; // (1+A2)*TX - B2*TY
    double r2_ty1 = q2 * TX + p2 * TY; // B2*TX + (1+A2)*TY

    T3.TX = w2.TX + r2_tx1;
    T3.TY = w2.TY + r2_ty1;

    return T3;
}

double SimilarityTransform::maxCornerDisplacement(double width, double height) const {
    double Cx = width / 2.0;
    double Cy = height / 2.0;
    Point ul_pt{0.0, 0.0};
    Point ur_pt{width - 1.0, 0.0}; // Use width-1, height-1 for corners
    Point ll_pt{0.0, height - 1.0};
    Point lr_pt{width - 1.0, height - 1.0};

    Point ul_w = warp(ul_pt, Cx, Cy);
    Point ur_w = warp(ur_pt, Cx, Cy);
    Point ll_w = warp(ll_pt, Cx, Cy);
    Point lr_w = warp(lr_pt, Cx, Cy);

    double dist_ul = ul_pt.distance(ul_w);
    double dist_ur = ur_pt.distance(ur_w);
    double dist_ll = ll_pt.distance(ll_w);
    double dist_lr = lr_pt.distance(lr_w);

    return std::max({dist_ul, dist_ur, dist_ll, dist_lr});
}


// --- Buffer Conversion ---

Halide::Runtime::Buffer<uint8_t> mat_to_halide_buffer_u8(const cv::Mat &mat)
{
    if (mat.type() != CV_8UC1) {
        // Allow automatic conversion if possible, otherwise throw
        cv::Mat gray_mat;
        if (mat.channels() == 3) {
            cv::cvtColor(mat, gray_mat, cv::COLOR_BGR2GRAY);
        } else if (mat.channels() == 4) {
             cv::cvtColor(mat, gray_mat, cv::COLOR_BGRA2GRAY);
        } else {
             throw std::runtime_error("Input cv::Mat must be convertible to 8-bit single-channel (grayscale) image.");
        }
         return mat_to_halide_buffer_u8(gray_mat); // Recursive call with grayscale
    }

    cv::Mat continuous_mat = mat;
    if (!mat.isContinuous()) {
        continuous_mat = mat.clone();
        if (!continuous_mat.isContinuous()) {
            throw std::runtime_error("Failed to create a continuous copy of the input cv::Mat.");
        }
    }

    // Halide dimension strides are in elements, not bytes
    halide_dimension_t shape[2] = {
        {0, continuous_mat.cols, 1}, // Stride in x is 1 element
        {0, continuous_mat.rows, continuous_mat.cols} // Stride in y is 'width' elements
    };

    Halide::Runtime::Buffer<uint8_t> halide_buffer(
        continuous_mat.data, // Pointer to the data
        2,                   // Dimensions
        shape                // Shape and strides
    );

    // Workaround for potential cv::Mat lifetime issues if the original wasn't continuous.
    // This adds the Mat to the Halide buffer's context, keeping it alive.
    // Only needed if continuous_mat might be different from mat.
    if (continuous_mat.data != mat.data) {
        halide_buffer.device_wrap_native(halide_buffer.raw_buffer()->device_interface, reinterpret_cast<uint64_t>(new cv::Mat(continuous_mat)));
        halide_buffer.set_device_dirty();
        // Note: Halide::Runtime::Buffer in the current Halide version does not
        // provide add_destructor(). If lifetime management is needed, it can
        // be handled externally, but for typical short‑lived buffers we skip
        // explicit destructor hooks to maintain compatibility.
    }


    return halide_buffer;
}

Halide::Runtime::Buffer<uint8_t> bgr_mat_to_halide_buffer_u8(const cv::Mat &mat)
{
    if (mat.type() != CV_8UC3) {
         throw std::runtime_error("Input cv::Mat must be an 8-bit 3-channel (BGR) image.");
    }

    cv::Mat continuous_mat = mat;
    if (!mat.isContinuous()) {
        continuous_mat = mat.clone();
         if (!continuous_mat.isContinuous()) {
            throw std::runtime_error("Failed to create a continuous copy of the input cv::Mat.");
        }
    }

    // Halide expects planar or interleaved depending on the target/schedule.
    // Assuming interleaved common format: width, height, channels
    halide_dimension_t shape[3] = {
        {0, continuous_mat.cols, 3}, // Stride in x is 3 (channels)
        {0, continuous_mat.rows, continuous_mat.cols * 3}, // Stride in y is width * channels
        {0, 3, 1}                     // Stride in c is 1
    };

    Halide::Runtime::Buffer<uint8_t> halide_buffer(
        continuous_mat.data,
        3,
        shape
    );

    // Add lifetime management if needed (similar to grayscale version)
     if (continuous_mat.data != mat.data) {
        halide_buffer.device_wrap_native(halide_buffer.raw_buffer()->device_interface, reinterpret_cast<uint64_t>(new cv::Mat(continuous_mat)));
        halide_buffer.set_device_dirty();
    }

    return halide_buffer;
}


cv::Mat halide_buffer_to_mat(const Halide::Runtime::Buffer<uint8_t> &buffer) {
    if (buffer.dimensions() != 2) {
        throw std::runtime_error("halide_buffer_to_mat (uint8_t): Only 2-dimensional Halide buffers supported.");
    }

    int width = buffer.width();
    int height = buffer.height();
    size_t stride_bytes = buffer.dim(1).stride() * sizeof(uint8_t); // Stride for the y-dimension in bytes

    // Create a cv::Mat header. Important: This does NOT copy data.
    // The cv::Mat will be invalid if the Halide buffer is destroyed or reallocated.
    cv::Mat mat(height, width, CV_8UC1, buffer.data(), stride_bytes);

    // If you need the cv::Mat to own the data, clone it:
    // return mat.clone();
    return mat; // Returning header only
}

cv::Mat halide_buffer_to_mat(const Halide::Runtime::Buffer<float> &buffer) {
    if (buffer.dimensions() != 2) {
        throw std::runtime_error("halide_buffer_to_mat (float): Only 2-dimensional Halide buffers supported.");
    }

    int width = buffer.width();
    int height = buffer.height();
    size_t stride_bytes = buffer.dim(1).stride() * sizeof(float);

    cv::Mat mat(height, width, CV_32FC1, buffer.data(), stride_bytes);

    // return mat.clone(); // Clone if ownership is needed
    return mat;
}

cv::Mat halide_vec4_to_mat(const Halide::Runtime::Buffer<double> &vec4)
{
    if (vec4.dimensions() != 1 || vec4.dim(0).extent() != 4) {
        throw std::runtime_error("halide_vec4_to_mat: Expected a 1D Halide buffer of length 4");
    }

    // Create a 4x1 CV_64F (double) matrix
    cv::Mat cv_vec(4, 1, CV_64F);

    // Copy data using memcpy for efficiency
    memcpy(cv_vec.data, vec4.data(), 4 * sizeof(double));

    // Or element-by-element:
    // for (int i = 0; i < 4; i++) {
    //     cv_vec.at<double>(i, 0) = vec4(i);
    // }

    return cv_vec;
}


// --- OpenCV Warping ---

// Warps src using the center-based SimilarityTransform via OpenCV
cv::Mat warpBySimilarityTransform(const cv::Mat& src, const SimilarityTransform& transform)
{
    if (src.empty()) {
        throw std::runtime_error("warpBySimilarityTransform: Input image is empty.");
    }

    double Cx = static_cast<double>(src.cols) / 2.0;
    double Cy = static_cast<double>(src.rows) / 2.0;

    // We need the inverse transform matrix M for warpAffine:
    // p_src = R_inv * p_dst + T_inv'
    // M = [ R_inv | T_inv' ]
    // where T_inv' = C - R_inv*C - R_inv*T

    SimilarityTransform T_inv = transform.inverse(Cx, Cy);

    // Build the 2x3 affine matrix M = [ R_inv | T_inv ] directly from T_inv
    cv::Mat M = (cv::Mat_<double>(2, 3) <<
                 1.0 + T_inv.A,  -T_inv.B,      T_inv.TX,
                 T_inv.B,        1.0 + T_inv.A, T_inv.TY);

    cv::Mat dst;
    cv::Size outputSize(src.cols, src.rows);

    // Use a good interpolation method like Lanczos4 if available, otherwise Linear.
    // cv::WARP_INVERSE_MAP is the default behavior, so it's not needed explicitly.
    int flags = cv::INTER_LANCZOS4;
    // int flags = cv::INTER_LINEAR;

    cv::warpAffine(
        src,
        dst,
        M,                   // The inverse transform matrix
        outputSize,
        flags,               // Interpolation flags
        cv::BORDER_CONSTANT, // How to handle pixels outside the source bounds
        cv::Scalar()         // Border value (e.g., black)
    );

    return dst;
}
