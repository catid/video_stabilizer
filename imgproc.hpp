#pragma once

#include <Halide.h>
#include <opencv2/opencv.hpp>

#include "constants.h" // Assuming this defines constants if needed

// Forward declare if needed, or include full definition
struct Point {
    double x = 0.0, y = 0.0;
    double distance(const Point& p) const;
};

struct SimilarityTransform {
    // Parameters for center-based warp:
    // Wx = (1+A)*(x - Cx) - B*(y - Cy) + Cx + TX
    // Wy = B*(x - Cx) + (1+A)*(y - Cy) + Cy + TY
    double A = 0.0, B = 0.0, TX = 0.0, TY = 0.0;

    std::string toString() const;

    // Computes the inverse transform. Requires image center.
    // If Cx,Cy are not provided, assume origin-centered at (0,0) for backward compatibility.
    SimilarityTransform inverse(double Cx = 0.0, double Cy = 0.0) const;

    // Computes composition: T3(p) = w2( this(p) )
    SimilarityTransform compose(const SimilarityTransform &w2) const;

    // Applies the center-based warp to a point.
    Point warp(Point p, double Cx, double Cy) const;

    // Convenience overload that assumes the center is at the origin (0,0).
    // This is primarily useful for unit tests or callers that wish to retain
    // the previous behaviour where rotations were taken about (0,0).
    // NOTE: Production code that works with actual image coordinates should
    //       prefer the three‑parameter version and pass the real image
    //       centre to avoid subtle accuracy issues.
    inline Point warp(Point p) const {
        return warp(p, 0.0, 0.0);
    }

    // Calculates max displacement of corners using center-based warp.
    double maxCornerDisplacement(double width, double height) const;
};


bool SparseJacobian(
    Halide::Runtime::Buffer<float>& grad_x,
    Halide::Runtime::Buffer<float>& grad_y,
    Halide::Runtime::Buffer<uint16_t>& local_max_x,
    Halide::Runtime::Buffer<uint16_t>& local_max_y,
    int image_width, // Added: Width of the gradient images
    int image_height, // Added: Height of the gradient images
    Halide::Runtime::Buffer<float>& output_x,
    Halide::Runtime::Buffer<float>& output_y);

bool PyrDown(
    Halide::Runtime::Buffer<uint8_t>& input,
    Halide::Runtime::Buffer<uint8_t>& output);

bool GradXY(
    Halide::Runtime::Buffer<uint8_t>& input,
    Halide::Runtime::Buffer<float>& output_x,
    Halide::Runtime::Buffer<float>& output_y);

// Returns floor(grad_x/tile_size) sized image with one (x,y) pair per tile,
// representing the maximum gradient magnitude position in pixels from image origin.
bool GradArgMax(
    Halide::Runtime::Buffer<float>& grad_x,
    Halide::Runtime::Buffer<float>& grad_y,
    int& tile_size, // Output: the determined tile size
    Halide::Runtime::Buffer<uint16_t>& local_max_x, // Output
    Halide::Runtime::Buffer<uint16_t>& local_max_y); // Output

// Applies the center-based similarity transform warp using Halide (bilinear).
bool ImageWarp(
    Halide::Runtime::Buffer<uint8_t>& input,
    const SimilarityTransform& transform,
    int image_width, // Added: Width of the input image
    int image_height, // Added: Height of the input image
    Halide::Runtime::Buffer<float>& output);

// Convenience overload: infers the image dimensions from the input buffer.
inline bool ImageWarp(
    Halide::Runtime::Buffer<uint8_t>& input,
    const SimilarityTransform& transform,
    Halide::Runtime::Buffer<float>& output)
{
    return ImageWarp(input, transform, input.width(), input.height(), output);
}

// Solve for parameter updates using sparse ICA based on center-warp model.
// Sum( jac[i] * ( template[i] - Warped{keyframe[i]} ) )
bool SparseICA(
    Halide::Runtime::Buffer<uint8_t>& input_template,
    Halide::Runtime::Buffer<uint8_t>& input_keyframe,
    Halide::Runtime::Buffer<uint16_t>& selected_pixels_x,
    Halide::Runtime::Buffer<uint16_t>& selected_pixels_y,
    Halide::Runtime::Buffer<float>& selected_jacobians_x,
    Halide::Runtime::Buffer<float>& selected_jacobians_y,
    const SimilarityTransform& transform,
    int image_width, // Added: Width of the images at this pyramid level
    int image_height, // Added: Height of the images at this pyramid level
    Halide::Runtime::Buffer<double>& output); // Output: 4-element update vector

// Calculates the difference between template and warped keyframe at sparse locations.
bool SparseWarpDiff(
    Halide::Runtime::Buffer<uint8_t>& input_template,
    Halide::Runtime::Buffer<uint8_t>& input_keyframe,
    Halide::Runtime::Buffer<uint16_t>& local_max, // Specifies pixel locations
    const SimilarityTransform& transform,
    int image_width, // Added: Width of the images
    int image_height, // Added: Height of the images
    Halide::Runtime::Buffer<uint16_t>& output); // Output: Difference values

// --- OpenCV / Halide Buffer Conversion ---
Halide::Runtime::Buffer<uint8_t> mat_to_halide_buffer_u8(const cv::Mat &mat);
Halide::Runtime::Buffer<uint8_t> bgr_mat_to_halide_buffer_u8(const cv::Mat &mat);
cv::Mat halide_buffer_to_mat(const Halide::Runtime::Buffer<uint8_t> &buffer);
cv::Mat halide_buffer_to_mat(const Halide::Runtime::Buffer<float> &buffer);
cv::Mat halide_vec4_to_mat(const Halide::Runtime::Buffer<double> &vec4);

// --- OpenCV Warping Utility ---
// Warps using OpenCV's warpAffine, calculating the correct matrix for center-based transform.
cv::Mat warpBySimilarityTransform(const cv::Mat& src, const SimilarityTransform& transform);
