#include "imgproc.hpp"

#include <sstream>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <vector>

#include "sparse_jac.h"
#include "pyr_down.h"
#include "image_warp.h"
#include "grad_xy.h"
#include "grad_argmax_20.h"
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

bool SparseJacobian(
    Halide::Runtime::Buffer<float>& grad_x,
    Halide::Runtime::Buffer<float>& grad_y,
    Halide::Runtime::Buffer<uint16_t>& local_max_x,
    Halide::Runtime::Buffer<uint16_t>& local_max_y,
    Halide::Runtime::Buffer<float>& output_x,
    Halide::Runtime::Buffer<float>& output_y)
{
    if (output_x.dimensions() != 3 ||
        output_x.dim(0).extent() != local_max_x.dim(0).extent() ||
        output_x.dim(1).extent() != local_max_x.dim(1).extent())
    {
        output_x = Halide::Runtime::Buffer<float>(local_max_x.dim(0).extent(), local_max_x.dim(1).extent(), 4);
        output_y = Halide::Runtime::Buffer<float>(local_max_x.dim(0).extent(), local_max_x.dim(1).extent(), 4);
    }

    int r = sparse_jac(grad_x, grad_y, local_max_x, local_max_y, output_x, output_y);
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
    Halide::Runtime::Buffer<double>& output)
{
    if (output.dimensions() != 1 ||
        output.dim(0).extent() != 4)
    {
        output = Halide::Runtime::Buffer<double>(4);
    }

    int r = sparse_ica(
        input_template,
        input_keyframe,
        selected_pixels_x,
        selected_pixels_y,
        selected_jacobians_x,
        selected_jacobians_y,
        static_cast<float>(transform.A),
        static_cast<float>(transform.B),
        // Convert center-based translation to UL-based for the kernel
        static_cast<float>(transform.TX - transform.A * (input_template.width()*0.5f) +
                                              transform.B * (input_template.height()*0.5f)),
        static_cast<float>(transform.TY - transform.B * (input_template.width()*0.5f) -
                                              transform.A * (input_template.height()*0.5f)),
        output);
    return r == 0;
}

bool SparseWarpDiff(
    Halide::Runtime::Buffer<uint8_t>& input_template,
    Halide::Runtime::Buffer<uint8_t>& input_keyframe,
    Halide::Runtime::Buffer<uint16_t>& local_max,
    const SimilarityTransform& transform,
    Halide::Runtime::Buffer<uint16_t>& output)
{
    if (output.dimensions() != 2 ||
        output.dim(0).extent() != local_max.dim(0).extent() ||
        output.dim(1).extent() != local_max.dim(1).extent())
    {
        output = Halide::Runtime::Buffer<uint16_t>(local_max.dim(0).extent(), local_max.dim(1).extent());
    }

    int r = sparse_warpdiff(
        input_template,
        input_keyframe,
        local_max,
        static_cast<float>(transform.A),
        static_cast<float>(transform.B),
        static_cast<float>(transform.TX - transform.A * (input_template.width()*0.5f) +
                                          transform.B * (input_template.height()*0.5f)),
        static_cast<float>(transform.TY - transform.B * (input_template.width()*0.5f) -
                                          transform.A * (input_template.height()*0.5f)),
        output);
    return r == 0;
}

bool PyrDown(
    Halide::Runtime::Buffer<uint8_t>& input,
    Halide::Runtime::Buffer<uint8_t>& output)
{
    int r = pyr_down(input, output);
    return r == 0;
}

bool ImageWarp(
    Halide::Runtime::Buffer<uint8_t>& input,
    const SimilarityTransform& transform,
    Halide::Runtime::Buffer<float>& output)
{
    // The Halide-generated kernel expects the translation component to be
    // specified with respect to the origin (upper-left).  Convert the provided
    // center-based (TX,TY) into the equivalent origin-based translation.

    double cx = (input.width()  - 1) * 0.5; // width and height are at least 1
    double cy = (input.height() - 1) * 0.5;

    double tx_ul = transform.TX - transform.A * cx + transform.B * cy;
    double ty_ul = transform.TY - transform.B * cx - transform.A * cy;

    int r = image_warp(input, transform.A, transform.B, tx_ul, ty_ul, output);
    return r == 0;
}

bool GradXY(
    Halide::Runtime::Buffer<uint8_t>& input,
    Halide::Runtime::Buffer<float>& output_x,
    Halide::Runtime::Buffer<float>& output_y)
{
    int r = grad_xy(input, output_x, output_y);
    return r == 0;
}

bool GradArgMax(
    Halide::Runtime::Buffer<float>& grad_x,
    Halide::Runtime::Buffer<float>& grad_y,
    int& tile_size,
    Halide::Runtime::Buffer<uint16_t>& local_max_x,
    Halide::Runtime::Buffer<uint16_t>& local_max_y)
{
    const int min_tiles = 1000;
    const int max_tile_size = 20; // even numbers between 2 and 20

    tile_size = 2;
    for (int i = 4; i <= max_tile_size; i += 2) {
        int tx = grad_x.width() / i;
        int ty = grad_y.height() / i;
        if (tx * ty < min_tiles) {
            break;
        }
        tile_size = i;
    }

    int width_tiles = grad_x.width() / tile_size;
    int height_tiles = grad_y.height() / tile_size;
    if (local_max_x.dimensions() != 3 ||
        local_max_x.dim(0).extent() != width_tiles ||
        local_max_x.dim(1).extent() != height_tiles)
    {
        local_max_x = Halide::Runtime::Buffer<uint16_t>(width_tiles, height_tiles, 2);
        local_max_y = Halide::Runtime::Buffer<uint16_t>(width_tiles, height_tiles, 2);
    }

    int r = -1;
    if (tile_size >= 19 /* 19-20*/) {
        r = grad_argmax_20(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 17 /* 17-18*/) {
        r = grad_argmax_18(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 15 /* 15-16*/) {
        r = grad_argmax_16(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 13 /* 13-14*/) {
        r = grad_argmax_14(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 11 /* 11-12*/) {
        r = grad_argmax_12(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 9 /* 9-10*/) {
        r = grad_argmax_10(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 7 /* 7-8*/) {
        r = grad_argmax_8(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 5 /* 5-6*/) {
        r = grad_argmax_6(grad_x, grad_y, local_max_x, local_max_y);
    } else if (tile_size >= 3 /* 3-4*/) {
        r = grad_argmax_4(grad_x, grad_y, local_max_x, local_max_y);
    } else { /* 1-2 */
        r = grad_argmax_2(grad_x, grad_y, local_max_x, local_max_y);
    }

    if (r != 0) {
        return false; // The underlying grad_argmax_xx(...) call failed
    }

    return true;
}

Halide::Runtime::Buffer<uint8_t> mat_to_halide_buffer_u8(const cv::Mat &mat)
{
    // Ensure the input is a single-channel 8-bit image
    if (mat.type() != CV_8UC1) {
        throw std::runtime_error("Input cv::Mat must be an 8-bit single-channel (grayscale) image.");
    }

    // Ensure the image data is stored continuously in memory
    cv::Mat continuous_mat = mat;
    if (!mat.isContinuous()) {
        // Create a continuous copy of the matrix
        continuous_mat = mat.clone();
        if (!continuous_mat.isContinuous()) {
            throw std::runtime_error("Failed to create a continuous copy of the input cv::Mat.");
        }
    }

    halide_dimension_t shape[2] = {
        {0, continuous_mat.cols, (int32_t)(continuous_mat.step1(1) / sizeof(uint8_t))},
        {0, continuous_mat.rows, (int32_t)(continuous_mat.step1(0) / sizeof(uint8_t))},
    };

    // Create the Halide buffer with the appropriate dimensions
    Halide::Runtime::Buffer<uint8_t> halide_buffer(
        continuous_mat.data, // Pointer to the image data
        2,
        shape
    );

    return halide_buffer;
}

Halide::Runtime::Buffer<uint8_t> bgr_mat_to_halide_buffer_u8(const cv::Mat &mat)
{
    // Ensure the input is a 3-channel 8-bit image
    if (mat.type() != CV_8UC3) {
        throw std::runtime_error("Input cv::Mat must be an 8-bit 3-channel (BGR) image.");
    }

    // Ensure the image data is stored continuously in memory
    cv::Mat continuous_mat = mat;
    if (!mat.isContinuous()) {
        // Create a continuous copy of the matrix
        continuous_mat = mat.clone();
        if (!continuous_mat.isContinuous()) {
            throw std::runtime_error("Failed to create a continuous copy of the input cv::Mat.");
        }
    }

    halide_dimension_t shape[3] = {
        {0, continuous_mat.cols, (int32_t)(continuous_mat.step1(1) / 3)},
        {0, continuous_mat.rows, (int32_t)(continuous_mat.step1(0) / 3)},
        {0, 3, (int32_t)3}
    };

    // Create the Halide buffer with the appropriate dimensions
    Halide::Runtime::Buffer<uint8_t> halide_buffer(
        continuous_mat.data, // Pointer to the image data
        3,
        shape
    );

    return halide_buffer;
}

cv::Mat halide_buffer_to_mat(const Halide::Runtime::Buffer<uint8_t> &buffer) {
    // Ensure the buffer is 2-dimensional
    if (buffer.dimensions() != 2) {
        throw std::runtime_error("Only 2-dimensional Halide buffers can be converted to cv::Mat.");
    }

    // Get buffer dimensions
    int width = buffer.width();
    int height = buffer.height();

    // Compute strides in bytes
    size_t stride_y = buffer.stride(1) * sizeof(uint8_t); // Bytes per row

    // Create a cv::Mat header that wraps the Halide buffer's data
    // Note: cv::Mat does not take ownership of the data; ensure the buffer remains valid.
    cv::Mat mat(height, width, CV_8UC1, (void*)buffer.data(), stride_y);

    return mat;
}

cv::Mat halide_buffer_to_mat(const Halide::Runtime::Buffer<float> &buffer) {
    // Ensure the buffer is 2-dimensional
    if (buffer.dimensions() != 2) {
        throw std::runtime_error("Only 2-dimensional Halide buffers can be converted to cv::Mat.");
    }

    // Get buffer dimensions
    int width = buffer.width();
    int height = buffer.height();

    // Compute strides in bytes
    size_t stride_y = buffer.stride(1) * sizeof(float);

    // Create a cv::Mat header that wraps the Halide buffer's data
    // Note: cv::Mat does not take ownership of the data; ensure the buffer remains valid.
    cv::Mat mat(height, width, CV_32FC1, (void*)buffer.data(), stride_y);

    return mat;
}

cv::Mat halide_vec4_to_mat(const Halide::Runtime::Buffer<double> &vec4)
{
    // Check that vec4 is a 1D buffer with extent == 4
    if (vec4.dimensions() != 1 || vec4.width() != 4) {
        throw std::runtime_error("Expected a 1D Halide buffer of length 4");
    }

    // Create a 4x1 float matrix in OpenCV
    cv::Mat cv_vec(4, 1, CV_64F);

    // Copy values over
    for (int i = 0; i < 4; i++) {
        cv_vec.at<double>(i, 0) = vec4(i);
    }

    return cv_vec; // 4x1 column vector
}

std::string SimilarityTransform::toString() const {
    std::stringstream ss;
    ss << "A=" << A << ", B=" << B << ", TX=" << TX << ", TY=" << TY;
    return ss.str();
}

SimilarityTransform SimilarityTransform::inverse() const
{
    // The forward matrix is M = [[p, -q], [q, p]], with p=(1+A), q=B
    // We want M^-1 plus the translation that inverts (TX,TY).  In the
    // center-pivot parameterization, the translation component t = (TX,TY)
    // is applied *after* the rotation about the center, so the inverse just
    // needs to apply the inverse rotation to -t.
    double p = 1.0 + A;   // (1 + A)
    double q = B;
    double denom = p * p + q * q; // (p^2 + q^2)

    // (1 + A_inv) = p / denom
    // B_inv       = - q / denom

    SimilarityTransform Tinv;
    Tinv.A  = (p / denom) - 1.0;   // (1 + A_inv) = p/denom
    Tinv.B  = -q / denom;          // B_inv = -q/denom

    // Translation: t_inv = - R_inv * t
    double invX = (-p * TX - q * TY) / denom;
    double invY = ( q * TX - p * TY) / denom;

    Tinv.TX = invX;
    Tinv.TY = invY;

    return Tinv;
}

SimilarityTransform SimilarityTransform::compose(const SimilarityTransform &w2) const
{
    // "this" is T1, param w2 is T2.
    // T3 = T2 ◦ T1 => T3(p) = T2( T1(p) )
    // => (1 + A3) = (1 + A2)*(1 + A1) - B2*B1
    //    B3       = (1 + A2)*B1 + B2*(1 + A1)
    //    TX3      = (1 + A2)*TX1 - B2*TY1 + TX2
    //    TY3      = B2*TX1 + (1 + A2)*TY1 + TY2

    double p1 = 1.0 + A;
    double q1 = B;
    double p2 = 1.0 + w2.A;
    double q2 = w2.B;

    double A3  = (p2 * p1 - q2 * q1) - 1.0;
    double B3  = (p2 * q1 + q2 * p1);
    double TX3 = p2 * TX  - q2 * TY + w2.TX;
    double TY3 = q2 * TX  + p2 * TY + w2.TY;

    SimilarityTransform T3;
    T3.A  = A3;
    T3.B  = B3;
    T3.TX = TX3;
    T3.TY = TY3;

    return T3;
}

Point SimilarityTransform::warp(Point p) const {
    Point W;
    W.x = (1 + A) * p.x - B * p.y + TX;
    W.y = B * p.x + (1 + A) * p.y + TY;
    return W;
}

// Warp with an explicit center of rotation (cx, cy). The parameters (A, B)
// represent scale+rotation exactly as in warp(), but the rotation is applied
// about the point (cx, cy) rather than the origin. After rotation/scale, an
// additional translation (TX, TY) is applied. When (cx, cy) == (0,0) this is
// equivalent to warp().
Point SimilarityTransform::warp(Point p, double cx, double cy) const {
    // Translate to center, apply rotation+scale, translate back, then translate
    // by (TX, TY).
    double px = p.x - cx;
    double py = p.y - cy;

    Point W;
    W.x = (1 + A) * px - B * py + cx + TX;
    W.y = B * px + (1 + A) * py + cy + TY;
    return W;
}

double Point::distance(const Point& p) const {
    double dx = x - p.x;
    double dy = y - p.y;
    return std::sqrt(dx * dx + dy * dy);
}

double SimilarityTransform::maxCornerDisplacement(double width, double height) const {
    // Compute center of the image. Corners will be rotated about this point.
    double cx = width  * 0.5;
    double cy = height * 0.5;

    Point ul{0.0,    0.0};
    Point ur{width,  0.0};
    Point ll{0.0,    height};
    Point lr{width,  height};

    // Find the maximum distance any corner moves.
    double max_d = 0.0;
    max_d = std::max(max_d, warp(ul, cx, cy).distance(ul));
    max_d = std::max(max_d, warp(ur, cx, cy).distance(ur));
    max_d = std::max(max_d, warp(ll, cx, cy).distance(ll));
    max_d = std::max(max_d, warp(lr, cx, cy).distance(lr));

    return max_d;
}

/**
 * \brief Warp a color image by the given SimilarityTransform using OpenCV.
 *
 * \param src       [in]  The input color image (CV_8UC3, etc.).
 * \param transform [in]  The similarity transform parameters.
 * \return               The warped output image.
 */
cv::Mat warpBySimilarityTransform(const cv::Mat& src, const SimilarityTransform& transform)
{
    // Build the forward affine matrix from (A, B, TX, TY).
    // Forward transform (x_out, y_out) = M * (x_in, y_in, 1)
    //   where M is a 2x3 matrix:
    //
    //   M = [ 1 + A,   -B,     TX ]
    //       [   B,   1 + A,   TY ]
    //
    // Because warpAffine() by default expects an inverse mapping,
    // we will enable the WARP_INVERSE_MAP flag below so it interprets M as forward.
    // Convert (TX,TY) from center-based to origin-based for warpAffine.
    double cx = (src.cols - 1) * 0.5;
    double cy = (src.rows - 1) * 0.5;

    double tx_ul = transform.TX - transform.A * cx + transform.B * cy;
    double ty_ul = transform.TY - transform.B * cx - transform.A * cy;

    cv::Mat M = (cv::Mat_<double>(2, 3) <<
                 1.0 + transform.A,  -transform.B,    tx_ul,
                 transform.B,        1.0 + transform.A, ty_ul);

    cv::Mat dst;
    // Use the same size as the source for the output. Adjust as needed.
    cv::Size outputSize(src.cols, src.rows);

    int flags = cv::INTER_LINEAR/* | cv::WARP_INVERSE_MAP*/; // Combine them with bitwise OR
    cv::warpAffine(
        src,
        dst,
        M,
        outputSize,
        flags,               // pass combined flags here
        cv::BORDER_CONSTANT, // border mode
        cv::Scalar()         // border value
    );

    return dst;
}
