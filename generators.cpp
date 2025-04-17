#include "generators_tools.h"
#include <Halide.h> // Ensure Halide header is included

using namespace Halide; // Use Halide namespace
using namespace Halide::BoundaryConditions; // Use BoundaryConditions namespace


////////////////////////////////////////////////////////////////////////////////

#if 0 /* baseline */

// A simple sinc function: sinc(x) = sin(pi*x)/(pi*x)
Expr sinc(Expr x) {
    // Avoid division-by-zero at x=0 by defining sinc(0)=1
    // (the correct limiting value)
    Expr pix = x * float(M_PI);
    Expr s   = fast_sin(pix) / pix;
    s        = select(x == 0.0f, 1.0f, s);
    return s;
}

// Standard Lanczos2 kernel has radius = 2
// lanczos2(x) = sinc(x) * sinc(x/2),  for |x| < 2
Expr lanczos2(Expr x) {
    // The raw product sinc(x)*sinc(x/2)
    Expr value = sinc(x) * sinc(x / 2.0f);

    // Force it to zero outside the [-2,2] window
    value = select(abs(x) >= 2.0f, 0.0f, value);

    return value;
}

#else /* 2.7x faster optimized by lanczos2_opt.cpp */

Expr lanczos2(Expr x)
{
    // We evaluate P(x) = a0 + a1*x^2 + a2*x^4 + ...
    // via Horner's method on x^2 for efficiency.
    Expr x2 = x * x;

    // Start from the highest coefficient:
    Expr val = 0.000858519f;                      // a6
    val = -0.0158853f + val * x2;                   // a5 + a6*x^2
    val = 0.128693f   + val * x2;                   // a4 + ...
    val = -0.583468f  + val * x2;                   // a3 + ...
    val = 1.52229f    + val * x2;                   // a2 + ...
    val = -2.05238f   + val * x2;                   // a1 + ...
    val = 0.999861f   + val * x2;                   // a0 + ...

    return select(abs(x) >= 2.0f, 0.0f, val);
}

#endif


////////////////////////////////////////////////////////////////////////////////

#include "schedules/pyr_down.schedule.h" // Assumed to exist

class pyr_down_generator : public Halide::Generator<pyr_down_generator>
{
public:
    GeneratorParam<int> expected_width{ "expected_width", 1920 }; // Added default
    GeneratorParam<int> expected_height{ "expected_height", 1080 }; // Added default

    // Input image: Monochrome (grayscale) 8-bit image
    Input<Buffer<uint8_t>> input{"input", 2};
    // Output image: Downsampled 8-bit monochrome image
    Output<Buffer<uint8_t>> output{"output", 2};

    Var x{"x"}, y{"y"}; // Declare loop variables

    void generate()
    {
        // Define the 1D Gaussian filter coefficients
        // [1/16, 4/16, 6/16, 4/16, 1/16]
        const float coeffs[5] = {1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16};

        Func input_bounded = BoundaryConditions::repeat_edge(input);

        // First pass: filter in Y direction
        Func blur_y("blur_y");
        blur_y(x, y) =
            (coeffs[0] * cast<float>(input_bounded(x, y-2)) +
             coeffs[1] * cast<float>(input_bounded(x, y-1)) +
             coeffs[2] * cast<float>(input_bounded(x, y)) +
             coeffs[3] * cast<float>(input_bounded(x, y+1)) +
             coeffs[4] * cast<float>(input_bounded(x, y+2)));

        // Second pass: filter in X direction
        Func blur_xy("blur_xy");
        blur_xy(x, y) =
            (coeffs[0] * cast<float>(blur_y(x-2, y)) +
             coeffs[1] * cast<float>(blur_y(x-1, y)) +
             coeffs[2] * cast<float>(blur_y(x, y)) +
             coeffs[3] * cast<float>(blur_y(x+1, y)) +
             coeffs[4] * cast<float>(blur_y(x+2, y)));

        // Downsample by taking every other pixel
        output(x, y) = cast<uint8_t>(blur_xy(x * 2, y * 2));
    }

    void schedule()
    {
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi"); // Declare scheduling vars

        input.dim(0).set_estimate(80, expected_width);
        input.dim(1).set_estimate(45, expected_height);
        output.dim(0).set_estimate(40, expected_width/2);
        output.dim(1).set_estimate(22, expected_height/2);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#else
        bool auto_schedule = false; // Assume no autoscheduler for older versions
#endif
        if (auto_schedule) {
            // Optional: Add specific constraints for the autoscheduler if needed
            // e.g., input.set_estimates(...); output.set_estimates(...);
            return; // Let the autoscheduler handle it
        }

        if (get_target().has_gpu_feature())
        {
            // Example GPU schedule (adjust tile sizes as needed)
            output.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
            // Schedule intermediate stages if beneficial
            // blur_y.compute_at(output, xo).gpu_threads(x, y);
            // blur_xy.compute_at(output, xo).gpu_threads(x, y);
        }
        else
        {
            // Assumes apply_schedule_pyr_down exists and is included
            apply_schedule_pyr_down(get_pipeline(), get_target());
        }
    }
};

HALIDE_REGISTER_GENERATOR(pyr_down_generator, pyr_down)

////////////////////////////////////////////////////////////////////////////////

#include "schedules/image_warp.schedule.h" // Assumed to exist

class image_warp_generator : public Halide::Generator<image_warp_generator>
{
public:
    GeneratorParam<int> expected_width{ "expected_width", 1920 }; // Added default
    GeneratorParam<int> expected_height{ "expected_height", 1080 }; // Added default

    // Input image: Monochrome or multi-channel image
    Input<Buffer<uint8_t>> input{"input", 2};
    // Transform parameters
    Input<float> A{"A"};
    Input<float> B{"B"};
    Input<float> TX{"TX"};
    Input<float> TY{"TY"};
    // Image dimensions (for center calculation)
    Input<int> image_width{"image_width"};
    Input<int> image_height{"image_height"};
    // Output image
    Output<Buffer<float>> output{"output", 2};

    Var x{"x"}, y{"y"}; // Declare loop variables

    void generate() {
        Expr Cx = cast<float>(image_width) / 2.0f;
        Expr Cy = cast<float>(image_height) / 2.0f;
        Expr px = cast<float>(x);
        Expr py = cast<float>(y);

        // Center-based Warp equations
        Expr W_x = (1.0f + A) * (px - Cx) - B * (py - Cy) + Cx + TX;
        Expr W_y = B * (px - Cx) + (1.0f + A) * (py - Cy) + Cy + TY;

        // Boundary condition
        Func clamped("clamped");
        clamped(x, y) = BoundaryConditions::repeat_edge(input)(x, y);

        // Compute the integer and fractional parts of the warp coordinates
        Expr W_x_floor = cast<int>(floor(W_x));
        Expr W_y_floor = cast<int>(floor(W_y));
        Expr wx_frac = W_x - cast<float>(W_x_floor); // Renamed to avoid conflict
        Expr wy_frac = W_y - cast<float>(W_y_floor); // Renamed to avoid conflict

        // Sample the four nearest neighbors
        Expr x0y0 = cast<float>( clamped(W_x_floor, W_y_floor) );
        Expr x1y0 = cast<float>( clamped(W_x_floor + 1, W_y_floor) );
        Expr x0y1 = cast<float>( clamped(W_x_floor, W_y_floor + 1) );
        Expr x1y1 = cast<float>( clamped(W_x_floor + 1, W_y_floor + 1) );

        // Bilinear interpolation
        Expr top = lerp(x0y0, x1y0, wx_frac);
        Expr bottom = lerp(x0y1, x1y1, wx_frac);
        output(x, y) = lerp(top, bottom, wy_frac);
    }

    void schedule()
    {
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi"); // Declare scheduling vars

        input.dim(0).set_estimate(80, expected_width);
        input.dim(1).set_estimate(45, expected_height);
        A.set_estimate(0.1f);
        B.set_estimate(0.1f);
        TX.set_estimate(1.f);
        TY.set_estimate(1.f);
        image_width.set_estimate(expected_width);
        image_height.set_estimate(expected_height);
        output.dim(0).set_estimate(80, expected_width);
        output.dim(1).set_estimate(45, expected_height);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#else
        bool auto_schedule = false;
#endif
        if (auto_schedule) {
            // input.set_estimates(...); // Provide estimates if helpful
            // output.set_estimates(...);
            return;
        }

        if (get_target().has_gpu_feature())
        {
            // Example GPU schedule
            output.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
        }
        else
        {
            // Assumes apply_schedule_image_warp exists and is included
            apply_schedule_image_warp(get_pipeline(), get_target());
        }
    }
};

HALIDE_REGISTER_GENERATOR(image_warp_generator, image_warp)

////////////////////////////////////////////////////////////////////////////////

#include "schedules/grad_xy.schedule.h" // Assumed to exist

class grad_xy_generator : public Halide::Generator<grad_xy_generator>
{
public:
    GeneratorParam<int> expected_width{ "expected_width", 1920 }; // Added default
    GeneratorParam<int> expected_height{ "expected_height", 1080 }; // Added default

    // Input image: Monochrome (grayscale) 8-bit image
    Input<Buffer<uint8_t>> input{"input", 2};

    // Output gradients: Float32 gradient images in X and Y directions
    Output<Buffer<float>> grad_x{"grad_x", 2};
    Output<Buffer<float>> grad_y{"grad_y", 2};

    Var x{"x"}, y{"y"}; // Declare loop variables

    void generate()
    {
        // Define the input boundary condition (repeat edge)
        Func input_bounded = BoundaryConditions::repeat_edge(input);

        // Central difference in X direction
        grad_x(x, y) = 0.5f * (cast<float>(input_bounded(x + 1, y)) -
                                cast<float>(input_bounded(x - 1, y)));

        // Central difference in Y direction
        grad_y(x, y) = 0.5f * (cast<float>(input_bounded(x, y + 1)) -
                                cast<float>(input_bounded(x, y - 1)));
    }

    void schedule()
    {
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi"); // Declare scheduling vars

        input.dim(0).set_estimate(40, expected_width);
        input.dim(1).set_estimate(22, expected_height);

        grad_x.dim(0).set_estimate(40, expected_width);
        grad_x.dim(1).set_estimate(22, expected_height);

        grad_y.dim(0).set_estimate(40, expected_width);
        grad_y.dim(1).set_estimate(22, expected_height);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#else
        bool auto_schedule = false;
#endif
        if (auto_schedule) {
            // input.set_estimates(...);
            // grad_x.set_estimates(...);
            // grad_y.set_estimates(...);
            return;
        }

        if (get_target().has_gpu_feature())
        {
            // Basic GPU schedule
            grad_x.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
            grad_y.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
        }
        else
        {
            // Assumes apply_schedule_grad_xy exists and is included
            apply_schedule_grad_xy(get_pipeline(), get_target());
        }
    }
};

HALIDE_REGISTER_GENERATOR(grad_xy_generator, grad_xy)

////////////////////////////////////////////////////////////////////////////////

// Include the schedule file if it exists, otherwise rely on default/auto schedule
// #include "schedules/grad_argmax.schedule.h" // This specific include might not exist

class grad_argmax_generator : public Halide::Generator<grad_argmax_generator>
{
public:
    GeneratorParam<int> expected_width{ "expected_width", 1920 }; // Added default
    GeneratorParam<int> expected_height{ "expected_height", 1080 }; // Added default

    // Input images: Gradients in X and Y directions
    Input<Buffer<float>> grad_x{"grad_x", 2};
    Input<Buffer<float>> grad_y{"grad_y", 2};

    // Tile dimensions in pixels
    GeneratorParam<int> tile_size{"tile_size", 8}; // Default tile size

    // Output: (tile_x, tile_y, coord_xy) -> pixel_coord
    Output<Buffer<uint16_t>> local_max_x{"local_max_x", 3};
    Output<Buffer<uint16_t>> local_max_y{"local_max_y", 3};

    Var x{"x"}, y{"y"}, c{"c"}; // Declare loop variables

    void generate()
    {
        RDom r(0, tile_size,
               0, tile_size,
               "r_tile");

        Expr global_x = x * tile_size + r.x;
        Expr global_y = y * tile_size + r.y;

        // Clamp coordinates to be within bounds of the gradient images
        Expr clamped_gx = clamp(global_x, 0, grad_x.width() - 1);
        // Use grad_x height assuming it's the same as grad_y height
        Expr clamped_gy = clamp(global_y, 0, grad_x.height() - 1);

        // Use clamped coordinates to access gradients safely
        Tuple grad_argmax_x = Halide::argmax(abs(grad_x(clamped_gx, clamped_gy)));
        // Use grad_y for the second argmax calculation
        Tuple grad_argmax_y = Halide::argmax(abs(grad_y(clamped_gx, clamped_gy)));

        // The result of argmax is relative to the RDom domain [0, tile_size-1].
        // Add the tile offset (x * tile_size, y * tile_size) to get global image coordinates.
        local_max_x(x, y, c) = cast<uint16_t>(mux(c, {
            grad_argmax_x[0] + x * tile_size, // Global X coordinate
            grad_argmax_x[1] + y * tile_size  // Global Y coordinate
        }));

        local_max_y(x, y, c) = cast<uint16_t>(mux(c, {
            grad_argmax_y[0] + x * tile_size, // Global X coordinate (using argmax from grad_y)
            grad_argmax_y[1] + y * tile_size  // Global Y coordinate (using argmax from grad_y)
        }));
    }

    void schedule()
    {
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi"); // Declare scheduling vars

        grad_x.dim(0).set_estimate(40, expected_width);
        grad_x.dim(1).set_estimate(22, expected_height);
        grad_y.dim(0).set_estimate(40, expected_width);
        grad_y.dim(1).set_estimate(22, expected_height);

        // Provide conservative constant estimates that satisfy Halide's
        // requirement for literal constant min/extents.
        const int kEstTilesX = 120; // ~1920 / 16
        const int kEstTilesY = 68;  // ~1080 / 16

        local_max_x.dim(0).set_estimate(0, kEstTilesX);
        local_max_x.dim(1).set_estimate(0, kEstTilesY);
        local_max_x.dim(2).set_estimate(0, 2);

        local_max_y.dim(0).set_estimate(0, kEstTilesX);
        local_max_y.dim(1).set_estimate(0, kEstTilesY);
        local_max_y.dim(2).set_estimate(0, 2);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#else
        bool auto_schedule = false;
#endif
        if (auto_schedule) {
            // grad_x.set_estimates(...); // Provide estimates
            // grad_y.set_estimates(...);
            // local_max_x.set_estimates(...);
            // local_max_y.set_estimates(...);
            return;
        }

        if (get_target().has_gpu_feature())
        {
            // Basic GPU schedule
            local_max_x.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 8, 8);
            local_max_y.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 8, 8);
        }
        else
        {
            // No specific CPU schedule provided, comment out the non-existent call
            // apply_schedule_grad_argmax(get_pipeline(), get_target());
            // Rely on default Halide schedule for CPU
             local_max_x.compute_root().parallel(y).vectorize(x, 8); // Example simple CPU schedule
             local_max_y.compute_root().parallel(y).vectorize(x, 8);
        }
    }
};

HALIDE_REGISTER_GENERATOR(grad_argmax_generator, grad_argmax)

////////////////////////////////////////////////////////////////////////////////

#include "schedules/sparse_jac.schedule.h" // Assumed to exist

class sparse_jac_generator : public Halide::Generator<sparse_jac_generator>
{
public:
    GeneratorParam<int> expected_width{ "expected_width", 1920 }; // Added default
    GeneratorParam<int> expected_height{ "expected_height", 1080 }; // Added default

    Input<Buffer<float>> grad_x{"grad_x", 2};
    Input<Buffer<float>> grad_y{"grad_y", 2};
    Input<Buffer<uint16_t>> local_max_x{"local_max_x", 3}; // (tile_x, tile_y, coord_xy) -> pixel_coord
    Input<Buffer<uint16_t>> local_max_y{"local_max_y", 3}; // (tile_x, tile_y, coord_xy) -> pixel_coord
    // Image dimensions (for center calculation and scale)
    Input<int> image_width{"image_width"};
    Input<int> image_height{"image_height"};

    Output<Buffer<float>> output_x{"output_x", 3}; // (tile_x, tile_y, param_index) -> jac_component
    Output<Buffer<float>> output_y{"output_y", 3}; // (tile_x, tile_y, param_index) -> jac_component

    Var x{"x"}, y{"y"}, c{"c"}; // Declare loop variables

    void generate()
    {
        // Calculate center and scale
        Expr Cx = cast<float>(image_width) / 2.0f;
        Expr Cy = cast<float>(image_height) / 2.0f;
        // Scale parameter derivatives for A, B by image width for consistency
        Expr scale = 1.f / cast<float>(image_width);

        // Get selected pixel coordinates (ensure they are clamped/valid before use)
        // Use local_max_x for output_x, local_max_y for output_y
        Expr ix0 = cast<int>(local_max_x(x, y, 0));
        Expr iy0 = cast<int>(local_max_x(x, y, 1));
        Expr ix1 = cast<int>(local_max_y(x, y, 0));
        Expr iy1 = cast<int>(local_max_y(x, y, 1));

        // Clamp coordinates to be within gradient image bounds
        Expr cl_ix0 = clamp(ix0, 0, grad_x.width() - 1);
        Expr cl_iy0 = clamp(iy0, 0, grad_x.height() - 1);
        Expr cl_ix1 = clamp(ix1, 0, grad_y.width() - 1); // Use grad_y dimensions
        Expr cl_iy1 = clamp(iy1, 0, grad_y.height() - 1);

        // Cast original (unclamped) pixel coordinates to float for calculations involving Cx, Cy
        Expr px0 = cast<float>(ix0);
        Expr py0 = cast<float>(iy0);
        Expr px1 = cast<float>(ix1);
        Expr py1 = cast<float>(iy1);

        // Get gradients at selected, clamped points
        Expr gx0 = grad_x(cl_ix0, cl_iy0);
        Expr gy1 = grad_y(cl_ix1, cl_iy1); // grad_y used for output_y

        // Calculate Jacobian components based on center-based warp derivatives
        // output_x corresponds to grad_x features
        output_x(x, y, c) = mux(c, {
            2.f * gx0 * (px0 - Cx) * scale,  // dWx/dA component * 2 * gx
            2.f * gx0 * -(py0 - Cy) * scale, // dWx/dB component * 2 * gx
            2.f * gx0,                       // dWx/dTX component * 2 * gx
            0.f,                             // dWx/dTY component * 2 * gx
        });

        // output_y corresponds to grad_y features
        output_y(x, y, c) = mux(c, {
            2.f * gy1 * (py1 - Cy) * scale,  // dWy/dA component * 2 * gy
            2.f * gy1 * (px1 - Cx) * scale,  // dWy/dB component * 2 * gy
            0.f,                             // dWy/dTX component * 2 * gy
            2.f * gy1,                       // dWy/dTY component * 2 * gy
        });
    }

    void schedule()
    {
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi"); // Declare scheduling vars

        // Estimate tile count based on a typical tile size (e.g., 8 or dynamic)
        const int default_tile_size_estimate = 8;
        Expr tile_estimate_x = expected_width / default_tile_size_estimate;
        Expr tile_estimate_y = expected_height / default_tile_size_estimate;

        grad_x.dim(0).set_estimate(128, expected_width);
        grad_x.dim(1).set_estimate(128, expected_height);
        grad_y.dim(0).set_estimate(128, expected_width);
        grad_y.dim(1).set_estimate(128, expected_height);
        local_max_x.set_estimates({{0, tile_estimate_x}, {0, tile_estimate_y}, {0, 2}});
        local_max_y.set_estimates({{0, tile_estimate_x}, {0, tile_estimate_y}, {0, 2}});
        image_width.set_estimate(expected_width);
        image_height.set_estimate(expected_height);
        output_x.set_estimates({{0, tile_estimate_x}, {0, tile_estimate_y}, {0, 4}});
        output_y.set_estimates({{0, tile_estimate_x}, {0, tile_estimate_y}, {0, 4}});

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#else
        bool auto_schedule = false;
#endif
        if (auto_schedule) {
            // Provide estimates
            // grad_x.set_estimates(...); grad_y.set_estimates(...);
            // local_max_x.set_estimates(...); local_max_y.set_estimates(...);
            // output_x.set_estimates(...); output_y.set_estimates(...);
            return;
        }

        if (get_target().has_gpu_feature())
        {
            // Basic GPU schedule
            output_x.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 8, 8);
            output_y.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 8, 8);
        }
        else
        {
            // Assumes apply_schedule_sparse_jac exists and is included
            apply_schedule_sparse_jac(get_pipeline(), get_target());
        }
    }
};

HALIDE_REGISTER_GENERATOR(sparse_jac_generator, sparse_jac)

////////////////////////////////////////////////////////////////////////////////

#include "schedules/sparse_ica.schedule.h" // Assumed to exist

class sparse_ica_generator : public Halide::Generator<sparse_ica_generator>
{
public:
    GeneratorParam<int> expected_width{ "expected_width", 1920 }; // Added default
    GeneratorParam<int> expected_height{ "expected_height", 1080 }; // Added default
    GeneratorParam<int> expected_points{ "expected_points", 2000 }; // Added default


    Input<Buffer<uint8_t>> input_template{"input_template", 2};
    Input<Buffer<uint8_t>> input_keyframe{"input_keyframe", 2}; // to be warped
    Input<Buffer<uint16_t>> selected_pixels_x{"selected_pixels_x", 2}; // (index, coord_xy) -> pixel_coord
    Input<Buffer<uint16_t>> selected_pixels_y{"selected_pixels_y", 2}; // (index, coord_xy) -> pixel_coord
    Input<Buffer<float>> selected_jacobians_x{"selected_jacobians_x", 2}; // (index, param_idx) -> jac_comp
    Input<Buffer<float>> selected_jacobians_y{"selected_jacobians_y", 2}; // (index, param_idx) -> jac_comp
    // 4-parameter similarity warp
    Input<float> A{"A"};
    Input<float> B{"B"};
    Input<float> TX{"TX"};
    Input<float> TY{"TY"};
    // Image dimensions (for center calculation)
    Input<int> image_width{"image_width"};
    Input<int> image_height{"image_height"};


    // Output: length-4 vector = sum of Jᵀ*(template - warped)
    Output<Buffer<double>> output{"output", 1};

    // Declare Funcs and Vars used in both generate and schedule
    Var i{"i"}, u{"u"}, c{"c"}; // Make index vars class members or redefine in schedule

    // Funcs for warping and reduction
    Func warp_pixel_x{"warp_pixel_x"}, warp_pixel_y{"warp_pixel_y"};
    Func reduce_4_x{"reduce_4_x"}, reduce_4_y{"reduce_4_y"};
    // Declare Lanczos weight funcs here to be visible in schedule
    Func weight_x{"weight_x"}, weight_y{"weight_y"};
    Func weight_x_y{"weight_x_y"}, weight_y_y{"weight_y_y"}; // Weights for Y pixels
    // Intermediate warp coordinate calculations
    Func floorWx_x{"floorWx_x"}, floorWy_x{"floorWy_x"};
    Func fracWx_x{"fracWx_x"}, fracWy_x{"fracWy_x"};
    Func floorWx_y{"floorWx_y"}, floorWy_y{"floorWy_y"};
    Func fracWx_y{"fracWx_y"}, fracWy_y{"fracWy_y"};


    void generate()
    {
        // Image center
        Expr Cx = cast<float>(image_width) / 2.0f;
        Expr Cy = cast<float>(image_height) / 2.0f;

        // Boundary-condition for reading the keyframe
        Func clamped = BoundaryConditions::repeat_edge(input_keyframe);

        // Calculate warp coordinates and fractions for selected_pixels_x
        {
            Expr orig_x = cast<float>( selected_pixels_x(i, 0) );
            Expr orig_y = cast<float>( selected_pixels_x(i, 1) );
            Expr Wx = (1.0f + A)*(orig_x - Cx) - B*(orig_y - Cy) + Cx + TX;
            Expr Wy = B*(orig_x - Cx) + (1.0f + A)*(orig_y - Cy) + Cy + TY;
            floorWx_x(i) = floor(Wx);
            floorWy_x(i) = floor(Wy);
            fracWx_x(i)  = Wx - floorWx_x(i);
            fracWy_x(i)  = Wy - floorWy_x(i);
        }
        // Calculate warp coordinates and fractions for selected_pixels_y
        {
            // Need a different Var for iterating selected_pixels_y if scheduling independently
            // Using 'i' here implies same dimension extent, which is typical but not guaranteed
            Expr orig_x = cast<float>( selected_pixels_y(i, 0) );
            Expr orig_y = cast<float>( selected_pixels_y(i, 1) );
            Expr Wx = (1.0f + A)*(orig_x - Cx) - B*(orig_y - Cy) + Cx + TX;
            Expr Wy = B*(orig_x - Cx) + (1.0f + A)*(orig_y - Cy) + Cy + TY;
            floorWx_y(i) = floor(Wx);
            floorWy_y(i) = floor(Wy);
            fracWx_y(i)  = Wx - floorWx_y(i);
            fracWy_y(i)  = Wy - floorWy_y(i);
        }

        // Define Lanczos weight calculations using the precomputed fractions
        // 'u' is the dimension for the 5 taps of Lanczos kernel
        weight_x(i, u) = lanczos2(cast<float>(u - 2) - fracWx_x(i)); // Use _x frac
        weight_y(i, u) = lanczos2(cast<float>(u - 2) - fracWy_x(i)); // Use _x frac

        // Define the warp pixel calculation using Lanczos interpolation for selected_pixels_x
        {
            RDom rxy(0, 5, 0, 5, "rxy");
            Expr w_2d = weight_x(i, rxy.x) * weight_y(i, rxy.y); // Use weight_x/y (derived from frac_x)

            Expr sample_x = cast<int>(floorWx_x(i)) + (rxy.x - 2);
            Expr sample_y = cast<int>(floorWy_x(i)) + (rxy.y - 2);
            Expr val      = cast<float>( clamped(sample_x, sample_y) );

            Expr sum_num  = sum(w_2d * val);
            Expr sum_den  = sum(w_2d);

            // Avoid division by zero if weights sum to zero
            warp_pixel_x(i) = select(abs(sum_den) > 1e-6f, sum_num / sum_den, 0.0f);
        }

         // Define weights for selected_pixels_y
        weight_x_y(i, u) = lanczos2(cast<float>(u - 2) - fracWx_y(i)); // Use _y frac
        weight_y_y(i, u) = lanczos2(cast<float>(u - 2) - fracWy_y(i)); // Use _y frac

        // Define the warp pixel calculation using Lanczos interpolation for selected_pixels_y
        {
            RDom rxy(0, 5, 0, 5, "rxy_y"); // Use different name for RDom if needed
            Expr w_2d = weight_x_y(i, rxy.x) * weight_y_y(i, rxy.y); // Use weight_x_y/y_y

            Expr sample_x = cast<int>(floorWx_y(i)) + (rxy.x - 2);
            Expr sample_y = cast<int>(floorWy_y(i)) + (rxy.y - 2);
            Expr val      = cast<float>( clamped(sample_x, sample_y) );

            Expr sum_num  = sum(w_2d * val);
            Expr sum_den  = sum(w_2d);

             // Avoid division by zero
            warp_pixel_y(i) = select(abs(sum_den) > 1e-6f, sum_num / sum_den, 0.0f);
        }

        //------------------------------------------------------------------
        // 2) Accumulate the final 4-vector: sum of Jᵀ * (template - warped)
        //------------------------------------------------------------------
        {
            // Reduce over selected_pixels_x
            RDom r_x(0, selected_pixels_x.dim(0).extent(), "r_x"); // Use extent for reduction domain
            reduce_4_x(c) = cast<double>(0); // Initialize reduction

            Expr warped_val_x = warp_pixel_x(r_x); // Use RDom variable r_x

            // Clamp template coordinates just in case
            Expr tmpl_x_x = clamp(cast<int>(selected_pixels_x(r_x, 0)), 0, input_template.width()  - 1);
            Expr tmpl_y_x = clamp(cast<int>(selected_pixels_x(r_x, 1)), 0, input_template.height() - 1);
            Expr template_val_x = cast<float>( input_template(tmpl_x_x, tmpl_y_x) );

            Expr residual_x = template_val_x - warped_val_x;

            // Accumulate Jᵀ * residual using selected_jacobians_x
            // Ensure reduction update uses +=
            reduce_4_x(c) += cast<double>(selected_jacobians_x(r_x, c) * residual_x); // Use c directly
        }
        {
            // Reduce over selected_pixels_y
            RDom r_y(0, selected_pixels_y.dim(0).extent(), "r_y"); // Use extent for reduction domain
            reduce_4_y(c) = cast<double>(0); // Initialize reduction

            Expr warped_val_y = warp_pixel_y(r_y); // Use RDom variable r_y

            // Clamp template coordinates
            Expr tmpl_x_y = clamp(cast<int>(selected_pixels_y(r_y, 0)), 0, input_template.width()  - 1);
            Expr tmpl_y_y = clamp(cast<int>(selected_pixels_y(r_y, 1)), 0, input_template.height() - 1);
            Expr template_val_y = cast<float>( input_template(tmpl_x_y, tmpl_y_y) );

            Expr residual_y = template_val_y - warped_val_y;

            // Accumulate Jᵀ * residual using selected_jacobians_y
            reduce_4_y(c) += cast<double>(selected_jacobians_y(r_y, c) * residual_y); // Use c directly
        }

        // Combine results from both sets of selected pixels
        output(c) = reduce_4_x(c) + reduce_4_y(c);
    }

    void schedule()
    {
        Var io("io"), ii("ii"); // GPU tiling variables for 'i' dimension
        Var c_outer("co"), c_inner("ci"); // GPU tiling variables for 'c' dimension
        const int thread_count = 128; // Example thread count per block

        input_template.dim(0).set_estimate(128, expected_width);
        input_template.dim(1).set_estimate(128, expected_height);
        input_keyframe.dim(0).set_estimate(128, expected_width);
        input_keyframe.dim(1).set_estimate(128, expected_height);
        selected_pixels_x.dim(0).set_estimate(800, expected_points);
        selected_pixels_x.dim(1).set_estimate(2, 2);
        selected_pixels_y.dim(0).set_estimate(800, expected_points);
        selected_pixels_y.dim(1).set_estimate(2, 2);
        selected_jacobians_x.dim(0).set_estimate(800, expected_points);
        selected_jacobians_x.dim(1).set_estimate(4, 4);
        selected_jacobians_y.dim(0).set_estimate(800, expected_points);
        selected_jacobians_y.dim(1).set_estimate(4, 4);
        image_width.set_estimate(expected_width);
        image_height.set_estimate(expected_height);

        A.set_estimate(0.1f);
        B.set_estimate(0.1f);
        TX.set_estimate(1.f);
        TY.set_estimate(1.f);

        // output is a length-4 vector, dim name is 'c'
        output.set_estimate(c, 0, 4);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#else
        bool auto_schedule = false;
#endif
        if (auto_schedule) {
            // Provide estimates
            // input_template.set_estimates(...); input_keyframe.set_estimates(...);
            // selected_pixels_x.set_estimates(...); selected_pixels_y.set_estimates(...);
            // selected_jacobians_x.set_estimates(...); selected_jacobians_y.set_estimates(...);
            // output.set_estimates(...);
            return;
        }

        if (get_target().has_gpu_feature())
        {
            // Schedule the final output reduction (over c)
             output.compute_root().gpu_tile(c, c_outer, c_inner, 4);

            // Schedule the update steps of the reduction (parallel over c)
            // These run after warp_pixel_* are computed
            reduce_4_x.update().gpu_threads(c);
            reduce_4_y.update().gpu_threads(c);

            // Schedule the intermediate warp calculations (over i)
            // Split 'i' into blocks (io) and threads (ii)
            warp_pixel_x.compute_root()
                .split(i, io, ii, thread_count) // Split dimension 'i'
                .gpu_blocks(io)                  // Map outer part to blocks
                .gpu_threads(ii);                // Map inner part to threads

            warp_pixel_y.compute_root()
                .split(i, io, ii, thread_count)
                .gpu_blocks(io)
                .gpu_threads(ii);

            // Schedule weights - compute inline or compute_at warp stages
             weight_x.compute_at(warp_pixel_x, ii); // Compute per thread
             weight_y.compute_at(warp_pixel_x, ii);
             weight_x_y.compute_at(warp_pixel_y, ii);
             weight_y_y.compute_at(warp_pixel_y, ii);

             // Schedule intermediate floor/frac calculations
             floorWx_x.compute_at(warp_pixel_x, io); // Compute per block
             floorWy_x.compute_at(warp_pixel_x, io);
             fracWx_x.compute_at(warp_pixel_x, ii); // Compute per thread
             fracWy_x.compute_at(warp_pixel_x, ii);
             // Repeat for _y versions
             floorWx_y.compute_at(warp_pixel_y, io);
             floorWy_y.compute_at(warp_pixel_y, io);
             fracWx_y.compute_at(warp_pixel_y, ii);
             fracWy_y.compute_at(warp_pixel_y, ii);


        }
        else
        {
            // Similar to sparse_warpdiff, fall back to a very simple CPU
            // schedule to avoid brittleness in the pre‑generated schedule.

            Var io("io"), ii("ii");
            const int vec = 16;

            output.compute_root();

            // Compute helper functions inline at root for simplicity
            warp_pixel_x.compute_root();
            warp_pixel_y.compute_root();
        }
    }
};

HALIDE_REGISTER_GENERATOR(sparse_ica_generator, sparse_ica)

////////////////////////////////////////////////////////////////////////////////

#include "schedules/sparse_warpdiff.schedule.h" // Assumed to exist

class sparse_warpdiff_generator : public Halide::Generator<sparse_warpdiff_generator>
{
public:
    GeneratorParam<int> expected_width{ "expected_width", 1920 }; // Added default
    GeneratorParam<int> expected_height{ "expected_height", 1080 }; // Added default

    Input<Buffer<uint8_t>> input_template{"input_template", 2};
    Input<Buffer<uint8_t>> input_keyframe{"input_keyframe", 2}; // to be warped
    Input<Buffer<uint16_t>> local_max{"local_max", 3}; // (tile_x, tile_y, coord_xy) -> pixel_coord
    // 4-parameter similarity warp
    Input<float> A{"A"};
    Input<float> B{"B"};
    Input<float> TX{"TX"};
    Input<float> TY{"TY"};
    // Image dimensions (for center calculation)
    Input<int> image_width{"image_width"};
    Input<int> image_height{"image_height"};

    Output<Buffer<uint16_t>> output{"output", 2}; // (tile_x, tile_y) -> diff

    // Declare Funcs and Vars used in scheduling
    Var x{"x"}, y{"y"}, u{"u"};
    Func weight_x{"weight_x"}, weight_y{"weight_y"}; // Declare weights here

    void generate()
    {
        // Image center
        Expr Cx = cast<float>(image_width) / 2.0f;
        Expr Cy = cast<float>(image_height) / 2.0f;

        // Boundary-condition for reading the keyframe
        Func clamped = BoundaryConditions::repeat_edge(input_keyframe);

        // Get the selected pixel's location from local_max
        Expr pix_x = cast<int>(local_max(x, y, 0));
        Expr pix_y = cast<int>(local_max(x, y, 1));

        // Clamp template pixel coordinates for safety when reading template
        Expr template_x = clamp(pix_x, 0, input_template.width() - 1);
        Expr template_y = clamp(pix_y, 0, input_template.height() - 1);

        // Pixel coordinates as float for warp calculation
        Expr orig_x = cast<float>(pix_x);
        Expr orig_y = cast<float>(pix_y);

        // Compute warped location (Wx, Wy) in the keyframe using center-based formula
        Expr Wx = (1.0f + A)*(orig_x - Cx) - B*(orig_y - Cy) + Cx + TX;
        Expr Wy = B*(orig_x - Cx) + (1.0f + A)*(orig_y - Cy) + Cy + TY;

        // Floor and fractional parts for interpolation
        Expr floorWx = floor(Wx);
        Expr floorWy = floor(Wy);
        Expr fracWx  = Wx - floorWx;
        Expr fracWy  = Wy - floorWy;

        // --- Lanczos Interpolation ---
        // Define weight calculation (using class member Funcs)
        // 'u' is the kernel tap index [0..4]
        weight_x(x, y, u) = lanczos2(cast<float>(u - 2) - fracWx);
        weight_y(x, y, u) = lanczos2(cast<float>(u - 2) - fracWy);

        // Small sum over 5x5 neighborhood
        RDom rxy(0, 5, 0, 5, "rxy");
        Expr w_2d = weight_x(x, y, rxy.x) * weight_y(x, y, rxy.y);

        Expr sample_x = cast<int>(floorWx) + (rxy.x - 2);
        Expr sample_y = cast<int>(floorWy) + (rxy.y - 2);
        // Use clamped boundary condition for reading keyframe samples
        Expr val      = cast<float>( clamped(sample_x, sample_y) );

        Expr sum_num  = sum(w_2d * val);
        Expr sum_den  = sum(w_2d);

        // Calculate interpolated value (handle potential division by zero)
        Expr interpolated = select(abs(sum_den) > 1e-6f, sum_num / sum_den, 0.0f);

        // Get template value at the original pixel location (use clamped coords)
        Expr template_value = cast<float>(input_template(template_x, template_y));

        // Calculate absolute difference
        Expr diff = abs(interpolated - template_value);

        // Clamp and cast output to uint16_t
        output(x, y) = cast<uint16_t>( clamp(diff, 0.0f, 65535.0f) );
    }


    void schedule()
    {
        Var xo("xo"), yo("yo"), xi("xi"), yi("yi"); // Declare scheduling vars

        // Estimate tile count based on a typical tile size (e.g., 8 or dynamic)
        const int default_tile_size_estimate = 8;
        Expr tile_estimate_x = expected_width / default_tile_size_estimate;
        Expr tile_estimate_y = expected_height / default_tile_size_estimate;


        input_template.dim(0).set_estimate(128, expected_width);
        input_template.dim(1).set_estimate(128, expected_height);
        input_keyframe.dim(0).set_estimate(128, expected_width);
        input_keyframe.dim(1).set_estimate(128, expected_height);
        local_max.set_estimates({{0, tile_estimate_x}, {0, tile_estimate_y}, {0, 2}});
        image_width.set_estimate(expected_width);
        image_height.set_estimate(expected_height);

        A.set_estimate(0.1f);
        B.set_estimate(0.1f);
        TX.set_estimate(1.f);
        TY.set_estimate(1.f);

        // Output dimensions match local_max tile dimensions
        output.set_estimates({{0, tile_estimate_x}, {0, tile_estimate_y}});

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#else
        bool auto_schedule = false;
#endif
        if (auto_schedule) {
            // Provide estimates
            // input_template.set_estimates(...); input_keyframe.set_estimates(...);
            // local_max.set_estimates(...); output.set_estimates(...);
            return;
        }

        if (get_target().has_gpu_feature())
        {
            // Basic GPU schedule
            output.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 8, 8);
            // Schedule intermediate funcs if necessary (e.g., compute weights inline or at tile level)
             weight_x.compute_at(output, xo); // Compute weights per output tile block
             weight_y.compute_at(output, xo);
        }
        else
        {
            // The pre-generated schedule from Adams2019 occasionally breaks
            // when minor changes to the Halide IR alter variable names
            // (e.g., the BoundaryConditions helper may rename loop vars).
            // Fall back to a very simple, portable CPU schedule that keeps
            // performance reasonable while ensuring the generator builds
            // successfully regardless of such renaming.

            // Compute everything root‑level and vectorize the inner dimension
            Var xo_s("xo_s"), xi_s("xi_s");
            const int vec = 16; // reasonable vector width for AVX2

            output
                .compute_root()
                .split(x, xo_s, xi_s, vec, Halide::TailStrategy::GuardWithIf)
                .vectorize(xi_s);

            // Inline auxiliary Funcs to avoid separate scheduling needs
            weight_x.compute_at(output, xo_s);
            weight_y.compute_at(output, xo_s);
        }
    }
};

HALIDE_REGISTER_GENERATOR(sparse_warpdiff_generator, sparse_warpdiff)