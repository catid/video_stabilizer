#include "generators_tools.h"

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

#include "schedules/pyr_down.schedule.h"

class pyr_down_generator : public Halide::Generator<pyr_down_generator>
{
public:
    // Input image: Monochrome (grayscale) 8-bit image
    Input<Buffer<uint8_t>> input{"input", 2};
    // Output image: Downsampled 8-bit monochrome image
    Output<Buffer<uint8_t>> output{"output", 2};

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
        input.dim(0).set_estimate(80, expected_width);
        input.dim(1).set_estimate(45, expected_height);
        output.dim(0).set_estimate(40, expected_width/2);
        output.dim(1).set_estimate(22, expected_height/2);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#endif
        if (auto_schedule) {
            return;
        }

        if (get_target().has_gpu_feature())
        {
            // FIXME
            output.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 8, 8);
        }
        else
        {
            apply_schedule_pyr_down(get_pipeline(), get_target());
        }
    }
};

HALIDE_REGISTER_GENERATOR(pyr_down_generator, pyr_down)

////////////////////////////////////////////////////////////////////////////////

#include "schedules/image_warp.schedule.h"

class image_warp_generator : public Halide::Generator<image_warp_generator>
{
public:
    // Input image: Monochrome or multi-channel image
    Input<Buffer<uint8_t>> input{"input", 2};
    // Transform parameters
    Input<float> A{"A"};
    Input<float> B{"B"};
    Input<float> TX{"TX"};
    Input<float> TY{"TY"};
    // Output image
    Output<Buffer<float>> output{"output", 2};

    void generate() {
        // Warp equations remain the same
        Expr W_x = (1.0f + A) * x - B * y + TX;
        Expr W_y = B * x + (1.0f + A) * y + TY;

        // Boundary condition
        Func clamped("clamped");
        clamped(x, y) = BoundaryConditions::repeat_edge(input)(x, y);

        // Compute the integer and fractional parts of the warp coordinates
        Expr W_x_floor = cast<int>(floor(W_x));
        Expr W_y_floor = cast<int>(floor(W_y));
        Expr wx = W_x - cast<float>(W_x_floor);
        Expr wy = W_y - cast<float>(W_y_floor);

        // Sample the four nearest neighbors
        Expr x0y0 = cast<float>( clamped(W_x_floor, W_y_floor) );
        Expr x1y0 = cast<float>( clamped(W_x_floor + 1, W_y_floor) );
        Expr x0y1 = cast<float>( clamped(W_x_floor, W_y_floor + 1) );
        Expr x1y1 = cast<float>( clamped(W_x_floor + 1, W_y_floor + 1) );

        // Bilinear interpolation
        Expr top = lerp(x0y0, x1y0, wx);
        Expr bottom = lerp(x0y1, x1y1, wx);
        output(x, y) = lerp(top, bottom, wy);
    }

    void schedule()
    {
        input.dim(0).set_estimate(80, expected_width);
        input.dim(1).set_estimate(45, expected_height);
        A.set_estimate(0.1f);
        B.set_estimate(0.1f);
        TX.set_estimate(1.f);
        TY.set_estimate(1.f);
        output.dim(0).set_estimate(40, expected_width/2);
        output.dim(1).set_estimate(22, expected_height/2);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#endif
        if (auto_schedule) {
            return;
        }

        if (get_target().has_gpu_feature())
        {
            // FIXME
            output.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 8, 8);
        }
        else
        {
            apply_schedule_image_warp(get_pipeline(), get_target());
        }
    }
};

HALIDE_REGISTER_GENERATOR(image_warp_generator, image_warp)

////////////////////////////////////////////////////////////////////////////////

#include "schedules/grad_xy.schedule.h"

class grad_xy_generator : public Halide::Generator<grad_xy_generator>
{
public:
    // Input image: Monochrome (grayscale) 8-bit image
    Input<Buffer<uint8_t>> input{"input", 2};

    // Output gradients: Float32 gradient images in X and Y directions
    Output<Buffer<float>> grad_x{"grad_x", 2};
    Output<Buffer<float>> grad_y{"grad_y", 2};

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
        input.dim(0).set_estimate(40, expected_width);
        input.dim(1).set_estimate(22, expected_height);

        grad_x.dim(0).set_estimate(40, expected_width);
        grad_x.dim(1).set_estimate(22, expected_height);

        grad_y.dim(0).set_estimate(40, expected_width);
        grad_y.dim(1).set_estimate(22, expected_height);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#endif
        if (auto_schedule) {
            return;
        }

        if (get_target().has_gpu_feature())
        {
        }
        else
        {
            apply_schedule_grad_xy(get_pipeline(), get_target());
        }
    }
};

HALIDE_REGISTER_GENERATOR(grad_xy_generator, grad_xy)

////////////////////////////////////////////////////////////////////////////////

#include "schedules/grad_argmax.schedule.h"

class grad_argmax_generator : public Halide::Generator<grad_argmax_generator>
{
public:
    // Input images: Gradients in X and Y directions
    Input<Buffer<float>> grad_x{"grad_x", 2};
    Input<Buffer<float>> grad_y{"grad_y", 2};

    // Tile dimensions in pixels
    GeneratorParam<int> tile_size{"tile_size", 8};

    Output<Buffer<uint16_t>> local_max_x{"local_max_x", 3};
    Output<Buffer<uint16_t>> local_max_y{"local_max_y", 3};

    void generate() 
    {
        RDom r(0, tile_size,
               0, tile_size,
               "r_tile");

        Expr global_x = x * tile_size + r.x;
        Expr global_y = y * tile_size + r.y;

        Tuple grad_argmax_x = Halide::argmax(abs(grad_x(global_x, global_y)));
        Tuple grad_argmax_y = Halide::argmax(abs(grad_y(global_x, global_y)));

        local_max_x(x, y, c) = cast<uint16_t>(mux(c, {
            grad_argmax_x[0] + x * tile_size,
            grad_argmax_x[1] + y * tile_size
        }));

        local_max_y(x, y, c) = cast<uint16_t>(mux(c, {
            grad_argmax_y[0] + x * tile_size,
            grad_argmax_y[1] + y * tile_size
        }));
    }

    void schedule() 
    {
        grad_x.dim(0).set_estimate(40, expected_width);
        grad_x.dim(1).set_estimate(22, expected_height);
        grad_y.dim(0).set_estimate(40, expected_width);
        grad_y.dim(1).set_estimate(22, expected_height);
        local_max_x.dim(0).set_estimate(40, 45);
        local_max_x.dim(1).set_estimate(22, 25);
        local_max_x.dim(2).set_estimate(0, expected_width);
        local_max_y.dim(0).set_estimate(40, 45);
        local_max_y.dim(1).set_estimate(22, 25);
        local_max_y.dim(2).set_estimate(0, expected_width);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#endif
        if (auto_schedule) {
            return;
        }

        if (get_target().has_gpu_feature())
        {
        }
        else
        {
            //apply_schedule_grad_argmax(get_pipeline(), get_target());
        }
    }
};

HALIDE_REGISTER_GENERATOR(grad_argmax_generator, grad_argmax)

////////////////////////////////////////////////////////////////////////////////

#include "schedules/sparse_jac.schedule.h"

class sparse_jac_generator : public Halide::Generator<sparse_jac_generator>
{
public:
    Input<Buffer<float>> grad_x{"grad_x", 2};
    Input<Buffer<float>> grad_y{"grad_y", 2};
    Input<Buffer<uint16_t>> local_max_x{"local_max_x", 3};
    Input<Buffer<uint16_t>> local_max_y{"local_max_y", 3};

    Output<Buffer<float>> output_x{"output_x", 3};
    Output<Buffer<float>> output_y{"output_y", 3};

    void generate()
    {
        // Boundary condition
        Expr ix0 = min(local_max_x(x, y, 0), grad_x.width() - 1);
        Expr iy0 = min(local_max_x(x, y, 1), grad_x.height() - 1);
        Expr ix1 = min(local_max_y(x, y, 0), grad_x.width() - 1);
        Expr iy1 = min(local_max_y(x, y, 1), grad_x.height() - 1);

        Expr scale = 1.f / grad_x.width();

        output_x(x, y, c) = mux(c, {
            (grad_x(ix0, iy0) * ix0 + grad_y(ix0, iy0) * iy0) * scale,
            (grad_x(ix0, iy0) * -iy0 + grad_y(ix0, iy0) * ix0) * scale,
            (grad_x(ix0, iy0)),
            (grad_y(ix0, iy0)),
        });

        output_y(x, y, c) = mux(c, {
            (grad_x(ix1, iy1) * ix1 + grad_y(ix1, iy1) * iy1) * scale,
            (grad_x(ix1, iy1) * -iy1 + grad_y(ix1, iy1) * ix1) * scale,
            (grad_x(ix1, iy1)),
            (grad_y(ix1, iy1)),
        });
    }

    void schedule()
    {
        grad_x.dim(0).set_estimate(128, expected_width);
        grad_x.dim(1).set_estimate(128, expected_height);
        grad_y.dim(0).set_estimate(128, expected_width);
        grad_y.dim(1).set_estimate(128, expected_height);
        local_max_x.dim(0).set_estimate(16, expected_width / 14);
        local_max_x.dim(1).set_estimate(16, expected_height / 14);
        local_max_x.dim(2).set_estimate(2, 2);
        local_max_y.dim(0).set_estimate(16, expected_width / 14);
        local_max_y.dim(1).set_estimate(16, expected_height / 14);
        local_max_y.dim(2).set_estimate(2, 2);
        output_x.set_estimate(x, 16, expected_width / 14).set_estimate(y, 16, expected_height / 14).set_estimate(c, 4, 4);
        output_y.set_estimate(x, 16, expected_width / 14).set_estimate(y, 16, expected_height / 14).set_estimate(c, 4, 4);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#endif
        if (auto_schedule) {
            return;
        }

        if (get_target().has_gpu_feature())
        {
            // FIXME
            output_x.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 8, 8);
            output_y.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 8, 8);
        }
        else
        {
            apply_schedule_sparse_jac(get_pipeline(), get_target());
        }
    }
};

HALIDE_REGISTER_GENERATOR(sparse_jac_generator, sparse_jac)

////////////////////////////////////////////////////////////////////////////////

#include "schedules/sparse_ica.schedule.h"

class sparse_ica_generator : public Halide::Generator<sparse_ica_generator>
{
public:
    Input<Buffer<uint8_t>> input_template{"input_template", 2};
    Input<Buffer<uint8_t>> input_keyframe{"input_keyframe", 2}; // to be warped
    Input<Buffer<uint16_t>> selected_pixels_x{"selected_pixels_x", 2};
    Input<Buffer<uint16_t>> selected_pixels_y{"selected_pixels_y", 2};
    Input<Buffer<float>> selected_jacobians_x{"selected_jacobians_x", 2};
    Input<Buffer<float>> selected_jacobians_y{"selected_jacobians_y", 2};
    // 4-parameter similarity warp
    Input<float> A{"A"};
    Input<float> B{"B"};
    Input<float> TX{"TX"};
    Input<float> TY{"TY"};

    // Output: length-4 vector = sum of Jᵀ*(template - warped)
    Output<Buffer<double>> output{"output", 1};

    // We'll define just one Func ("warp_pixel") for the 2D warp,
    // and then a reduction Func ("reduce_4") to accumulate the 4-vector.
    Func warp_pixel_x, warp_pixel_y, reduce_4_x, reduce_4_y;

    void generate()
    {
        // Boundary-condition for reading the keyframe
        Func clamped = BoundaryConditions::repeat_edge(input_keyframe);

        //------------------------------------------------------------------
        // 1) Warp each selected pixel with a full 2D Lanczos2 sample
        //------------------------------------------------------------------
        {
            // We'll define warp_pixel(i) = the warped intensity for pixel "i".
            // 'i' indexes into selected_pixels, which has (x,y) in image coords.
            Var i("i");

            // The selected pixel's original location:
            Expr orig_x = cast<float>( selected_pixels_x(i, 0) );
            Expr orig_y = cast<float>( selected_pixels_x(i, 1) );

            // Compute warped location (Wx, Wy) in the keyframe
            Expr Wx = (1.0f + A)*orig_x - B*orig_y + TX;
            Expr Wy = B*orig_x + (1.0f + A)*orig_y + TY;

            // Floor and fractional parts
            Expr floorWx = floor(Wx);
            Expr floorWy = floor(Wy);
            Expr fracWx  = Wx - floorWx;
            Expr fracWy  = Wy - floorWy;

            // A 5×5 sampling kernel using Lanczos2 in 2D
            RDom rxy(0, 5, 0, 5, "rxy"); // kernel radius=2 => 5 taps
            Expr rx = rxy.x - 2;        // in [-2..2]
            Expr ry = rxy.y - 2;        // in [-2..2]

            // Compute 2D weights
            Expr distx = rx - fracWx;
            Expr disty = ry - fracWy;
            Expr w_x   = lanczos2(distx);
            Expr w_y   = lanczos2(disty);
            Expr w2D   = w_x * w_y;

            // Sample from keyframe at (floorWx+rx, floorWy+ry)
            // with boundary checks
            Expr sample_x = cast<int>(floorWx) + rx;
            Expr sample_y = cast<int>(floorWy) + ry;
            Expr val = cast<float>( clamped(sample_x, sample_y) );

            // Sum up w2D*val and w2D separately
            Expr sum_num = sum(w2D * val);
            Expr sum_den = sum(w2D);

            warp_pixel_x(i) = sum_num / sum_den;
        }
        {
            // We'll define warp_pixel(i) = the warped intensity for pixel "i".
            // 'i' indexes into selected_pixels, which has (x,y) in image coords.
            Var i("i");

            // The selected pixel's original location:
            Expr orig_x = cast<float>( selected_pixels_y(i, 0) );
            Expr orig_y = cast<float>( selected_pixels_y(i, 1) );

            // Compute warped location (Wx, Wy) in the keyframe
            Expr Wx = (1.0f + A)*orig_x - B*orig_y + TX;
            Expr Wy = B*orig_x + (1.0f + A)*orig_y + TY;

            // Floor and fractional parts
            Expr floorWx = floor(Wx);
            Expr floorWy = floor(Wy);
            Expr fracWx  = Wx - floorWx;
            Expr fracWy  = Wy - floorWy;

            // A 5×5 sampling kernel using Lanczos2 in 2D
            RDom rxy(0, 5, 0, 5, "rxy"); // kernel radius=2 => 5 taps
            Expr rx = rxy.x - 2;        // in [-2..2]
            Expr ry = rxy.y - 2;        // in [-2..2]

            // Compute 2D weights
            Expr distx = rx - fracWx;
            Expr disty = ry - fracWy;
            Expr w_x   = lanczos2(distx);
            Expr w_y   = lanczos2(disty);
            Expr w2D   = w_x * w_y;

            // Sample from keyframe at (floorWx+rx, floorWy+ry)
            // with boundary checks
            Expr sample_x = cast<int>(floorWx) + rx;
            Expr sample_y = cast<int>(floorWy) + ry;
            Expr val = cast<float>( clamped(sample_x, sample_y) );

            // Sum up w2D*val and w2D separately
            Expr sum_num = sum(w2D * val);
            Expr sum_den = sum(w2D);

            warp_pixel_y(i) = sum_num / sum_den;
        }

        //------------------------------------------------------------------
        // 2) Accumulate the final 4-vector: sum of Jᵀ * (template - warped)
        //------------------------------------------------------------------
        {
            // We'll define reduce_4(c), for c in [0..3].
            // Start from 0, then reduce over all i in [0..#selected_pixels).
            Var c("c");
            reduce_4_x(c) = cast<double>(0);

            RDom r(0, selected_pixels_x.dim(0).extent(), "r");

            // The warped value for pixel i
            Expr warped_val = warp_pixel_x(r);

            // The corresponding template value
            Expr tmpl_x = min(selected_pixels_x(r, 0), input_template.width()  - 1);
            Expr tmpl_y = min(selected_pixels_x(r, 1), input_template.height() - 1);
            Expr template_val = cast<float>( input_template(tmpl_x, tmpl_y) );

            // Residual
            Expr residual = template_val - warped_val;

            // Accumulate Jᵀ * residual
            // selected_jacobians(r,c) is the c-th channel of the Jacobian at pixel i.
            reduce_4_x(0) += selected_jacobians_x(r, 0) * residual;
            reduce_4_x(1) += selected_jacobians_x(r, 1) * residual;
            reduce_4_x(2) += selected_jacobians_x(r, 2) * residual;
            reduce_4_x(3) += selected_jacobians_x(r, 3) * residual;
        }
        {
            // We'll define reduce_4(c), for c in [0..3].
            // Start from 0, then reduce over all i in [0..#selected_pixels).
            Var c("c");
            reduce_4_y(c) = cast<double>(0);

            RDom r(0, selected_pixels_y.dim(0).extent(), "r");

            // The warped value for pixel i
            Expr warped_val = warp_pixel_y(r);

            // The corresponding template value
            Expr tmpl_x = min(selected_pixels_y(r, 0), input_template.width()  - 1);
            Expr tmpl_y = min(selected_pixels_y(r, 1), input_template.height() - 1);
            Expr template_val = cast<float>( input_template(tmpl_x, tmpl_y) );

            // Residual
            Expr residual = template_val - warped_val;

            // Accumulate Jᵀ * residual
            // selected_jacobians(r,c) is the c-th channel of the Jacobian at pixel i.
            reduce_4_y(0) += selected_jacobians_y(r, 0) * residual;
            reduce_4_y(1) += selected_jacobians_y(r, 1) * residual;
            reduce_4_y(2) += selected_jacobians_y(r, 2) * residual;
            reduce_4_y(3) += selected_jacobians_y(r, 3) * residual;
        }

        // Finally map reduce_4(c) to output(x) with x in [0..3].
        output(x) = (reduce_4_x(x) + reduce_4_y(x)) * 0.5f;
    }

    void schedule()
    {
        input_template.dim(0).set_estimate(128, expected_width);
        input_template.dim(1).set_estimate(128, expected_height);
        input_keyframe.dim(0).set_estimate(128, expected_width);
        input_keyframe.dim(1).set_estimate(128, expected_height);
        selected_pixels_x.dim(0).set_estimate(800, 2000);
        selected_pixels_x.dim(1).set_estimate(2, 2);
        selected_pixels_y.dim(0).set_estimate(800, 2000);
        selected_pixels_y.dim(1).set_estimate(2, 2);
        selected_jacobians_x.dim(0).set_estimate(800, 2000);
        selected_jacobians_x.dim(1).set_estimate(4, 4);
        selected_jacobians_y.dim(0).set_estimate(800, 2000);
        selected_jacobians_y.dim(1).set_estimate(4, 4);

        A.set_estimate(0.1f);
        B.set_estimate(0.1f);
        TX.set_estimate(1.f);
        TY.set_estimate(1.f);

        // output is a length-4 vector
        output.set_estimate(x, 4, 4);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#endif
        if (auto_schedule) {
            return;
        }

        if (get_target().has_gpu_feature())
        {
            // If needed, add GPU scheduling here
            output.compute_root().gpu_tile(x, xo, xi, 4);
        }
        else
        {
            apply_schedule_sparse_ica(get_pipeline(), get_target());
        }
    }
};

HALIDE_REGISTER_GENERATOR(sparse_ica_generator, sparse_ica)

////////////////////////////////////////////////////////////////////////////////

#include "schedules/sparse_warpdiff.schedule.h"

class sparse_warpdiff_generator : public Halide::Generator<sparse_warpdiff_generator>
{
public:
    Input<Buffer<uint8_t>> input_template{"input_template", 2};
    Input<Buffer<uint8_t>> input_keyframe{"input_keyframe", 2}; // to be warped
    Input<Buffer<uint16_t>> local_max{"local_max", 3};
    // 4-parameter similarity warp
    Input<float> A{"A"};
    Input<float> B{"B"};
    Input<float> TX{"TX"};
    Input<float> TY{"TY"};

    Output<Buffer<uint16_t>> output{"output", 2};

    void generate()
    {
        // Boundary-condition for reading the keyframe
        Func clamped = BoundaryConditions::repeat_edge(input_keyframe);

        // The selected pixel's original location:
        Expr tile_x = min(local_max(x, y, 0), input_keyframe.width() - 1);
        Expr tile_y = min(local_max(x, y, 1), input_keyframe.height() - 1);

        Expr orig_x = cast<float>( tile_x );
        Expr orig_y = cast<float>( tile_y );

        // Compute warped location (Wx, Wy) in the keyframe
        Expr Wx = (1.0f + A)*orig_x - B*orig_y + TX;
        Expr Wy = B*orig_x + (1.0f + A)*orig_y + TY;

        // Floor and fractional parts
        Expr floorWx = floor(Wx);
        Expr floorWy = floor(Wy);
        Expr fracWx  = Wx - floorWx;
        Expr fracWy  = Wy - floorWy;

        // A 5×5 sampling kernel using Lanczos2 in 2D
        RDom rxy(0, 5, 0, 5, "rxy"); // kernel radius=2 => 5 taps
        Expr rx = rxy.x - 2;        // in [-2..2]
        Expr ry = rxy.y - 2;        // in [-2..2]

        // Compute 2D weights
        Expr distx = rx - fracWx;
        Expr disty = ry - fracWy;
        Expr w_x   = lanczos2(distx);
        Expr w_y   = lanczos2(disty);
        Expr w2D   = w_x * w_y;

        // Sample from keyframe at (floorWx+rx, floorWy+ry)
        // with boundary checks
        Expr sample_x = cast<int>(floorWx) + rx;
        Expr sample_y = cast<int>(floorWy) + ry;
        Expr val = cast<float>( clamped(sample_x, sample_y) );

        // Sum up w2D*val and w2D separately
        Expr sum_num = sum(w2D * val);
        Expr sum_den = sum(w2D);

        Expr interpolated = sum_num / sum_den;

        Expr diff = abs(interpolated - input_template(tile_x, tile_y));
        output(x, y) = cast<uint16_t>( clamp(diff, 0.0f, 65535.0f) );
    }

    void schedule()
    {
        input_template.dim(0).set_estimate(128, expected_width);
        input_template.dim(1).set_estimate(128, expected_height);
        input_keyframe.dim(0).set_estimate(128, expected_width);
        input_keyframe.dim(1).set_estimate(128, expected_height);
        local_max.dim(0).set_estimate(40, 45);
        local_max.dim(1).set_estimate(22, 25);

        A.set_estimate(0.1f);
        B.set_estimate(0.1f);
        TX.set_estimate(1.f);
        TY.set_estimate(1.f);

        // output is a length-4 vector
        output.set_estimate(x, 40, 45);
        output.set_estimate(y, 22, 25);

#if HALIDE_VERSION_MAJOR >= 15
        bool auto_schedule = using_autoscheduler();
#endif
        if (auto_schedule) {
            return;
        }

        if (get_target().has_gpu_feature())
        {
            // If needed, add GPU scheduling here
            output.compute_root().gpu_tile(x, xo, xi, 4);
        }
        else
        {
            apply_schedule_sparse_warpdiff(get_pipeline(), get_target());
        }
    }
};

HALIDE_REGISTER_GENERATOR(sparse_warpdiff_generator, sparse_warpdiff)
