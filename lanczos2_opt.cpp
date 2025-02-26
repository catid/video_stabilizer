#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

//------------------------------------------------------------------------------
// Baseline version: lanczos2(x) = sinc(x)*sinc(x/2)
// where sinc(x) = sin(pi*x)/(pi*x), with sinc(0)=1
//------------------------------------------------------------------------------
static inline double baseline_sinc(double x)
{
    if (x == 0.0) {
        return 1.0;
    }
    double z = M_PI * x; // pi*x
    return std::sin(z) / z;
}

static inline double baseline_lanczos2(double x)
{
    if (std::fabs(x) >= 2.0) {
        return 0.0;
    }
    return baseline_sinc(x) * baseline_sinc(0.5 * x);
}

//------------------------------------------------------------------------------
// 1) A small timing harness: measure median microseconds
//    of 10 repeated "blocks" of 100 calls each.
//------------------------------------------------------------------------------
template <typename Func>
double measure_median_runtime(Func func, double test_x)
{
    using clock = std::chrono::high_resolution_clock;

    constexpr int NUM_BLOCKS = 10;
    constexpr int CALLS_PER_BLOCK = 100;

    std::vector<double> times;
    times.reserve(NUM_BLOCKS);

    for (int b = 0; b < NUM_BLOCKS; b++) {
        auto t0 = clock::now();

        double dummy = 0.0;
        for (int i = 0; i < CALLS_PER_BLOCK; i++) {
            // Add a tiny variation to 'test_x' each call
            dummy += func(test_x + i * 1e-9);
        }

        auto t1 = clock::now();

        // Duration in microseconds
        double micros = std::chrono::duration<double, std::micro>(t1 - t0).count();
        times.push_back(micros);

        // Prevent optimization
        if (std::fabs(dummy) < 0.0) {
            std::cout << "";
        }
    }

    // Sort times and pick the median
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

//------------------------------------------------------------------------------
// 2) Utility: solve a small linear system using naive Gauss elimination
//    This is just for demonstration. For larger systems, use a robust library.
//------------------------------------------------------------------------------
bool solve_linear_system(std::vector<double> &A, std::vector<double> &b, int n)
{
    // A is n*n, b is n
    // We'll do partial pivoting for a bit more robustness
    for (int i = 0; i < n; i++) {
        // Find pivot
        double max_abs = std::fabs(A[i*n + i]);
        int pivot = i;
        for (int r = i+1; r < n; r++) {
            double val = std::fabs(A[r*n + i]);
            if (val > max_abs) {
                max_abs = val;
                pivot = r;
            }
        }
        if (max_abs < 1e-14) {
            return false; // Degenerate
        }
        if (pivot != i) {
            // swap rows pivot <-> i
            for (int c = 0; c < n; c++) {
                std::swap(A[i*n + c], A[pivot*n + c]);
            }
            std::swap(b[i], b[pivot]);
        }

        // Eliminate below
        double diag = A[i*n + i];
        for (int r = i+1; r < n; r++) {
            double factor = A[r*n + i] / diag;
            A[r*n + i] = 0.0;
            for (int c = i+1; c < n; c++) {
                A[r*n + c] -= factor * A[i*n + c];
            }
            b[r] -= factor * b[i];
        }
    }

    // Back substitution
    for (int i = n-1; i >= 0; i--) {
        double sum = b[i];
        for (int c = i+1; c < n; c++) {
            sum -= A[i*n + c] * b[c];
        }
        b[i] = sum / A[i*n + i];
    }

    return true;
}

//------------------------------------------------------------------------------
// 3) Fit a polynomial for lanczos2(x), exploiting even symmetry
//    We choose a polynomial in x^2 of the form:
//
//       P(x) = a0 + a1*x^2 + a2*x^4 + ... + aN*x^(2N)
//
//    We only need to fit [0..2], because the function is even:
//       lanczos2(-x) == lanczos2(x)
//
//    This example uses a simple least-squares approach on M sample points.
//------------------------------------------------------------------------------
std::vector<double> fit_even_polynomial_lanczos2(int poly_degree,
                                                 int num_samples)
{
    // poly_degree = number of even terms minus 1
    //   e.g. if poly_degree = 4, we have: a0 + a1*x^2 + a2*x^4 + a3*x^6 + a4*x^8

    // We'll sample the function at 'num_samples' points in [0..2].
    // Then set up the system A * coeffs = b for least squares.
    // Dimensions:
    //   A is num_samples x (poly_degree+1), but we typically solve
    //   (A^T * A) * coeffs = (A^T * b), which is (poly_degree+1) x (poly_degree+1).

    // 1) Collect sample points
    std::vector<double> xs(num_samples);
    std::vector<double> ys(num_samples);
    double step = 2.0 / (num_samples - 1); // from 0..2

    for (int i = 0; i < num_samples; i++) {
        double x = i * step; // 0..2
        xs[i] = x;
        ys[i] = baseline_lanczos2(x); // exact values
    }

    // 2) Construct the matrix A (num_samples x (deg+1)) and vector b
    //    Then we build the normal equations: (A^T A) a = (A^T b).
    int N = poly_degree + 1; // number of unknown coefficients
    std::vector<double> ATA(N*N, 0.0);
    std::vector<double> ATb(N, 0.0);

    // Fill A row by row, but we don't store A fully; we accumulate A^T*A, A^T*b
    for (int i = 0; i < num_samples; i++) {
        double x2 = xs[i] * xs[i];
        // basis vector = [1, x^2, x^4, x^6, ..., x^(2*poly_degree)]
        // We can generate these powers on the fly:
        std::vector<double> basis(N);
        basis[0] = 1.0;
        for (int p = 1; p < N; p++) {
            basis[p] = basis[p-1] * x2;
        }

        // Add to ATA = A^T * A
        for (int r = 0; r < N; r++) {
            for (int c = r; c < N; c++) { // symmetrical
                ATA[r*N + c] += basis[r] * basis[c];
            }
        }

        // Add to ATb = A^T * y
        double y = ys[i];
        for (int r = 0; r < N; r++) {
            ATb[r] += basis[r] * y;
        }
    }

    // Fill the symmetric part of ATA
    for (int r = 0; r < N; r++) {
        for (int c = r+1; c < N; c++) {
            ATA[c*N + r] = ATA[r*N + c];
        }
    }

    // 3) Solve the linear system (ATA)(coeffs) = ATb for 'coeffs'
    //    We'll store the result in 'coeffs'.
    std::vector<double> coeffs(ATb.begin(), ATb.end()); // copy ATb
    if (!solve_linear_system(ATA, coeffs, N)) {
        std::cerr << "Fitting system is singular or ill-conditioned.\n";
        return {}; // empty => signal failure
    }

    return coeffs;
}

//------------------------------------------------------------------------------
// 4) Evaluate the fitted polynomial with the given coefficients
//    P(x) = a0 + a1*x^2 + a2*x^4 + ...
//------------------------------------------------------------------------------
static inline double eval_even_poly(double x, const std::vector<double> &coeffs)
{
    double ax2 = x * x;
    double sum = coeffs[0]; // a0
    double pow_x2 = ax2;
    for (size_t i = 1; i < coeffs.size(); i++) {
        sum += coeffs[i] * pow_x2;
        pow_x2 *= ax2;
    }
    return sum;
}

//------------------------------------------------------------------------------
// 5) Our new "on‐the‐fly" polynomial approximation for lanczos2(x):
//    - We fit an even polynomial in [0..2]
//    - For |x| >= 2 => 0
//    - For x=0 => 1
//    - Otherwise => evaluate P(|x|)
//------------------------------------------------------------------------------
struct AutoPolyLanczos
{
    std::vector<double> coeffs; // a0, a1, a2, ...
};

// Evaluate the approximation
static inline double lanczos2_auto_poly(double x, const AutoPolyLanczos &ap)
{
    double ax = std::fabs(x);
    if (ax >= 2.0) {
        return 0.0;
    }
    if (ax < 1e-15) {
        return 1.0;
    }
    return eval_even_poly(ax, ap.coeffs);
}

// ---------------------------------------------------------------------
// Hard-coded polynomial approximation for Lanczos2(x).
//
// Uses the coefficients from your fit:
//    a0 = 0.999861
//    a1 = -2.05238
//    a2 = 1.52229
//    a3 = -0.583468
//    a4 = 0.128693
//    a5 = -0.0158853
//    a6 = 0.000858519
//
// This represents an even polynomial up to x^12:
//    P(x) = a0 + a1*x^2 + a2*x^4 + a3*x^6 + a4*x^8 + a5*x^10 + a6*x^12
// ---------------------------------------------------------------------
static inline double lanczos2_auto_poly_hardcoded(double x)
{
    // We only define it in the range |x| < 2.0.
    // Outside, it is zero by definition of the Lanczos2 filter.
    // For x=0, we return 1.0 (sinc(0)*sinc(0)=1).
    double ax = std::fabs(x);
    if (ax >= 2.0) {
        return 0.0;
    }
    if (ax < 1e-15) {
        return 1.0;
    }

    // We evaluate P(x) = a0 + a1*x^2 + a2*x^4 + ...
    // via Horner's method on x^2 for efficiency.
    double x2 = ax * ax;

    // Start from the highest coefficient:
    double val = 0.000858519;                      // a6
    val = -0.0158853 + val * x2;                   // a5 + a6*x^2
    val = 0.128693   + val * x2;                   // a4 + ...
    val = -0.583468  + val * x2;                   // a3 + ...
    val = 1.52229    + val * x2;                   // a2 + ...
    val = -2.05238   + val * x2;                   // a1 + ...
    val = 0.999861   + val * x2;                   // a0 + ...

    return val; // P(|x|)
}


//------------------------------------------------------------------------------
// Main test driver
//------------------------------------------------------------------------------
int main()
{
    // 1) Fit a polynomial of chosen degree
    //    For example, let's do up to x^(2*6) => 7 terms: a0, a1*x^2, ..., a6*x^12
    int poly_degree = 6;
    // Number of sample points in [0..2]
    int num_samples = 200;

    std::vector<double> auto_poly_coeffs = fit_even_polynomial_lanczos2(poly_degree, num_samples);
    if (auto_poly_coeffs.empty()) {
        // If we failed, just exit
        std::cerr << "Polynomial fitting failed!\n";
        return 1;
    }
    
    // Store in a struct for convenience
    AutoPolyLanczos autoPoly{ auto_poly_coeffs };

    // 2) Evaluate error vs baseline over a dense grid in [-2..2]
    double max_err = 0.0;
    double sum_err = 0.0;
    int count = 0;

    double step = 1e-4;
    for (double x = -2.0; x <= 2.0; x += step) {
        double exact = baseline_lanczos2(x);
        double approx = lanczos2_auto_poly_hardcoded(x);
        double err = std::fabs(approx - exact);
        if (err > max_err) {
            max_err = err;
        }
        sum_err += err;
        count++;
    }
    double avg_err = sum_err / count;

    // 3) Benchmark speed
    double test_x = 1.2345;
    double t_baseline = measure_median_runtime(baseline_lanczos2, test_x);

    // We'll capture the environment (autoPoly) by copy in the lambda:
    auto auto_poly_fun = [autoPoly](double x){ return lanczos2_auto_poly_hardcoded(x); };
    double t_auto_poly = measure_median_runtime(auto_poly_fun, test_x);

    // 4) Print results
    std::cout << "=== On-the-fly Polynomial Approximation of Lanczos2 ===\n";
    std::cout << "Polynomial degree: " << poly_degree
              << " (highest power: x^" << 2*poly_degree << ")\n";
    std::cout << "Num samples used for fitting: " << num_samples << "\n";

    std::cout << "\nFitted coefficients (a0, a1, ..., aN):\n";
    for (size_t i = 0; i < autoPoly.coeffs.size(); i++) {
        std::cout << "  a" << i << " = " << autoPoly.coeffs[i] << "\n";
    }

    std::cout << "\nErrors in [-2..2]:\n";
    std::cout << "  Max error: " << max_err << "\n";
    std::cout << "  Avg error: " << avg_err << "\n";

    std::cout << "\nMedian runtime of 10 blocks of 100 calls (microseconds):\n";
    std::cout << "  Baseline    : " << t_baseline << " us\n";
    std::cout << "  Auto-Poly   : " << t_auto_poly << " us\n";

    std::cout << "\nSpeedup (relative to baseline): "
              << t_baseline / t_auto_poly << "x\n";

    return 0;
}

/*
=== On-the-fly Polynomial Approximation of Lanczos2 ===
Polynomial degree: 6 (highest power: x^12)
Num samples used for fitting: 200

Fitted coefficients (a0, a1, ..., aN):
  a0 = 0.999861
  a1 = -2.05238
  a2 = 1.52229
  a3 = -0.583468
  a4 = 0.128693
  a5 = -0.0158853
  a6 = 0.000858519

Errors in [-2..2]:
  Max error: 0.000383624
  Avg error: 0.000101414

Median runtime of 10 blocks of 100 calls (microseconds):
  Baseline    : 14.35 us
  Auto-Poly   : 5.245 us

Speedup (relative to baseline): 2.73594x
*/
