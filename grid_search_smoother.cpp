#include "smoother.hpp"

#include <iostream>
#include <random>
#include <cmath>

struct Metrics {
    double rms_ratio;   // smoothed / raw
    double step_error;  // abs(output - 50)
};

// Evaluate given parameters and return metrics
Metrics evaluate(int lag, int mem, double lambda, unsigned seed = 42)
{
    constexpr int N = 100;
    constexpr double jitter = 2.0;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-jitter, jitter);

    L1SmootherCenter smoother(lag, mem, lambda);

    double err2_raw = 0.0;
    double err2_sm  = 0.0;
    int     n_out   = 0;

    for (int i = 0; i < N; ++i)
    {
        double clean = 100.0 * i / (N - 1);
        double noisy = clean + dist(rng);

        err2_raw += (noisy - clean) * (noisy - clean);

        SimilarityTransform out;
        if (smoother.update(SimilarityTransform{0,0,noisy,0}, out))
        {
            double e = out.TX - clean;
            err2_sm += e * e;
            ++n_out;
        }
    }

    double rms_raw = std::sqrt(err2_raw / N);
    double rms_sm  = std::sqrt(err2_sm  / n_out);

    // Step response evaluation
    L1SmootherCenter smoother_step(lag/2 + 1, mem/2 + 1, lambda);
    SimilarityTransform tmp;
    for (int i = 0; i < 20; ++i) smoother_step.update(SimilarityTransform{0,0,0,0}, tmp);
    smoother_step.update(SimilarityTransform{0,0,50,0}, tmp);
    SimilarityTransform out; bool ok=false;
    for (int i = 0; i < 15 && !ok; ++i) ok = smoother_step.update(SimilarityTransform{0,0,50,0}, out);
    double step_err = ok ? std::fabs(out.TX - 50.0) : 50.0; // penalize if not produced

    return { rms_sm / rms_raw, step_err };
}

int main()
{
    int best_lag=0, best_mem=0; double best_lambda=0; Metrics best{1e9,1e9};

    int lag_vals[]   = {3,5,8,10,12,15,18,20,25,30};
    int mem_vals[]   = {5,8,10,12,15,18,20,25,30};
    double lam_vals[] = {1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0,12.0,15.0};

    for(int lag : lag_vals)
    for(int mem : mem_vals)
    for(double lam: lam_vals)
    {
        auto m = evaluate(lag, mem, lam);
        double score = m.rms_ratio + 0.02 * m.step_error; // weight step error
        std::cout << "lag=" << lag << " mem=" << mem << " lambda=" << lam
                  << "  rms_ratio=" << m.rms_ratio << "  step_err=" << m.step_error
                  << "  score=" << score << std::endl;
        if(score < (best.rms_ratio + 0.02*best.step_error))
        {
            best = m; best_lag=lag; best_mem=mem; best_lambda=lam;
        }
    }

    std::cout << "\nBest parameters: lag=" << best_lag << " smoother_memory=" << best_mem
              << " lambda=" << best_lambda << std::endl;
    std::cout << "Best metrics: rms_ratio=" << best.rms_ratio
              << "  step_err=" << best.step_error << std::endl;

    return 0;
}
