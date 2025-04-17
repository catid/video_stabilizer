#pragma once

#include "imgproc.hpp"

#include <vector>
#include <cmath>
#include <algorithm>
#include <deque>

class L1SmootherCenter
{
public:
    // lagBehind  : past samples included in the causal window
    // lagAhead   : symmetric look‑ahead normally used by off‑line filters;
    //              here we approximate it by using additional past samples.
    // lambda_tx  : threshold for TX/TY step detection
    // lambda_rot : threshold for A/B (if <0, defaults to 0.5*lambda_tx)
    L1SmootherCenter(int lagBehind, int lagAhead,
                     double lambda_tx = 1.0,
                     double lambda_rot = -1.0);

    /**
     * Push a new measurement (for frame index = current total input - 1).
     * Returns an optional "finalized" transform if any frame is ready.
     * If none is ready, returns an identity transform and sets a flag.
     */
    bool update(const SimilarityTransform& meas,
                SimilarityTransform& outFinalized);

private:
    int m_lagBehind;          ///< how many past frames to include (window size‑1)
    int m_lagAhead;           ///< kept for backward compatibility (ignored in causal mode)
    double m_lambda_tx;       ///< lambda for translations (TX,TY)
    double m_lambda_rot;      ///< lambda for rotations/shear (A,B)

    std::vector<SimilarityTransform> m_measurements; ///< all raw measurements so far

    // Sliding‑window regression stats (for O(1) update)
    struct RunningStat {
        int n = 0;                 // window size
        double sum_y  = 0.0;       // Σ y
        double sum_xy = 0.0;       // Σ i*y
        std::deque<double> buf;    // y values (index = position in buf)

        void push(double y, int window);
        double predict_last() const;
    } m_statA, m_statB, m_statTX, m_statTY;

    int m_windowSize = 0; // = lagBehind + lagAhead
};