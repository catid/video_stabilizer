#include "stabilizer.hpp"

#include <opencv2/opencv.hpp>
#include <HalideRuntime.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <mutex>
#include <atomic>

// ----------------------------------------------------------------------------------
// Utility helpers copied from grid_search_align.cpp
// ----------------------------------------------------------------------------------

// Compute median of a vector<double>
static double median(std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    double med = v[n];
    if (v.size() % 2 == 0)
    {
        std::nth_element(v.begin(), v.begin() + n - 1, v.end());
        med = 0.5 * (med + v[n - 1]);
    }
    return med;
}

// Measure jitter (median optical-flow magnitude) for a sequence of frames
static double measure_jitter(const std::vector<cv::Mat>& frames)
{
    if (frames.size() < 2) return 0.0;

    std::vector<double> meds;

    cv::Mat prevGray;
    cv::cvtColor(frames[0], prevGray, cv::COLOR_BGR2GRAY);

    for (size_t i = 1; i < frames.size(); ++i)
    {
        cv::Mat gray;
        cv::cvtColor(frames[i], gray, cv::COLOR_BGR2GRAY);

        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prevGray, gray, flow,
                                     0.5, 3, 15, 3, 5, 1.2, 0);

        std::vector<cv::Mat> comps(2);
        cv::split(flow, comps);
        cv::Mat mag;
        cv::magnitude(comps[0], comps[1], mag);

        mag = mag.reshape(1, 1);          // flatten to single row
        std::vector<float> magVec;
        mag.copyTo(magVec);
        if (!magVec.empty())
        {
            size_t n = magVec.size() / 2;
            std::nth_element(magVec.begin(), magVec.begin() + n, magVec.end());
            meds.push_back(magVec[n]);
        }

        prevGray = gray;
    }

    return median(meds);
}

// ----------------------------------------------------------------------------------

struct Combo
{
    // Smoother parameters
    int    lag;
    int    mem;
    double lambda;

    // Reset/decay parameters
    double min_disp;
    double max_disp;
    double min_decay;
    double max_decay;
};

// ----------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " video.mp4 [-j N]" << std::endl;
        return 1;
    }

    // ------------------------------------------------------------------
    // Parse CLI arguments
    // ------------------------------------------------------------------
    std::string videoPath;
    int jobCount = std::thread::hardware_concurrency();

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg == "-j" || arg == "--jobs")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value after " << arg << std::endl;
                return 1;
            }
            jobCount = std::max(1, std::atoi(argv[++i]));
        }
        else
        {
            videoPath = arg;
        }
    }

    if (videoPath.empty())
    {
        std::cerr << "Video path not provided" << std::endl;
        return 1;
    }

    // ------------------------------------------------------------------
    // Configure threading for Halide + OpenCV to avoid nested-parallelism
    // ------------------------------------------------------------------
    halide_set_num_threads(1);
    cv::setNumThreads(1);

    // ------------------------------------------------------------------
    // Load entire video into memory
    // ------------------------------------------------------------------
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Cannot open " << videoPath << std::endl;
        return 1;
    }

    std::vector<cv::Mat> frames;
    cv::Mat frm;
    while (cap.read(frm))
    {
        frames.push_back(frm.clone());
    }

    if (frames.size() < 2)
    {
        std::cerr << "Video too short" << std::endl;
        return 1;
    }

    double inputJitter = measure_jitter(frames);
    std::cout << "Input median jitter: " << inputJitter << " px" << std::endl;

    // ------------------------------------------------------------------
    // Parameter grid
    // ------------------------------------------------------------------

    // Smoother parameters to consider (same as old synthetic search)
    const int    lagVals[]   = {3, 5, 8, 10};
    const int    memVals[]   = {5, 8, 10};
    const double lambdaVals[] = {4.0, 6.0, 8.0, 10.0};

    // Displacement / decay parameters
    const double minDispVals[]  = {16.0, 32.0, 48.0};
    const double maxDispVals[]  = {64.0, 96.0, 128.0};
    const double minDecayVals[] = {0.99, 0.95, 0.9};
    const double maxDecayVals[] = {0.7, 0.5, 0.3};

    std::vector<Combo> combos;

    for (int lag : lagVals)
        for (int mem : memVals)
            for (double lam : lambdaVals)
                for (double mind : minDispVals)
                    for (double maxd : maxDispVals)
                        if (mind < maxd)
                            for (double mindec : minDecayVals)
                                for (double maxdec : maxDecayVals)
                                    if (mindec > maxdec)
                                        combos.push_back({lag, mem, lam, mind, maxd, mindec, maxdec});

    const size_t total = combos.size();

    std::cout << "Evaluating " << total << " parameter combinations using "
              << jobCount << " threads" << std::endl;

    // ------------------------------------------------------------------
    // Multithreaded grid search
    // ------------------------------------------------------------------

    std::atomic<size_t> nextIdx{0};
    std::atomic<size_t> doneCnt{0};

    double bestRatio = 1e9;
    Combo  bestCombo{0,0,0,0,0,0,0};
    std::mutex bestMutex;

    auto tStart = std::chrono::steady_clock::now();

    auto worker = [&]() {
        while (true)
        {
            size_t idx = nextIdx.fetch_add(1);
            if (idx >= total) break;

            const Combo& c = combos[idx];

            VideoStabilizerParams params;

            // Smoother configuration
            params.lag             = c.lag;
            params.smoother_memory = c.mem;
            params.lambda          = c.lambda;

            // Displacement / decay configuration
            params.min_disp  = c.min_disp;
            params.max_disp  = c.max_disp;
            params.min_decay = c.min_decay;
            params.max_decay = c.max_decay;

            VideoStabilizer stab(params);

            std::vector<cv::Mat> outs;
            for (const auto& f : frames)
            {
                cv::Mat o = stab.processFrame(f);
                if (!o.empty()) outs.push_back(o.clone());
            }

            if (outs.size() < 2) continue; // insufficient output frames

            double outJitter = measure_jitter(outs);
            double ratio = outJitter / inputJitter;

            size_t finished = ++doneCnt;

            {
                std::lock_guard<std::mutex> lk(bestMutex);

                auto now = std::chrono::steady_clock::now();
                double sec = std::chrono::duration<double>(now - tStart).count();

                std::cout << "[" << finished << "/" << total << "] "
                          << "lag=" << c.lag << " mem=" << c.mem << " lambda=" << c.lambda
                          << "  minDisp=" << c.min_disp << " maxDisp=" << c.max_disp
                          << "  minDecay=" << c.min_decay << " maxDecay=" << c.max_decay
                          << "  outJit=" << outJitter << "  ratio=" << ratio
                          << "  elapsed=" << sec << "s";

                if (ratio < bestRatio)
                {
                    bestRatio = ratio;
                    bestCombo = c;
                    std::cout << "  ** new best **";
                }

                std::cout << std::endl;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < jobCount; ++i) threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    // ------------------------------------------------------------------
    // Report best result
    // ------------------------------------------------------------------
    std::cout << "\nBest parameters:" << std::endl;
    std::cout << "  lag             = " << bestCombo.lag    << std::endl;
    std::cout << "  smoother_memory = " << bestCombo.mem    << std::endl;
    std::cout << "  lambda          = " << bestCombo.lambda << std::endl;
    std::cout << "  min_disp        = " << bestCombo.min_disp  << std::endl;
    std::cout << "  max_disp        = " << bestCombo.max_disp  << std::endl;
    std::cout << "  min_decay       = " << bestCombo.min_decay << std::endl;
    std::cout << "  max_decay       = " << bestCombo.max_decay << std::endl;
    std::cout << "  jitter ratio    = " << bestRatio << std::endl;

    return 0;
}
