#include "stabilizer.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include "HalideRuntime.h"   // for halide_set_num_threads()

// Compute median of vector<double>
static double median(std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    size_t n = v.size()/2;
    std::nth_element(v.begin(), v.begin()+n, v.end());
    double med = v[n];
    if(v.size()%2==0) {
        std::nth_element(v.begin(), v.begin()+n-1, v.end());
        med = 0.5*(med + v[n-1]);
    }
    return med;
}

// Measure jitter (median optical-flow magnitude) for a sequence of frames
static double measure_jitter(const std::vector<cv::Mat>& frames)
{
    if(frames.size() < 2) return 0.0;

    std::vector<double> meds;
    cv::Mat prevGray;
    cv::cvtColor(frames[0], prevGray, cv::COLOR_BGR2GRAY);

    for(size_t i=1;i<frames.size();++i)
    {
        cv::Mat gray;
        cv::cvtColor(frames[i], gray, cv::COLOR_BGR2GRAY);

        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prevGray, gray, flow,
                                     0.5,3,15,3,5,1.2,0);

        std::vector<cv::Mat> comps(2);
        cv::split(flow, comps);
        cv::Mat mag;
        cv::magnitude(comps[0], comps[1], mag);

        mag = mag.reshape(1,1);
        std::vector<float> mvec; mag.copyTo(mvec);
        if(!mvec.empty()) {
            size_t n = mvec.size()/2;
            std::nth_element(mvec.begin(), mvec.begin()+n, mvec.end());
            meds.push_back(mvec[n]);
        }
        prevGray = gray;
    }

    return median(meds);
}

int main(int argc, char** argv)
{
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " video.mp4 [-j N]" << std::endl;
        return 1;
    }

    // ------------------------------------------------------------------
    // Parse CLI
    // ------------------------------------------------------------------

    std::string videoPath;
    int jobCount = std::thread::hardware_concurrency();

    for(int i=1;i<argc;++i) {
        std::string arg = argv[i];
        if(arg == "-j" || arg == "--jobs") {
            if(i+1 >= argc) { std::cerr << "Missing value after " << arg << std::endl; return 1; }
            jobCount = std::max(1, std::atoi(argv[++i]));
        } else {
            videoPath = arg;
        }
    }

    if(videoPath.empty()) {
        std::cerr << "Video path not provided" << std::endl;
        return 1;
    }

    // ------------------------------------------------------------------
    // Configure the underlying Halide thread-pool to avoid accidental
    // oversubscription (or under-subscription).  Halide uses a single
    // global thread-pool that defaults to std::thread::hardware_concurrency()
    // threads.  When we also spawn our own pool of worker threads, this can
    // easily lead to the application utilising just a single core: Halide
    // detects nested parallelism and falls back to serial execution to
    // prevent runaway thread creation.  As a consequence, the heavy image
    // processing done inside each worker ends up running single-threaded.
    //
    // We explicitly tell Halide to use *one* thread per pipeline here so
    // that the outer worker threads may all execute in parallel and fully
    // utilise the machine.
    // ------------------------------------------------------------------
    halide_set_num_threads(1);
    cv::VideoCapture cap(videoPath);
    if(!cap.isOpened()) {
        std::cerr << "Cannot open " << videoPath << std::endl;
        return 1;
    }

    // For OpenCV as well, disable its internal parallelism so that we keep
    // full control over how many OS threads are active.  Each worker thread
    // (spawned below) will now run the OpenCV algorithms single-threaded,
    // but since we have `jobCount` workers running in parallel the overall
    // utilisation scales while avoiding the nested-parallelism pitfalls that
    // otherwise keep the CPU usage low.
    cv::setNumThreads(1);

    // Load entire video into memory once (small videos assumed)
    std::vector<cv::Mat> frames;
    cv::Mat f;
    while(cap.read(f)) frames.push_back(f.clone());

    if(frames.size() < 2) {
        std::cerr << "Video too short." << std::endl;
        return 1;
    }

    double inputJitter = measure_jitter(frames);
    std::cout << "Input median jitter: " << inputJitter << " px\n";


    // Parameter grids (can tweak or expose via CLI later)
    bool phase_vals[] = {false, true};
    double thresh_vals[]   = {0.02, 0.03, 0.05};
    float frac_vals[]      = {0.3f, 0.5f, 0.8f};
    double maxDisp_vals[]  = {6.0, 8.0, 10.0};

    struct Combo { bool pc; double thr; float frac; double md; };
    std::vector<Combo> combos;
    for(bool pc : phase_vals)
    for(double thr : thresh_vals)
    for(float frac : frac_vals)
    for(double md : maxDisp_vals)
        combos.push_back({pc,thr,frac,md});

    const size_t total = combos.size();

    std::cout << "Running " << total << " parameter combinations using "
              << jobCount << " threads" << std::endl;

    std::atomic<size_t> nextIdx{0};
    std::atomic<size_t> doneCnt{0};

    auto t0 = std::chrono::steady_clock::now();
    std::mutex bestMutex; double bestRatio = 1e9; VideoAlignerParams bestP;

    auto worker = [&]() {
        while(true) {
            size_t idx = nextIdx.fetch_add(1);
            if(idx >= combos.size()) break;
            auto c = combos[idx];

            VideoStabilizerParams params;
            params.enable_smoother = false;
            params.lag = 1; params.smoother_memory = 0;

            params.aligner.phase_correlate    = c.pc;
            params.aligner.threshold          = c.thr;
            params.aligner.smallest_fraction  = c.frac;
            params.aligner.max_displacement   = c.md;

            VideoStabilizer stab(params);

            std::vector<cv::Mat> outs;
            for(const auto& frm: frames) {
                cv::Mat o = stab.processFrame(frm);
                if(!o.empty()) outs.push_back(o.clone());
            }
            if(outs.size()<2) continue;

            double outJit = measure_jitter(outs);
            double ratio  = outJit / inputJitter;

            size_t finished = ++doneCnt;

            {
                std::lock_guard<std::mutex> lk(bestMutex);

                auto now = std::chrono::steady_clock::now();
                double sec = std::chrono::duration<double>(now - t0).count();

                std::cout << "[" << finished << "/" << total << "] "
                          << "PC=" << c.pc << " thr=" << c.thr << " frac=" << c.frac
                          << " maxDisp=" << c.md << "  outJit=" << outJit
                          << "  ratio=" << ratio << "  elapsed=" << sec << "s" << std::endl;

                if(ratio < bestRatio) {
                    bestRatio = ratio;
                    bestP = params.aligner;
                    std::cout << "  ** New best so far! **" << std::endl;
                }
            }
        }
    };

    std::vector<std::thread> threads;
    for(int i=0;i<jobCount;++i) threads.emplace_back(worker);
    for(auto& t: threads) t.join();

    std::cout << "\nBest params: phase_correlate=" << bestP.phase_correlate
              << "  threshold=" << bestP.threshold
              << "  smallest_fraction=" << bestP.smallest_fraction
              << "  max_displacement=" << bestP.max_displacement
              << "  ratio=" << bestRatio << std::endl;

    return 0;
}
