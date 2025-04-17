#include "stabilizer.hpp"

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp> // Make sure to include OpenCV headers as needed
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    // ---------------- Command‑line arguments ----------------
    // Very light‑weight parsing for a handful of flags; unknown flags are ignored.

    std::string inputDir  = "../recordings";
    std::string outputDir = "output";
    std::string singleFile;

    VideoStabilizerParams params; // uses tuned defaults

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        auto next = [&](double &dst){ if(i+1<argc) dst = std::stod(argv[++i]); };
        auto nextInt = [&](int &dst){ if(i+1<argc) dst = std::stoi(argv[++i]); };

        if (arg == "--input")      { if(i+1<argc) inputDir  = argv[++i]; }
        else if (arg == "--output") { if(i+1<argc) outputDir = argv[++i]; }
        else if (arg == "--file")      { if(i+1<argc) singleFile = argv[++i]; }
        else if (arg == "--threshold")          next(params.aligner.threshold);
        else if (arg == "--smallest_fraction") {
            if (i+1<argc) params.aligner.smallest_fraction = std::stof(argv[++i]);
        }
        else if (arg == "--phase_corr") {
            params.aligner.phase_correlate = true;
            if(i+1<argc && argv[i+1][0] != '-') next(params.aligner.phase_correlate_threshold);
        }
        else if (arg == "--max_iters")          nextInt(params.aligner.max_iters);
        else if (arg == "--lambda")             next(params.lambda);
        else if (arg == "--crop")               nextInt(params.crop_pixels);
        else if (arg == "--lag")                nextInt(params.lag);
        // silently ignore unknown flags so the script can pass extras
    }

    // Clamp some params to sane bounds
    params.aligner.smallest_fraction = std::max(0.05f, std::min(0.9f, params.aligner.smallest_fraction));

    // Create the output directory if it doesn't exist (may be relative to cwd)
    try {
        if (!fs::exists(outputDir)) {
            fs::create_directory(outputDir);
            std::cout << "Created output directory: " << outputDir << std::endl;
        } else {
            std::cout << "Output directory already exists: " << outputDir << std::endl;
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating output directory: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::string> videoFiles;

    if (!singleFile.empty()) {
        // Use only the specified file
        videoFiles.push_back(singleFile);
    } else {
        // Collect all .mp4 files in the input directory
        try {
            if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
                std::cerr << "Error: Input directory does not exist or is not a directory.\n";
                return EXIT_FAILURE;
            }

            for (const auto& entry : fs::directory_iterator(inputDir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".mp4") {
                    videoFiles.push_back(entry.path().filename().string());
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error reading input directory: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }

        if (videoFiles.empty()) {
            std::cerr << "No .mp4 files found in the input directory: " << inputDir << std::endl;
            return EXIT_FAILURE;
        }
    }

    // 'params' already initialised from command‑line flags earlier.

    // Iterate over each video file
    for (const auto& videoFile : videoFiles) {
        std::string inputPath = singleFile.empty() ? (fs::path(inputDir) / videoFile).string()
                                                  : videoFile;
        std::cout << "\nProcessing video: " << inputPath << std::endl;

        // Open the video file
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file " << inputPath << std::endl;
            continue; // Skip to the next video file
        }

        // Retrieve video properties
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0.0) {
            std::cerr << "Warning: Unable to retrieve FPS for " << videoFile << ". Defaulting to 30 FPS." << std::endl;
            fps = 30.0; // Default FPS
        }
        const int crop = params.crop_pixels;
        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)) - crop * 2; // Crop both edges
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) - crop * 2; // Crop both edges

        int fourcc = cv::VideoWriter::fourcc('x', '2', '6', '4');

        // Define output video path
        std::string outputPath = fs::path(outputDir) / ("processed_" + fs::path(videoFile).filename().string());

        // Initialize VideoWriter
        cv::VideoWriter writer;
        writer.open(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);
        writer.set(cv::VIDEOWRITER_PROP_QUALITY, 90);

        if (!writer.isOpened()) {
            std::cerr << "Error: Could not open the output video for write: " << outputPath << std::endl;
            cap.release();
            continue; // Skip to the next video file
        }

        std::cout << "Input FPS: " << fps 
                  << " | Frame Size: " << frameWidth << "x" << frameHeight << std::endl;
        std::cout << "Writing processed video to: " << outputPath << std::endl;

        cv::Mat frame;
        cv::Mat processedFrame;
        int frameCount = 0;
        VideoStabilizer stabilizer(params);

        // Process each frame
        while (cap.read(frame)) {
            // Pass the frame to the processing function
            processedFrame = stabilizer.processFrame(frame);

            // Write the processed frame to the output video
            writer.write(processedFrame);

            frameCount++;

            // Optional: Display progress
            if (frameCount % 100 == 0) {
                std::cout << "Processed " << frameCount << " frames..." << std::endl;
            }
        }

        std::cout << "Finished processing " << frameCount << " frames for video: " << videoFile << std::endl;

        // Release resources
        cap.release();
        writer.release();
    }

    std::cout << "\nAll videos have been processed successfully." << std::endl;
    return EXIT_SUCCESS;
}
