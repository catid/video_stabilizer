#include "stabilizer.hpp"

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp> // Make sure to include OpenCV headers as needed
#include <vector>

namespace fs = std::filesystem;

int main() {
    // Define input and output directories
    std::string inputDir = "../recordings";
    std::string outputDir = "output";

    // Create the output directory if it doesn't exist
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

    // Collect all .mp4 files in the input directory
    std::vector<std::string> videoFiles;
    try {
        if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
            std::cerr << "Error: Input directory does not exist or is not a directory.\n";
            return EXIT_FAILURE;
        }

        for (const auto& entry : fs::directory_iterator(inputDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".mp4") {
                // Store the filename (or the full path if you prefer)
                videoFiles.push_back(entry.path().filename().string());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error reading input directory: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // If no .mp4 files are found, exit early
    if (videoFiles.empty()) {
        std::cerr << "No .mp4 files found in the input directory: " << inputDir << std::endl;
        return EXIT_FAILURE;
    }

    // Iterate over each video file
    for (const auto& videoFile : videoFiles) {
        std::string inputPath = fs::path(inputDir) / videoFile;
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
        int crop = 0;
        int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)) - crop * 2; // Crop both edges
        int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) - crop * 2; // Crop both edges

        int fourcc = cv::VideoWriter::fourcc('x', '2', '6', '4');

        // Define output video path
        std::string outputPath = fs::path(outputDir) / ("processed_" + videoFile);

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
        VideoStabilizer stabilizer;

        // Process each frame
        while (cap.read(frame)) {
            // Pass the frame to the processing function
            processedFrame = stabilizer.processFrame(frame, crop);

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
