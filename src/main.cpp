#include <cstdlib>
#include <cassert>
#include <string>  
#include <filesystem>
#include <print>
#include <cstdio>
#include <exception>
#include <format>
#include <stdexcept>
#include <opencv2/core.hpp>
#include "calibrate.h"
#include "visualNavigation.h"

int main(int argc, char* argv [])
{
    const cv::String keys =
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this help message}"
        "{@input          | <none>   | path to input video or configuration XML}"
        "{calibrate c     |          | perform camera calibration for given configuration XML}"
        "{scenario s      | 6        | run visual navigation on input video with scenario type (4:flight, 5:tag, 6:room)}"
        "{interactive i   | 0        | interactivity (0:none, 1:last frame, 2:all frames)}"
        "{export e        |          | export video}";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Assignment 2");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    int scenario = parser.get<int>("scenario");
    int interactive = parser.get<int>("interactive");
    bool hasExport = parser.has("export");
    bool hasCalibrate = parser.has("calibrate");
    std::filesystem::path inputPath = parser.get<std::string>("@input");

    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    std::filesystem::path outputDirectory;
    if (hasExport)
    {
        std::filesystem::path appPath = parser.getPathToApplication();
        outputDirectory = appPath / ".." / "out";

        // Create output directory if we need to
        if (!std::filesystem::exists(outputDirectory))
        {
            std::println("Creating directory {}", outputDirectory.string());
            std::filesystem::create_directory(outputDirectory);
        }
    }

    try
    {
        if (hasCalibrate)
        {
            std::println("Calibrating camera");
            std::println("Configuration file: {}", inputPath.string());
            calibrateCamera(inputPath);
        }
        else
        {
            // Runtime validation (assertions vanish under NDEBUG in Release)
            if (scenario < 4 || scenario > 6)
                throw std::invalid_argument(std::format("unsupported scenario {} (supported: 4, 5, 6)", scenario));
            if (interactive < 0 || interactive > 2)
                throw std::invalid_argument(std::format("invalid interactive mode {} (supported: 0, 1, 2)", interactive));
            std::println("Running visual navigation");
            std::println("Input video: {}", inputPath.string());
            std::filesystem::path cameraPath = inputPath.parent_path() / "camera.xml"; 
            runVisualNavigationFromVideo(inputPath, cameraPath, scenario, interactive, outputDirectory);
        }
    }
    catch (const std::exception & e)
    {
        std::println(stderr, "Error: {}", e.what());
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::println(stderr, "Error: unknown exception");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

