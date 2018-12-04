#ifndef PLACE_RECOGNITION_HPP
#define PLACE_RECOGNITION_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <stdio.h>
#include <vector>
#include <limits>
#include <math.h>
#include <unistd.h>
#include <queue>
#include <string>

#define PI 3.141592653589793
#define PRIOR_PROB_INCREMENT 0.01

#define FEATURE_COLOR_HIST
//#define FEATURE_SUPERPIXEL_SEMANTIC

#ifdef FEATURE_COLOR_HIST
typedef cv::Mat FEATURE_T; // cv::Mat used for histograms
typedef double FEATURE_DISTANCE_T; // double used for distance between histograms
#endif // FEATURE_COLOR_HIST

#ifdef FEATURE_SUPERPIXEL_SEMANTIC
struct FEATURE_T
{
    std::vector<unsigned int> superpixelColors; // 0 = red, 1 = orange, 2 = yellow, 3 = green, 4 = blue, 5 = purple
    std::vector<cv::Point2d> superpixelCenters;
};
typedef double FEATURE_DISTANCE_T; // double used for distance between histograms
#endif // FEATURE_SUPERPIXEL_SEMANTIC

struct RESULTS_T
{
    unsigned int numPlace0Correct;
    unsigned int numPlace1Correct;
    unsigned int numPlace2Correct;
    unsigned int numPlace0False;
    unsigned int numPlace1False;
    unsigned int numPlace2False;
    double place0Prior;
    double place1Prior;
    double place2Prior;

    RESULTS_T()
    {
        numPlace0Correct = 0;
        numPlace1Correct = 0;
        numPlace2Correct = 0;
        numPlace0False = 0;
        numPlace1False = 0;
        numPlace2False = 0;
        place0Prior = 0.0;
        place1Prior = 0.0;
        place2Prior = 0.0;
    }

    RESULTS_T(const RESULTS_T& other) // Copy constructor
    {
        this->numPlace0Correct = other.numPlace0Correct;
        this->numPlace1Correct = other.numPlace1Correct;
        this->numPlace2Correct = other.numPlace2Correct;
        this->numPlace0False = other.numPlace0False;
        this->numPlace1False = other.numPlace1False;
        this->numPlace2False = other.numPlace2False;
        this->place0Prior = other.place0Prior;
        this->place1Prior = other.place1Prior;
        this->place2Prior = other.place2Prior;
    }
};

class PlaceRecognition
{
public:
    // Members
    std::queue<RESULTS_T> resultQueue;

    // Methods
    void run(int argc, char** argv);
    FEATURE_T extractFeatures(cv::Mat imgIn);
    FEATURE_DISTANCE_T computeFeatureDistance(FEATURE_T testFeatures, FEATURE_T refFeatures);
    double computeConditionalProb(FEATURE_DISTANCE_T distance);
    void ShowManyImages(cv::String title, int nArgs, ...);
};

#endif // PLACE_RECOGNITION_HPP
