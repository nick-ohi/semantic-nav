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
};

class PlaceRecognition
{
public:
    // Members
    std::queue<RESULTS_T> resultsQueue;

    // Methods
    void run();
    FEATURE_T extractFeatures(cv::Mat imgIn);
    FEATURE_DISTANCE_T computeFeatureDistance(FEATURE_T testFeatures, FEATURE_T refFeatures);
    double computeConditionalProb(FEATURE_DISTANCE_T distance);
    void ShowManyImages(cv::String title, int nArgs, ...);
};

#endif // PLACE_RECOGNITION_HPP
