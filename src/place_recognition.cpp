#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <vector>
#include <limits>
#include <math.h>

#define PI 3.141592653589793

typedef cv::Mat FEATURE_T; // cv::Mat used for histograms
typedef double FEATURE_DISTANCE_T; // double used for distance between histograms

FEATURE_T extractFeatures(cv::Mat imgIn);
FEATURE_DISTANCE_T computeFeatureDistance(FEATURE_T testFeatures, FEATURE_T refFeatures);
double computeConditionalProb(FEATURE_DISTANCE_T distance);
void ShowManyImages(cv::String title, int nArgs, ...);

int main( int argc, char** argv )
{
    // Initialize variables
    std::vector<std::vector<cv::Mat>> refImgs;
    std::vector<std::vector<FEATURE_T>> refImageData; // First index: class label; second index: feature number
    cv::Mat testImg;
    FEATURE_T testImgFeatures;
    std::vector<cv::String> refFilenames;
    std::vector<cv::String> refFolderpaths;
    std::vector<FEATURE_DISTANCE_T> minDistance;
    std::vector<double> posteriorProbs;
    std::vector<double> priorProbs;
    refFolderpaths.push_back("../images/place0");
    refFolderpaths.push_back("../images/place1");
    cv::String testImageFilepath = cv::String(argv[1]);
    const unsigned int numClasses = refFolderpaths.size();
    refImgs.resize(numClasses);
    refImageData.resize(numClasses);
    posteriorProbs.resize(numClasses);
    priorProbs.resize(numClasses, 1.0/(float)numClasses); // *** Uniform priors, for now

    // Load reference images and extract and record features
    for(unsigned int i=0; i<numClasses; i++)
    {
        cv::glob(refFolderpaths.at(i), refFilenames);
        for(unsigned int j=0; j<refFilenames.size(); j++)
        {
            cv::Mat img;
            img = cv::imread(refFilenames.at(j));
            refImageData.at(i).push_back(extractFeatures(img));
            refImgs.at(i).push_back(img);
        }
    }

    // Load test image and extract features
    testImg = cv::imread(testImageFilepath);
    testImgFeatures = extractFeatures(testImg);

    // Compute distance between features in test image and all reference images and record minimum distance for each class
    minDistance.resize(numClasses);
    for(unsigned int i=0; i<numClasses; i++)
    {
        minDistance.at(i) = std::numeric_limits<FEATURE_DISTANCE_T>::infinity();
        for(unsigned int j=0; j<refImageData.at(i).size(); j++)
        {
            FEATURE_DISTANCE_T candidateDistance = computeFeatureDistance(testImgFeatures, refImageData.at(i).at(j));
            if(candidateDistance < minDistance.at(i))
            {
                minDistance.at(i) = candidateDistance;
            }
        }
    }

    // Compute probabilities for each class, based on the minimum feature distance found between the test image and each class
    for(unsigned int i=0; i<numClasses; i++)
    {
        posteriorProbs.at(i) = computeConditionalProb(minDistance.at(i))*priorProbs.at(i);
        printf("place %u prob = %lf\n",i,posteriorProbs.at(i));
    }

    ShowManyImages(cv::String("Place 0"), 4, refImgs.at(0).at(0), refImgs.at(0).at(1), refImgs.at(0).at(2), refImgs.at(0).at(3));
    ShowManyImages(cv::String("Place 1"), 4, refImgs.at(1).at(0), refImgs.at(1).at(1), refImgs.at(1).at(2), refImgs.at(1).at(3));
    cv::namedWindow("Test Image",cv::WINDOW_NORMAL);
    cv::resizeWindow("Test Image", 800, 600);
    cv::imshow("Test Image", testImg);
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}

FEATURE_T extractFeatures(cv::Mat imgIn)
{
    // Hue-Saturation histogram
    FEATURE_T histogram;
    cv::Mat hsvImg;
    int channels[] = {0,1};
    int histSize[] = {50,60}; // {hue bins, saturation bins}
    float hRanges[] = {0,180};
    float sRanges[] = {0,256};
    const float* ranges[] = {hRanges,sRanges};

    cv::cvtColor(imgIn, hsvImg, cv::COLOR_BGR2HSV);
    cv::calcHist(&hsvImg, 1, channels, cv::Mat(), histogram, 2, histSize, ranges, true, false);
    cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    return histogram;
}

FEATURE_DISTANCE_T computeFeatureDistance(FEATURE_T testFeatures, FEATURE_T refFeatures)
{
    // Hue-Saturation histogram
    return cv::compareHist(refFeatures, testFeatures, CV_COMP_BHATTACHARYYA);
}

double computeConditionalProb(FEATURE_DISTANCE_T distance)
{
    // Gaussian conditional probability distribution
    double sigmaSquared = pow(1.0/3.0, 2.0);
    return 1.0/sqrt(2.0*PI*sigmaSquared)*exp(-pow(distance, 2.0)/(2.0*sigmaSquared));
}

void ShowManyImages(cv::String title, int nArgs, ...)
{
    using namespace std;
    using namespace cv;
    int size;
    int i;
    int m, n;
    int x, y;

    // w - Maximum number of images in a row
    // h - Maximum number of images in a column
    int w, h;

    // scale - How much we have to resize the image
    float scale;
    int max;

    // If the number of arguments is lesser than 0 or greater than 12
    // return without displaying
    if(nArgs <= 0) {
        printf("Number of arguments too small....\n");
        return;
    }
    else if(nArgs > 14) {
        printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
        return;
    }
    // Determine the size of the image,
    // and the number of rows/cols
    // from number of arguments
    else if (nArgs == 1) {
        w = h = 1;
        size = 300;
    }
    else if (nArgs == 2) {
        w = 2; h = 1;
        size = 300;
    }
    else if (nArgs == 3 || nArgs == 4) {
        w = 2; h = 2;
        size = 300;
    }
    else if (nArgs == 5 || nArgs == 6) {
        w = 3; h = 2;
        size = 200;
    }
    else if (nArgs == 7 || nArgs == 8) {
        w = 4; h = 2;
        size = 200;
    }
    else {
        w = 4; h = 3;
        size = 150;
    }

    // Create a new 3 channel image
    Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8UC3);

    // Used to get the arguments passed
    va_list args;
    va_start(args, nArgs);

    // Loop for nArgs number of arguments
    for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
        // Get the Pointer to the IplImage
        Mat img = va_arg(args, Mat);

        // Check whether it is NULL or not
        // If it is NULL, release the image, and return
        if(img.empty()) {
            printf("Invalid arguments");
            return;
        }

        // Find the width and height of the image
        x = img.cols;
        y = img.rows;

        // Find whether height or width is greater in order to resize the image
        max = (x > y)? x: y;

        // Find the scaling factor to resize the image
        scale = (float) ( (float) max / size );

        // Used to Align the images
        if( i % w == 0 && m!= 20) {
            m = 20;
            n+= 20 + size;
        }

        // Set the image ROI to display the current image
        // Resize the input image and copy the it to the Single Big Image
        Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
        Mat temp; resize(img,temp, Size(ROI.width, ROI.height));
        temp.copyTo(DispImage(ROI));
    }

    // Create a new window, and show the Single Big Image
    namedWindow( title, 1 );
    imshow( title, DispImage);
    //waitKey();

    // End the number of arguments
    va_end(args);
}
