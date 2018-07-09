#include "opencv2/opencv.hpp"
#include <stdio.h>

//#define CAP_REF_IMAGE

int main(int, char**)
{
    cv::VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    cv::Mat frame;
    cv::namedWindow("colorFrame",1);
    cv::Mat grayFrame;
    cv::namedWindow("grayFrame",1);
#ifndef CAP_REF_IMAGE
    cv::Mat blurredFrame;
    cv::namedWindow("blurredFrame",1);
    cv::Mat fileRefFrame = cv::imread("ref_image.jpg",1);
    cv::Mat refFrame;
    cv::cvtColor(fileRefFrame, refFrame, cv::COLOR_BGR2GRAY);
    cv::namedWindow("refFrame",1);
    cv::Mat blurredRefFrame;
    cv::Mat diffFrame;
    cv::namedWindow("diffFrame",1);
    double diffImageFactor;
#endif // CAP_REF_IMAGE
    const double lambdaMax = 150.0;
    double lambda = lambdaMax;
    while(1)
    {
        cap >> frame; // get a new frame from camera
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
#ifdef CAP_REF_IMAGE
        cv::imshow("grayFrame",grayFrame);
        cv::waitKey(1000);
        cv::imwrite("ref_image.jpg", grayFrame);
        break;
#else
        cv::GaussianBlur(grayFrame, blurredFrame, cv::Size(0,0), lambda, lambda);
        cv::GaussianBlur(refFrame, blurredRefFrame, cv::Size(0,0), lambda, lambda);
        cv::subtract(blurredRefFrame, blurredFrame, diffFrame);
        diffImageFactor = cv::sum(diffFrame)[0]/((double)diffFrame.rows*(double)diffFrame.cols*255.0);
        lambda = diffImageFactor*lambdaMax;
        printf("diffImageFactor = %f; lambda = %f\n",diffImageFactor,lambda);
        cv::imshow("colorFrame", frame);
        cv::imshow("blurredFrame", blurredFrame);
        cv::imshow("grayFrame",grayFrame);
        cv::imshow("refFrame",blurredRefFrame);
        cv::imshow("diffFrame",diffFrame);
        cv::waitKey(20);
#endif // CAP_REF_IMAGE
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
