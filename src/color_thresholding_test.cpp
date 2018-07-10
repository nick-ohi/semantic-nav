#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <vector>

#define NUM_COLORS 5

int main(int, char**)
{
    cv::VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    cv::Mat frame;
    cv::namedWindow("rawFrame",1);
    cv::Mat hsvFrame;
    cv::Mat binaryMask;
    cv::Mat binaryMask2;
    //cv::namedWindow("binaryMaskFrame",1);
    cv::Mat segmentedFrame;
    cv::namedWindow("segmentedFrame",1);
    unsigned int hsvLowerThresholds[6][3] = { // HSV
                                            {0,50,40}, // red
                                            {11,50,40}, // yellow
                                            {33,50,40}, // green
                                            {83,50,80}, // blue
                                            {131,20,80}, // purple
                                            {160,50,40} // upper red
                                            };
        ;
    unsigned int hsvUpperThresholds[6][3] = { // HSV
                                            {10,255,255}, // red
                                            {32,255,255}, // yellow
                                            {82,255,255}, // green
                                            {130,255,255}, // blue
                                            {159,255,255}, // purple
                                            {179,255,255} // upper red
                                            };
    unsigned int imageFillColors[5][3] = { // BGR
                                         {0,0,255}, // red
                                         {0,255,255}, // yellow
                                         {0,255,0}, // green
                                         {255,0,0}, // blue
                                         {255,0,170} // purple
                                         };
    cap >> frame; // get an initial frame from camera to get size info
    std::vector<cv::Mat> fillFrames;
    fillFrames.resize(NUM_COLORS);
    for(int i=0; i<NUM_COLORS; i++)
    {
        fillFrames.at(i) = cv::Mat(frame.size(), frame.type(), cv::Scalar(imageFillColors[i][0],imageFillColors[i][1],imageFillColors[i][2]));
    }
    while(1)
    {
        cap >> frame; // get a new frame from camera
        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
        segmentedFrame = cv::Mat(frame.size(), frame.type(), cv::Scalar(0,0,0));
        for(int i=0; i<NUM_COLORS; i++)
        {
            cv::inRange(hsvFrame,cv::Scalar(hsvLowerThresholds[i][0],hsvLowerThresholds[i][1],hsvLowerThresholds[i][2]),
                                 cv::Scalar(hsvUpperThresholds[i][0],hsvUpperThresholds[i][1],hsvUpperThresholds[i][2]),binaryMask);
            if(i==0) // red hue value wraps around 0 to 180
            {
                cv::inRange(hsvFrame,cv::Scalar(hsvLowerThresholds[5][0],hsvLowerThresholds[5][1],hsvLowerThresholds[5][2]),
                        cv::Scalar(hsvUpperThresholds[5][0],hsvUpperThresholds[5][1],hsvUpperThresholds[5][2]),binaryMask2);
                cv::add(binaryMask,binaryMask2,binaryMask);
            }
            fillFrames.at(i).copyTo(segmentedFrame,binaryMask);
        }
        cv::imshow("rawFrame", frame);
        //cv::imshow("binaryMaskFrame",binaryMask);
        cv::imshow("segmentedFrame",segmentedFrame);
        cv::waitKey(20);
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
