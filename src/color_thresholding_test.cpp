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
    cv::namedWindow("frame",1);
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
                                            {90,50,80}, // blue
                                            {131,20,80}, // purple
                                            {160,50,40} // upper red
                                            };
        ;
    unsigned int hsvUpperThresholds[6][3] = { // HSV
                                            {10,255,255}, // red
                                            {32,255,255}, // yellow
                                            {89,255,255}, // green
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

    /*cv::SimpleBlobDetector::Params blobDectectorParams;
    blobDectectorParams.minThreshold = 0;
    blobDectectorParams.maxThreshold = 255;
    blobDectectorParams.filterByArea = true;
    blobDectectorParams.minArea = 1000;
    blobDectectorParams.maxArea = 50000000;
    blobDectectorParams.filterByCircularity = true;
    blobDectectorParams.minCircularity = 0.01;
    blobDectectorParams.filterByConvexity = false;
    blobDectectorParams.filterByInertia = true;
    blobDectectorParams.minInertiaRatio = 0.0001;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> perColorKeypoints;
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(blobDectectorParams);*/
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> perColorContours;
    std::vector<cv::Vec4i> perColorHierarchy;
    cap >> frame; // get an initial frame from camera to get size info
    std::vector<cv::Mat> fillFrames;
    fillFrames.resize(NUM_COLORS);
    for(int i=0; i<NUM_COLORS; i++)
    {
        fillFrames.at(i) = cv::Mat(frame.size(), frame.type(), cv::Scalar(imageFillColors[i][0],imageFillColors[i][1],imageFillColors[i][2]));
    }
    while(1)
    {
        //keypoints.clear();
        contours.clear();
        hierarchy.clear();
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
            cv::medianBlur(binaryMask,binaryMask,3);
            fillFrames.at(i).copyTo(segmentedFrame,binaryMask);
            /*perColorKeypoints.clear();
            detector->detect(binaryMask,perColorKeypoints);
            keypoints.insert(keypoints.end(), perColorKeypoints.begin(), perColorKeypoints.end());*/
            perColorContours.clear();
            perColorHierarchy.clear();
            cv::findContours(binaryMask,perColorContours,perColorHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
            for(int j=0; j<perColorContours.size(); j++)
            {
                if(cv::contourArea(perColorContours.at(j)) > 600)
                {
                    contours.push_back(perColorContours.at(j));
                }
            }
        }
        //cv::drawKeypoints(frame,keypoints,frame,cv::Scalar(0,0,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        for(int k=0; k<contours.size(); k++)
        {
            cv::drawContours(frame,contours,k,cv::Scalar(0,0,255),2,8);
        }
        cv::imshow("frame", frame);
        //cv::imshow("binaryMaskFrame",binaryMask);
        cv::imshow("segmentedFrame",segmentedFrame);
        cv::waitKey(20);
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
