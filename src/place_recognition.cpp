#include <place_recognition.hpp>

void PlaceRecognition::run()
{
    // Initialize variables
    std::vector<std::vector<cv::Mat>> refImgs;
    std::vector<std::vector<FEATURE_T>> refImageData; // First index: class label; second index: feature number
    std::vector<cv::Mat> testImgs;
    std::vector<FEATURE_T> testImageData;
    std::vector<cv::String> refFilenames;
    std::vector<cv::String> refFolderpaths;
    std::vector<cv::String> testFilenames;
    cv::String testFolderpath = "../images/Evansdale_small/test";
    unsigned int testImageTruthClass[18] = {0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,2};
    std::vector<unsigned int> testImageEstimatedClassCorrect;
    std::vector<std::vector<FEATURE_DISTANCE_T>> minDistance;
    std::vector<std::vector<double>> posteriorProbs;
    std::vector<std::vector<double>> priorProbs;
    refFolderpaths.push_back("../images/Evansdale_small/place0");
    refFolderpaths.push_back("../images/Evansdale_small/place1");
    refFolderpaths.push_back("../images/Evansdale_small/place2");
    const unsigned int numClasses = refFolderpaths.size();
    refImgs.resize(numClasses);
    refImageData.resize(numClasses);

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
            //usleep(500000);
            /*for(unsigned int k=0; k<refImageData.at(i).at(j).superpixelColors.size(); k++)
            {
                printf("refImage %u class %u superpixel %u colors = %u\n",j,i,k,refImageData.at(i).at(j).superpixelColors.at(k));
            }*/
        }
    }

    // Load test images and extract features
    cv::glob(testFolderpath, testFilenames);
    for(unsigned int j=0; j<testFilenames.size(); j++)
    {
        cv::Mat img;
        img = cv::imread(testFilenames.at(j));
        testImageData.push_back(extractFeatures(img));
        testImgs.push_back(img);
        //usleep(500000);
        /*printf("testImage %u: ",j);
        for(unsigned int k=0; k<testImageData.at(j).superpixelColors.size(); k++)
        {
            printf("%u ",testImageData.at(j).superpixelColors.at(k)); // color
            //printf("[%.2f,%.2f] ",testImageData.at(j).superpixelCenters.at(k).x,testImageData.at(j).superpixelCenters.at(k).y);
        }
        printf("\n");*/
    }

    // Compute distance between features in test images and all reference images and record minimum distance for each class
    minDistance.resize(testImageData.size());
    for(unsigned int a=0; a<minDistance.size(); a++)
    {
        minDistance.at(a).resize(numClasses);
    }
    for(unsigned int k=0; k<testImageData.size(); k++)
    {
        for(unsigned int i=0; i<numClasses; i++)
        {
            minDistance.at(k).at(i) = std::numeric_limits<FEATURE_DISTANCE_T>::infinity();
            for(unsigned int j=0; j<refImageData.at(i).size(); j++)
            {
                FEATURE_DISTANCE_T candidateDistance = computeFeatureDistance(testImageData.at(k), refImageData.at(i).at(j));
                if(candidateDistance < minDistance.at(k).at(i))
                {
                    minDistance.at(k).at(i) = candidateDistance;
                }
            }
        }
    }

    // Compute probabilities for each class, based on the minimum feature distance found between the test image and each class
    posteriorProbs.resize(testImageData.size(),std::vector<double>(numClasses));
    priorProbs.resize(testImageData.size(), std::vector<double>(numClasses, 1.0/(float)numClasses)); // *** Uniform priors, for now
    for(unsigned int k=0; k<testImageData.size(); k++)
    {
        //printf("image %u, ",k);
        for(unsigned int i=0; i<numClasses; i++)
        {
            //printf("minDistance(%u)(%u) = %lf\n",k,i,minDistance.at(k).at(i));
            posteriorProbs.at(k).at(i) = computeConditionalProb(minDistance.at(k).at(i))*priorProbs.at(k).at(i);
            //printf("place %u prob = %lf\t",i,posteriorProbs.at(k).at(i));
        }
        //printf("\n");
    }

    // List MAP estimated results and success rate
    testImageEstimatedClassCorrect.resize(testImageData.size());
    double successRate = 0.0;
    for(unsigned int k=0; k<testImageData.size(); k++)
    {
        double maxProb = 0.0;
        unsigned int estimatedPlace = 0;
        for(unsigned int i=0; i<numClasses; i++)
        {
            if(posteriorProbs.at(k).at(i) > maxProb)
            {
                maxProb = posteriorProbs.at(k).at(i);
                estimatedPlace = i;
            }
        }
        if(estimatedPlace == testImageTruthClass[k])
        {
            testImageEstimatedClassCorrect.at(k) = 1;
        }
        else
        {
            testImageEstimatedClassCorrect.at(k) = 0;
        }
        successRate += (double)testImageEstimatedClassCorrect.at(k);
        printf("image %u is: place %u, with prob %lf, correct = %u\n",k,estimatedPlace,maxProb,testImageEstimatedClassCorrect.at(k));
    }
    successRate /= (double)testImageData.size();
    printf("success rate = %lf\n",successRate);

    //ShowManyImages(cv::String("Place 0"), 4, refImgs.at(0).at(0), refImgs.at(0).at(1), refImgs.at(0).at(2), refImgs.at(0).at(3));
    //ShowManyImages(cv::String("Place 1"), 4, refImgs.at(1).at(0), refImgs.at(1).at(1), refImgs.at(1).at(2), refImgs.at(1).at(3));
    //cv::namedWindow("Test Image",cv::WINDOW_NORMAL);
    //cv::resizeWindow("Test Image", 800, 600);
    //cv::imshow("Test Image", testImg);
    cv::destroyAllWindows();
}

FEATURE_T PlaceRecognition::extractFeatures(cv::Mat imgIn)
{
#ifdef FEATURE_COLOR_HIST
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
#endif // FEATURE_COLOR_HIST

#ifdef FEATURE_SUPERPIXEL_SEMANTIC
    FEATURE_T imageFeatures;
    int num_iterations = 7; // 4
    int prior = 3; // 2
    bool double_step = false;
    int num_superpixels = 70; // 400
    int num_levels = 6; // 4
    int num_histogram_bins = 5;
    cv::Mat result, mask, converted, labels;
    cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds;
    int width, height;
    cv::Mat histogram;
    cv::Mat hsvImg;
    int channels = 0;
    int histSize = 12; // hue bins
    float hRanges[] = {0,180};
    const float* ranges[] = {hRanges};
    std::vector<unsigned int> colorHistogram;
    colorHistogram.resize(6); // 6 colors
    width = imgIn.size().width;
    height = imgIn.size().height;
    seeds = cv::ximgproc::createSuperpixelSEEDS(width, height, imgIn.channels(), num_superpixels, num_levels, prior, num_histogram_bins, double_step);
    cv::cvtColor(imgIn, converted, cv::COLOR_BGR2HSV);
    seeds->iterate(converted, num_iterations);
    result = imgIn;
    // Retrieve the segmentation result
    seeds->getLabels(labels);
    // Print output
    /*seeds->getLabelContourMask(mask, false);
    result.setTo(cv::Scalar(0, 0, 255), mask);
    cv::imshow("test",result);
    cv::waitKey(500);*/
    // Find superpixel centers and overall color
    unsigned int numSuperpixelsActual = seeds->getNumberOfSuperpixels();
    cv::Moments superpixelMoment;
    cv::Point2d superpixelCenter;
    cv::Mat singleSuperpixelBinaryFrame;
    imageFeatures.superpixelCenters.resize(numSuperpixelsActual);
    imageFeatures.superpixelColors.resize(numSuperpixelsActual);
    for(unsigned int i=0; i<numSuperpixelsActual; i++)
    {
        singleSuperpixelBinaryFrame = cv::Mat(labels.size(), labels.type(), cv::Scalar(0));
        cv::inRange(labels, cv::Scalar(i), cv::Scalar(i), singleSuperpixelBinaryFrame);
        superpixelMoment = cv::moments(singleSuperpixelBinaryFrame, true);
        superpixelCenter = cv::Point2d(superpixelMoment.m10/superpixelMoment.m00, superpixelMoment.m01/superpixelMoment.m00);
        cv::calcHist(&converted, 1, &channels, singleSuperpixelBinaryFrame, histogram, 1, &histSize, ranges, true, false);
        cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        colorHistogram.at(0) = histogram.at<unsigned long>(0) + histogram.at<unsigned long>(11); // red
        colorHistogram.at(1) = histogram.at<unsigned long>(1) + histogram.at<unsigned long>(2); // orange
        colorHistogram.at(2) = histogram.at<unsigned long>(3) + histogram.at<unsigned long>(4); // yellow
        colorHistogram.at(3) = histogram.at<unsigned long>(5) + histogram.at<unsigned long>(6); // green
        colorHistogram.at(4) = histogram.at<unsigned long>(7) + histogram.at<unsigned long>(8); // blue
        colorHistogram.at(5) = histogram.at<unsigned long>(9) + histogram.at<unsigned long>(10); // purple
        unsigned long maxColorCount = 0;
        unsigned int maxColorIndex = 0;
        for(unsigned int j=0; j<colorHistogram.size(); j++)
        {
            if(colorHistogram.at(j) > maxColorCount)
            {
                maxColorCount = colorHistogram.at(j);
                maxColorIndex = j;
            }
        }
        imageFeatures.superpixelCenters.at(i) = superpixelCenter;
        imageFeatures.superpixelColors.at(i) = maxColorIndex;
    }
    return imageFeatures;
#endif // FEATURE_SUPERPIXEL_SEMANTIC
}

FEATURE_DISTANCE_T PlaceRecognition::computeFeatureDistance(FEATURE_T testFeatures, FEATURE_T refFeatures)
{
#ifdef FEATURE_COLOR_HIST
    // Hue-Saturation histogram
    return cv::compareHist(refFeatures, testFeatures, CV_COMP_BHATTACHARYYA);
#endif // FEATURE_COLOR_HIST

#ifdef FEATURE_SUPERPIXEL_SEMANTIC
    auto findRelativePositionType = [](cv::Point2d testPos, cv::Point2d basePos) -> unsigned int // 0 = same,same; 1 = left,same; 2 = left,up; 3 = same,up; 4 = right,up; 5 = right,same; 6 = right,down; 7 = same,down; 8 = left,down
    {
        if(testPos.x == basePos.x && testPos.y == basePos.y) // 0 = same,same
        {
            return 0;
        }
        else if(testPos.x < basePos.x && testPos.y == basePos.y) // 1 = left,same
        {
            return 1;
        }
        else if(testPos.x < basePos.x && testPos.y < basePos.y) // 2 = left,up
        {
            return 2;
        }
        else if(testPos.x == basePos.x && testPos.y < basePos.y) // 3 = same,up
        {
            return 3;
        }
        else if(testPos.x > basePos.x && testPos.y < basePos.y) // 4 = right,up
        {
            return 4;
        }
        else if(testPos.x > basePos.x && testPos.y == basePos.y) // 5 = right,same
        {
            return 5;
        }
        else if(testPos.x > basePos.x && testPos.y > basePos.y) // 6 = right,down
        {
            return 6;
        }
        else if(testPos.x == basePos.x && testPos.y > basePos.y) // 7 = same,down
        {
            return 7;
        }
        else if(testPos.x < basePos.x && testPos.y > basePos.y) // 8 = left,down
        {
            return 8;
        }
        else // Failsafe
        {
            return 0;
        }
    };
    const unsigned int numColors = 6;
    const unsigned int numRelPositions = 9;
    unsigned int numSuperpixels = refFeatures.superpixelColors.size();
    unsigned int bestBaseSuperpixelIndex = 0;
    double bestBaseSuperpixelDistance = std::numeric_limits<double>::infinity();
    double candidateDistance;
    unsigned int refSemanticHistogram[numColors][numRelPositions] = {0}; // colors, relative positions
    // Ref image features
    for(unsigned int i=0; i<numSuperpixels; i++)
    {
        refSemanticHistogram[refFeatures.superpixelColors.at(i)]
                [findRelativePositionType(refFeatures.superpixelCenters.at(i),refFeatures.superpixelCenters.at(0))]++;
    }
    // Test image features
    for(unsigned int j=0; j<numSuperpixels; j++)
    {
        unsigned int testSemanticHistogram[numColors][numRelPositions] = {0}; // colors, relative positions
        for(unsigned int i=0; i<numSuperpixels; i++)
        {
            testSemanticHistogram[testFeatures.superpixelColors.at(i)]
                    [findRelativePositionType(testFeatures.superpixelCenters.at(i),testFeatures.superpixelCenters.at(j))]++;
        }
        double superpixelDistance;
        // If the color of the "base base" does not match between the ref and test image, distance is infinite, otherwise compute the distance
        if(testFeatures.superpixelColors.at(j) == refFeatures.superpixelColors.at(0)) // TODO: reconsider this condition
        {
            superpixelDistance = 0.0;
            for(unsigned int m=0; m<numColors; m++)
            {
                for(unsigned int n=0; n<numRelPositions; n++)
                {
                    superpixelDistance += fabs((double)testSemanticHistogram[m][n] - (double)refSemanticHistogram[m][n]);
                }
            }
        }
        else
        {
            superpixelDistance = std::numeric_limits<double>::infinity();
        }
        if(superpixelDistance < bestBaseSuperpixelDistance)
        {
            bestBaseSuperpixelDistance = superpixelDistance;
            bestBaseSuperpixelIndex = j;
        }
    }
    return bestBaseSuperpixelDistance;
#endif // FEATURE_SUPERPIXEL_SEMANTIC
}

double PlaceRecognition::computeConditionalProb(FEATURE_DISTANCE_T distance)
{
#ifdef FEATURE_COLOR_HIST
    // Gaussian conditional probability distribution
    double sigmaSquared = pow(1.0/3.0, 2.0);
    return 1.0/sqrt(2.0*PI*sigmaSquared)*exp(-pow(distance, 2.0)/(2.0*sigmaSquared));
#endif // FEATURE_COLOR_HIST

#ifdef FEATURE_SUPERPIXEL_SEMANTIC
    // Gaussian conditional probability distribution
    //printf("distance = %lf\n",distance);
    double sigmaSquared = pow(25.0, 2.0);
    return 1.0/sqrt(2.0*PI*sigmaSquared)*exp(-pow(distance, 2.0)/(2.0*sigmaSquared));
#endif // FEATURE_SUPERPIXEL_SEMANTIC
}

void PlaceRecognition::ShowManyImages(cv::String title, int nArgs, ...)
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
