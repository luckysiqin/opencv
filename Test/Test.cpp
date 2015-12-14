  
#include "StdAfx.h"
#include "cv.h"  
#include "highgui.h"  
#include <opencv2/opencv.hpp>   
#include <cstdio>   
#include <cstdlib>   
#include "opencv2/core/core.hpp"  
#include "opencv2/contrib/contrib.hpp"  
#include "opencv2/highgui/highgui.hpp"  
  
#include <iostream>  
#include <fstream>  
#include <sstream>  
  
using namespace cv;  
using namespace std;  
  
static Mat norm_0_255(InputArray _src) {  
   Mat src = _src.getMat();  
    // Create and return normalized image:  
    Mat dst;  
    switch(src.channels()) {  
    case 1:  
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);  
        break;  
    case 3:  
       cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);  
       break;  
    default:  
       src.copyTo(dst);  
        break;  
    }  
    return dst;  
}  
  
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {  
    std::ifstream file(filename.c_str(), ifstream::in);  
    if (!file) {  
        string error_message = "No valid input file was given, please check the given filename.";  
        CV_Error(CV_StsBadArg, error_message);  
    }  
    string line, path, classlabel;  
    while (getline(file, line)) {  
        stringstream liness(line);  
       getline(liness, path, separator);  
        getline(liness, classlabel);  
       if(!path.empty() && !classlabel.empty()) {  
            images.push_back(imread(path, 0));  
           labels.push_back(atoi(classlabel.c_str()));  
       }  
    }  
}  
  
int main(int argc, const char *argv[]) {  
    // Check for valid command line arguments, print usage  
   // if no arguments were given.  
    if (argc < 2) {  
        cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;  
        exit(1);  
    }  
    string output_folder;  
    if (argc == 3) {  
        output_folder = string(argv[2]);  
   }  
    // Get the path to your CSV.  
    string fn_csv = string(argv[1]);  
    // These vectors hold the images and corresponding labels.  
    vector<Mat> images;  
		vector<int> labels;  
    // Read in the data. This can fail if no valid  
   // input filename is given.  
    try {  
        read_csv(fn_csv, images, labels);  
    } catch (cv::Exception& e) {  
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;  
        // nothing more we can do  
		exit(1);  
   }  
    // Quit if there are not enough images for this demo.  
    if(images.size() <= 1) {  
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";  
        CV_Error(CV_StsError, error_message);  
    }  
¡¢
    int height = images[0].rows;  
   
    Mat testSample = images[images.size() - 1];  
    int testLabel = labels[labels.size() - 1];  
    images.pop_back();  
    labels.pop_back();  

    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();  
    model->train(images, labels);  

    int predictedLabel = model->predict(testSample);  

    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);  
    cout << result_message << endl;  

    Mat eigenvalues = model->getMat("eigenvalues");  
    Mat W = model->getMat("eigenvectors");  
  
    Mat mean = model->getMat("mean");  

    if(argc == 2) {  
       imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));  
    } else {  
        imwrite(format("%s/AA.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));  
    }  
 
    for (int i = 0; i < min(16, W.cols); i++) {  
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));  
        cout << msg << endl;  

        Mat ev = W.col(i).clone();  
 
        Mat grayscale = norm_0_255(ev.reshape(1, height));  
        // Show the image & apply a Bone colormap for better sensing.  
        Mat cgrayscale;  
        applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);  
 
        if(argc == 2) {  
            imshow(format("fisherface_%d", i), cgrayscale);  
        } else {  
            imwrite(format("%s/.png", output_folder.c_str(), i), norm_0_255(cgrayscale));  
        }  
    }  

    for(int num_component = 0; num_component < min(16, W.cols); num_component++) {  

        Mat ev = W.col(num_component);  
       Mat projection = subspaceProject(ev, mean, images[0].reshape(1,1));  
        Mat reconstruction = subspaceReconstruct(ev, mean, projection);  

        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));  

       if(argc == 2) {  
            imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);  
        } else {  
            imwrite(format("%s/CC.png", output_folder.c_str(), num_component), reconstruction);  
        }  
    }  

    if(argc == 2) {  
        waitKey(0);  
    }  
    return 0;  
}  
