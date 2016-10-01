/*
* Copyright (c) 2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
* Released to public domain under terms of the BSD Simplified license.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
*   * Neither the name of the organization nor the names of its contributors
*     may be used to endorse or promote products derived from this software
*     without specific prior written permission.
*
*   See <http://www.opensource.org/licenses/bsd-license>
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;
using namespace ml;

string face_cascade_name = "C:/OpenCV/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml";
CascadeClassifier face_cascade;
bool detected = false;

// Reads the images and labels from a given CSV file, a valid file would
// look like this:
//
//      /path/to/person0/image0.jpg;0
//      /path/to/person0/image1.jpg;0
//      /path/to/person1/image0.jpg;1
//      /path/to/person1/image1.jpg;1
//      ...
//
void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels) {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file)
		throw std::exception();
	std::string line, path, classlabel;
	// For each line in the given file:
	while (std::getline(file, line)) {
		// Get the current line:
		std::stringstream liness(line);
		// Split it at the semicolon:
		std::getline(liness, path, ';');
		std::getline(liness, classlabel);
		// And push back the data into the result vectors:
		images.push_back(imread(path, IMREAD_GRAYSCALE));
		labels.push_back(atoi(classlabel.c_str()));
	}
}

// Normalizes a given image into a value range between 0 and 255.
Mat norm_0_255(const Mat& src) {
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
	// Number of samples:
	size_t n = src.size();
	// Return empty matrix if no matrices given:
	if (n == 0)
		return Mat();
	// dimensionality of (reshaped) samples
	size_t d = src[0].total();
	// Create resulting data matrix:
	Mat data(n, d, rtype);
	// Now copy data:
	for (int i = 0; i < n; i++) {
		//
		if (src[i].empty()) {
			string error_message = format("Image number %d was empty, please check your input data.", i);
			CV_Error(CV_StsBadArg, error_message);
		}
		// Make sure data can be reshaped, throw a meaningful exception if not!
		if (src[i].total() != d) {
			string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
			CV_Error(CV_StsBadArg, error_message);
		}
		// Get a hold of the current row:
		Mat xi = data.row(i);
		
		// Make reshape happy by cloning for non-continuous matrices:
		if (src[i].isContinuous()) {
			src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		}
		else {
			src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
		}
	}
	return data;
}

void TrainSVM(Ptr<SVM>& svm, Mat trainData, Mat labels, PCA pca) {
	svm = SVM::create();
	// edit: the params struct got removed,
	// we use setter/getter now:
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setGamma(3);

	//	Mat trainData; // one row per feature
	svm->train(trainData, ROW_SAMPLE, labels);
	
}

bool fexists(const char *filename)
{
	ifstream ifile(filename);
	return (bool)ifile;
}

void save(const string &file_name, Mat mean, Mat vectors, Mat values)
{
	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "mean" << mean;
	fs << "e_vectors" << vectors;
	fs << "e_values" << values;
	fs.release();
}

PCA load(const string &file_name, cv::PCA pca_)
{
	FileStorage fs(file_name, FileStorage::READ);
	fs["mean"] >> pca_.mean;
	fs["e_vectors"] >> pca_.eigenvectors;
	fs["e_values"] >> pca_.eigenvalues;
	fs.release();
	return pca_;
}

Mat detectFace(Mat im, Mat frame, Mat& cropped)
{
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(im, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	Rect largest;
	for (int i = 0; i < faces.size(); i++)
	{
		if (faces[i].area() > largest.area())
			largest = faces[i];
		//Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		//ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}
	if (faces.size() == 0)
		return frame;

	Mat crop = frame(largest);
	Size s(200, 200);
	resize(crop, crop, s);
	cropped = crop;

	Point center(largest.x + largest.width*0.5, largest.y + largest.height*0.5);
	ellipse(frame, center, Size(largest.width*0.5, largest.height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

	
	//Point pt1(largest.x, largest.y); // Display detected faces on main window - live stream from camera
	//Point pt2((largest.x + largest.height), (largest.y + largest.width));
	//rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);

	return frame;
}
string getEmotion(int ans) {
	switch (ans) {
	case 0:return "neutral";
	case 1:return "anger";
	case 2:return "disgust";
	case 3:return "fear";
	case 4:return "happy";
	case 5:return "sad";
	case 6:return "surprise";
	default:return"nothing";
	}
}
void recognizeEmotion(Mat f, PCA& pca, int num_components, Ptr<SVM>& svm,vector<Mat> db) {
	Mat re(1, num_components, CV_32FC1);
	
	vector<Mat> temp;
	temp.push_back(f);
	Mat frameMatrix = asRowMatrix(temp, CV_32FC1);
	for (int i = 0; i < frameMatrix.rows; i++)
	{
		
		Mat frame = frameMatrix.row(i);
		if (frame.empty())
			cout << "EMPTY--------------------" << endl;
		//Size s(200, 200);
		//resize(frame, frame, s);
		//frame = frame.reshape(1, 1);
		frame.convertTo(frame, CV_32FC1);
		//cout << "input frame length: " << frame.rows << "--" << frame.cols << endl;
		pca.project(frame, re);
		//imshow("avg", re.reshape(1, db[0].));
		//imshow("pc1", norm_0_255(pca.eigenvectors.row(pca.eigenvectors.rows-1)).reshape(1, db[0].rows));
		int ans = svm->predict(re);
		cout << i << " This photo is person " << getEmotion(ans) << endl;
		
	}
	
}
void generatePCA(PCA&pca, vector<Mat>db,int num_components,Mat& data, Mat& eigenStuff) {
	for (int i = 0; i < db.size(); i++) {
		Mat projectedMat(1, num_components, CV_32FC1);
		//cout << "image(" << i << ")" << data.row(i).cols << endl;
		pca.project(data.row(i), projectedMat);
		projectedMat.row(0).copyTo(eigenStuff.row(i));
	}

	// And copy the PCA results:
	Mat mean = pca.mean.clone();
	Mat eigenvalues = pca.eigenvalues.clone();
	Mat eigenvectors = pca.eigenvectors.clone();
	save("pca.xml", mean, eigenvalues, eigenvectors);
}
int main(int argc, const char *argv[]) {
	if (!face_cascade.load(face_cascade_name)) {
		printf("--(!)Error loading\n");
		return (-1);
	}
	// Holds some images:
	vector<Mat> db;

	vector<int> labels;
	read_csv("faces.csv", db, labels);

	// Build a matrix with the observations in row:
	Mat data = asRowMatrix(db, CV_32FC1);
	cout << "data col" << data.cols << endl;
	// Number of components to keep for the PCA:
	int num_components = 20;

	// Perform a PCA:
	PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, num_components);
	Ptr<SVM> svm;
	//svm->setType(SVM::C_SVC);
	//svm->setKernel(SVM::LINEAR);
	//cout << "----SVM:"<<svm->SVM::getKernelType() << endl;
	Mat eigenStuff(data.rows, num_components, CV_32FC1); //This Mat will contain all the Eigenfaces that will be used later with SVM for detection
	//cout << "rows" << data.rows << "eigenCol" << eigenStuff.cols << endl;
	generatePCA(pca, db, num_components, data, eigenStuff);
	
	//pca = load("pca.xml", pca);
	// The mean face:
	//imshow("avg", norm_0_255(mean.reshape(1, db[0].rows)));
	//cout << "number: "<< eigenvalues.rows<<endl;
	//cout << "numberLabels: " << labels.size()<< endl;
	//cout << "numberColumns: " << eigenvalues.cols << endl;

	//create svm
	Mat labelsMatrix(labels, true);
	
	//cout << "matrix row "<<labelsMatrix.rows << endl;
	TrainSVM(svm, eigenStuff, labelsMatrix, pca);
	svm->save("emotionRecognition.xml");

	VideoCapture cap(0);
	if (!cap.isOpened()) {
		return -1;
	}
	clock_t start = clock(); cout << clock()/CLOCKS_PER_SEC<< endl;
	//Mat imgs = asRowMatrix(db, CV_32FC1);
	while (waitKey(15) != 'q') {
		Mat frame;
		
		cap >> frame;

		if (!frame.empty()) {
			cvtColor(frame, frame, CV_BGR2GRAY);
			equalizeHist(frame, frame);
			//frame = frame.reshape(1, 1);
			//imgs.push_back(inp);
			//frame.convertTo(frame, CV_32F);
			//Mat face = detectFace(frame);
			//for (int i = 110; i < 115; i++)
			//{
			//Mat temp = asRowMatrix(imgs, CV_32F);
			//cout << temp.rows << "--tempRows--" << pca.eigenvalues.rows;
			//Mat frame = imread("S010_001_00000001.png");'
			Mat cropped;
			//cout << "cropsize" << cropped.size() << endl;
			frame = detectFace(frame, frame, cropped);
			
			if (!cropped.empty()) {
				imshow("cr", cropped);
				recognizeEmotion(cropped, pca, num_components, svm, db);
			}
			
			//int deltaTime = (clock() / CLOCKS_PER_SEC - start/ CLOCKS_PER_SEC) ;
			//if (deltaTime%3==0  ) {
				//detected = true;
				//cout << deltaTime << endl;
				//if (!cropped.empty())
				//	recognizeEmotion(cropped, pca, num_components, svm, db);
				//else
				//	cout << "no face" << endl;
			//}

			

			imshow("final", frame);
		}
		


	}
	//cout <<re.cols << "and" << endl;
	
	

	// Show the images:
	waitKey(0);

	// Success!
	return 0;
}





