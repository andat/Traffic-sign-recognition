// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdio.h>
#include <queue>
#include <random>
#include <fstream>

#define RED_HMIN1 0
#define RED_HMAX1 6
#define RED_HMIN2 175
#define RED_HMAX2 180
#define RED_SMIN 25
#define RED_SMAX 255
#define RED_VMIN 30
#define RED_VMAX 255

#define BLUE_HMIN 95
#define BLUE_HMAX 130
#define BLUE_SMIN 70
#define BLUE_SMAX 255
#define BLUE_VMIN 56
#define BLUE_VMAX 255

#define YELLOW_HMIN 20
#define YELLOW_HMAX 30
#define YELLOW_SMIN 25
#define YELLOW_SMAX 250
#define YELLOW_VMIN 75
#define YELLOW_VMAX 250


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}


void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}



void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		imshow("source", frame);
		imshow("gray", grayFrame);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

Mat opening(Mat img) {
	Mat dest;
	int morph_size = 1;
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	morphologyEx(img, dest, MORPH_OPEN, kernel);

	return dest;
}

Mat closing(Mat img) {
	Mat dest;
	int morph_size = 1;
	Mat kernel = getStructuringElement(MORPH_CROSS, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	morphologyEx(img, dest, MORPH_CLOSE, kernel);

	return dest;
}

Mat contour(Mat src_gray, int thresh) {
	RNG rng(12345);
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using canny
	Canny(src_gray, canny_output, thresh, thresh * 2, 3);
	/// Find contours
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	return drawing;
}


void colour_segmentation() {
	char fname[MAX_PATH];
	if (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat img;
		resizeImg(src, img, 500, false);
		imshow("original image", img);
		Mat hsvImg;
		cv::cvtColor(img, hsvImg, CV_BGR2HSV);

		//filter red
		Mat red_mask1;
		cv::inRange(hsvImg, Scalar(RED_HMIN1, RED_SMIN, RED_VMIN), Scalar(RED_HMAX1, RED_SMAX, RED_VMAX), red_mask1);
		Mat red_mask2;
		cv::inRange(hsvImg, Scalar(RED_HMIN2, RED_SMIN, RED_VMIN), Scalar(RED_HMAX2, RED_SMAX, RED_VMAX), red_mask2);

		
		//saturation
		Mat red_mask = red_mask1 | red_mask2;
		Mat im_red = opening(red_mask);
		imshow("red segmentation", im_red);

		//draw contour
		RNG rng(12345);
		Mat canny_output;
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		double thresh = 100;

		/// Detect edges using canny
		Canny(im_red, canny_output, thresh * 2, 3);
		/// Find contours
		findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		/// Draw contours
		Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
		for (int i = 0; i< contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		}

		imshow("contours", drawing);



		//filter blue
		Mat blue_mask;
		cv::inRange(hsvImg, Scalar(BLUE_HMIN, BLUE_SMIN, BLUE_VMIN), Scalar(BLUE_HMAX, BLUE_SMAX, BLUE_VMAX), blue_mask);
		imshow("blue segmentation", opening(blue_mask));

		//filter yellow
		/*Mat yellow_mask;
		cv::inRange(hsvImg, Scalar(YELLOW_HMIN, YELLOW_SMIN, YELLOW_VMIN), Scalar(YELLOW_HMAX, YELLOW_SMAX, YELLOW_VMAX), yellow_mask);
		imshow("yellow segmentation", yellow_mask);*/
		waitKey(0);
	}
}


int main()
{
	//red_colour_enhancement1();
	colour_segmentation();
	return 0;
}