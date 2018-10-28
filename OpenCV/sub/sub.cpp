#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
int main(int argc, char **argv)
{
	if ((argc < 3) || (argc > 3)) {
		printf("Application expected argumets are %s <image1> <image2> \n", argv[0]);
		return 0;
	}
	Mat bg_frame = imread(argv[1]);
	Mat cam_frame = imread(argv[2]);

	Mat motion;
	
	Mat bg_frame_grey;
	Mat cam_frame_grey;
	cvtColor(bg_frame, bg_frame_grey, CV_BGR2GRAY);
	cvtColor(cam_frame, cam_frame_grey, CV_BGR2GRAY);

	cv::absdiff(bg_frame_grey, cam_frame_grey, motion);
	cv::threshold(motion, motion, 80, 255, cv::THRESH_BINARY);
	cv::erode(motion, motion, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

//	imshow("Difference of image", motion);

	int sum = 0;

	for(int j=0;j<motion.rows;j++) 
	{
		for (int i=0;i<motion.cols;i++)
		{
			sum += motion.at<uchar>(i,j);
		}
	}

	if (sum) {
		printf("Difference observed\n");
		imwrite( "/home/pi/Desktop/motion.jpg",  motion);
	}else {
		printf("Image are same\n");
	}

	return 0;
}
