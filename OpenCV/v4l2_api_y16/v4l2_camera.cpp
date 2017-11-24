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
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

uint8_t *buffer;
#define CAM_WIDTH	1280 
#define CAM_HEIGHT 	 720
static int xioctl(int fd, int request, void *arg)
{
	int r;

	do r = ioctl (fd, request, arg);
	while (-1 == r && EINTR == errno);

	return r;
}

int print_caps(int fd)
{
	struct v4l2_capability caps = {};
	if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &caps))
	{
		perror("Querying Capabilities");
		return 1;
	}

	printf( "Driver Caps:\n"
			"  Driver: \"%s\"\n"
			"  Card: \"%s\"\n"
			"  Bus: \"%s\"\n"
			"  Version: %d.%d\n"
			"  Capabilities: %08x\n",
			caps.driver,
			caps.card,
			caps.bus_info,
			(caps.version>>16)&&0xff,
			(caps.version>>24)&&0xff,
			caps.capabilities);


	struct v4l2_cropcap cropcap;
	cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl (fd, VIDIOC_CROPCAP, &cropcap))
	{
		perror("Querying Cropping Capabilities");
		return 1;
	}

	printf( "Camera Cropping:\n"
			"  Bounds: %dx%d+%d+%d\n"
			"  Default: %dx%d+%d+%d\n"
			"  Aspect: %d/%d\n",
			cropcap.bounds.width, cropcap.bounds.height, cropcap.bounds.left, cropcap.bounds.top,
			cropcap.defrect.width, cropcap.defrect.height, cropcap.defrect.left, cropcap.defrect.top,
			cropcap.pixelaspect.numerator, cropcap.pixelaspect.denominator);

	int support_grbg10 = 0;

	struct v4l2_fmtdesc fmtdesc = {0};
	fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	char fourcc[5] = {0};
	char c, e;
	printf("  FMT : CE Desc\n--------------------\n");
	while (0 == xioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc))
	{
		strncpy(fourcc, (char *)&fmtdesc.pixelformat, 4);
		if (fmtdesc.pixelformat == V4L2_PIX_FMT_SGRBG10)
			support_grbg10 = 1;
		c = fmtdesc.flags & 1? 'C' : ' ';
		e = fmtdesc.flags & 2? 'E' : ' ';
		printf("  %s: %c%c %s\n", fourcc, c, e, fmtdesc.description);
		fmtdesc.index++;
	}
	/*
	   if (!support_grbg10)
	   {
	   printf("Doesn't support GRBG10.\n");
	   return 1;
	   }*/

	struct v4l2_format fmt;
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width = CAM_WIDTH;
	fmt.fmt.pix.height = CAM_HEIGHT;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_Y16;//V4L2_PIX_FMT_MJPEG;
	fmt.fmt.pix.field = V4L2_FIELD_NONE;

	if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
	{
		perror("Setting Pixel Format");
		return 1;
	}

	strncpy(fourcc, (char *)&fmt.fmt.pix.pixelformat, 4);
	printf( "Selected Camera Mode:\n"
			"  Width: %d\n"
			"  Height: %d\n"
			"  PixFmt: %s\n"
			"  Field: %d\n",
			fmt.fmt.pix.width,
			fmt.fmt.pix.height,
			fourcc,
			fmt.fmt.pix.field);
	return 0;
}

int init_mmap(int fd)
{
	struct v4l2_requestbuffers req = {0};
	req.count = 1;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req))
	{
		perror("Requesting Buffer");
		return 1;
	}

	struct v4l2_buffer buf = {0};
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = 0;
	if(-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
	{
		perror("Querying Buffer");
		return 1;
	}

	buffer = (uint8_t*)mmap (NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
	printf("Length: %d\nAddress: %p\n", buf.length, buffer);
	printf("Image Length: %d\n", buf.bytesused);

	return 0;
}

int convert_yuyv_rgb888(unsigned char* yuyv_buffer,unsigned int width,unsigned int height, unsigned char* rgb888)
{
	int i = 0, j = 0;
	unsigned char* rgb_buffer;
	int temp;
	int yuvcount = 0;
	int rgbcount = 0;
	float u_val, v_val, y1_val, y2_val;
	unsigned int time;

	rgb_buffer = rgb888;

	// memset(rgb_buffer, 0x00, (still_height * still_width * 3));
	// R = Y + 1.403V'
	// G = Y - 0.344U' - 0.714V'	
	// B = Y + 1.770U'

	for (i=0; i < height; i++)
	{
		for (j = 0; j < width; j+=2)
		{
			y1_val = (float)yuyv_buffer[yuvcount++];
			u_val = (float)yuyv_buffer[yuvcount++];				
			y2_val = (float)yuyv_buffer[yuvcount++];
			v_val = (float)yuyv_buffer[yuvcount++];

			u_val = u_val - 128;
			v_val = v_val - 128;		

			temp = (int)(y1_val + (1.770 * u_val));
			rgb_buffer[(((height-1)-i) * width * 3) + j*3 +0] = (temp > 255) ? 255 : ((temp < 0) ? 0 :(unsigned char)temp);

			temp = (int)(y1_val - (0.344 * u_val) - (0.714 * v_val));
			rgb_buffer[(((height-1)-i) * width * 3) + j*3 +1] = (temp > 255) ? 255 : ((temp < 0) ? 0 :(unsigned char)temp);

			temp = (int)(y1_val + (1.403 * v_val));
			rgb_buffer[(((height-1)-i) * width * 3) + j*3 +2] = (temp > 255) ? 255 : ((temp < 0) ? 0 :(unsigned char)temp);

			temp = (int)(y2_val + (1.770 * u_val));
			rgb_buffer[(((height-1)-i) * width * 3) + j*3 +3] = (temp > 255) ? 255 : ((temp < 0) ? 0 :(unsigned char)temp);

			temp = (int)(y2_val - (0.344 * u_val) - (0.714 * v_val));
			rgb_buffer[(((height-1)-i) * width * 3) + j*3 +4] = (temp > 255) ? 255 : ((temp < 0) ? 0 :(unsigned char)temp);

			temp = (int)(y2_val + (1.403 * v_val));
			rgb_buffer[(((height-1)-i) * width * 3) + j*3 +5] = (temp > 255) ? 255 : ((temp < 0) ? 0 :(unsigned char)temp);
		}
	}

	return 0;
}

int convert_y16_rgb888(unsigned short* yuyv_buffer,unsigned int width,unsigned int height, unsigned char* rgb888)
{
	int i = 0, j = 0;
	unsigned char* rgb_buffer;
	int temp;
	int yuvcount = 0;
	int rgbcount = 0;
	int y16;
	unsigned int time;

	rgb_buffer = rgb888;

	// memset(rgb_buffer, 0x00, (still_height * still_width * 3));
	// R = Y + 1.403V'
	// G = Y - 0.344U' - 0.714V'	
	// B = Y + 1.770U'

	for (i=0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			y16 = (float)yuyv_buffer[yuvcount++];
			temp = y16 >>2;
			rgb_buffer[(((height-1)-i) * width * 3) + j*3 +0] = (temp > 255) ? 255 : ((temp < 0) ? 0 :(unsigned char)temp);
			rgb_buffer[(((height-1)-i) * width * 3) + j*3 +1] = (temp > 255) ? 255 : ((temp < 0) ? 0 :(unsigned char)temp);				
			rgb_buffer[(((height-1)-i) * width * 3) + j*3 +2] = (temp > 255) ? 255 : ((temp < 0) ? 0 :(unsigned char)temp);
		}
	}

	return 0;
}

int capture_image(int fd)
{
	struct v4l2_buffer buf = {0};
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = 0;

	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(fd, &fds);
	struct timeval tv = {0};
	tv.tv_sec = 2;
	int r = select(fd+1, &fds, NULL, NULL, &tv);
	if(-1 == r)
	{
		perror("Waiting for Frame");
		return 1;
	}

	if(-1 == xioctl(fd, VIDIOC_DQBUF, &buf))
	{
		perror("Retrieving Frame");
		return 1;
	}
	printf ("saving image\n");
	IplImage *frame, *grayImg;
	IplImage* m_RGB = cvCreateImage(cvSize(CAM_WIDTH, CAM_HEIGHT), IPL_DEPTH_8U, 3);

	//    CvMat cvmat = cvMat(CAM_HEIGHT, CAM_WIDTH, CV_8UC3, (void*)buffer);
	//    frame = cvDecodeImage(&cvmat, 1);
	//    convert_yuyv_rgb888((unsigned char*)buffer,CAM_WIDTH, CAM_HEIGHT,(unsigned char*)m_RGB->imageData);
	convert_y16_rgb888((unsigned short*)buffer,CAM_WIDTH, CAM_HEIGHT,(unsigned char*)m_RGB->imageData);
	grayImg = cvCreateImage( cvSize(CAM_WIDTH,CAM_HEIGHT), IPL_DEPTH_8U, 1 );
	//    cvSaveImage("image.jpg", m_RGB, 0);
	cvCvtColor(m_RGB , grayImg, CV_BGR2GRAY );
	//    cvCanny(grayImg, grayImg, 50, 150, 3);   
	cvSaveImage("image.jpg", grayImg, 0);
	cvReleaseImage(&grayImg);
	cvReleaseImage(&m_RGB);

	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = 0;
	if(-1 == xioctl(fd, VIDIOC_QBUF, &buf))
	{
		perror("Query Buffer");
		return 1;
	}

	return 0;
}

int main()
{
	int fd;

	fd = open("/dev/video0", O_RDWR);
	if (fd == -1)
	{
		perror("Opening video device");
		return 1;
	}
	if(print_caps(fd))
		return 1;

	if(init_mmap(fd))
		return 1;
	int i;

	struct v4l2_buffer buf = {0};
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;
	buf.index = 0;
	if(-1 == xioctl(fd, VIDIOC_QBUF, &buf))
	{
		perror("Query Buffer");
		return 1;
	}

	if(-1 == xioctl(fd, VIDIOC_STREAMON, &buf.type))
	{
		perror("Start Capture");
		return 1;
	}

	for(i=0; i<5; i++)
	{
		if(capture_image(fd))
			return 1;
	}
	close(fd);
	return 0;
}

