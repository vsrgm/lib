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

uint8_t *buffer;
#define CAM_WIDTH	1280 
#define CAM_HEIGHT 	 720

#define CLEAR(x) memset(&(x), 0, sizeof(x))

#define EXP_MANUAL_VAL 	325

unsigned int sizeimage;

typedef struct {
        void   *start;
        size_t  length;
} buffers;

buffers *g_buffers = NULL;
int n_buffers = 0;

struct image {
	struct v4l2_queryctrl controls;
	struct v4l2_querymenu menu;
	struct v4l2_control usr_ctrl;
	struct v4l2_capability cap;
	struct v4l2_fmtdesc desc;
	struct v4l2_format frmt;
	struct v4l2_frmsizeenum frames;
	struct v4l2_frmivalenum frm_int;
	struct v4l2_requestbuffers bufreq;
	struct v4l2_buffer bufinfo;
};

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
	struct v4l2_cropcap cropcap;
	struct v4l2_format fmt;
	struct v4l2_fmtdesc fmtdesc = {0};
	int support_grbg10 = 0, min = 0;
	char fourcc[5] = {0};
	char c, e;
	
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
	
	fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
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

	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width = CAM_WIDTH;
	fmt.fmt.pix.height = CAM_HEIGHT;
	fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_UYVY;//V4L2_PIX_FMT_MJPEG;
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
	
	/* Buggy driver paranoia. */
	min = fmt.fmt.pix.width * 2;
	if (fmt.fmt.pix.bytesperline < min) {
		fmt.fmt.pix.bytesperline = min;
	}
	min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
	if (fmt.fmt.pix.sizeimage < min)
			fmt.fmt.pix.sizeimage = min;
	
	sizeimage = fmt.fmt.pix.sizeimage;
	return 0;
}

int init_userp(int fd, unsigned int buffer_size)
{
	struct v4l2_requestbuffers req = {0};

	CLEAR(req);

	req.count  = 3;
	req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_USERPTR;
	
	if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
		perror("Requesting Buffer");
		return 1;
	}
	g_buffers = (buffers*) calloc(3, sizeof (*g_buffers));

	if (!g_buffers) {
			perror("Out of memory\n");
			return 1;
	}
	for (n_buffers = 0; n_buffers < 3; ++n_buffers) {
		g_buffers[n_buffers].length = buffer_size;
		//g_buffers[n_buffers].start = malloc(buffer_size);
		if(posix_memalign(&g_buffers[n_buffers].start, getpagesize(), buffer_size) != 0) {
			printf("\n %d \n", __LINE__);
			return 1;
		}

		if (!g_buffers[n_buffers].start) {
			perror("Out of memory\n");
			return 1;
		}
	}
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

int convert_uyvy_rgb888(unsigned char* uyvy_buffer,unsigned int width,unsigned int height, unsigned char* rgb888)
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
			u_val = (float)uyvy_buffer[yuvcount++];				
			y1_val = (float)uyvy_buffer[yuvcount++];
			v_val = (float)uyvy_buffer[yuvcount++];
			y2_val = (float)uyvy_buffer[yuvcount++];

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

int capture_image(int fd, int count, int init)
{
	struct v4l2_buffer buf = {0};
	int i = 0;
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	//buf.memory = V4L2_MEMORY_MMAP;
	buf.memory = V4L2_MEMORY_USERPTR;
	
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
	
	//while(buf.flags & V4L2_BUF_FLAG_ERROR){
	if(init == 0) {
		for(i = 0; i < 4; ++i) {
			if(-1 == xioctl(fd, VIDIOC_DQBUF, &buf))
			{
				perror("Retrieving Frame");
				return 1;
			}
			if(-1 == xioctl(fd, VIDIOC_QBUF, &buf))
			{
				perror("Query Buffer");
				return 1;
			}
		}
	}
	
	if(-1 == xioctl(fd, VIDIOC_DQBUF, &buf))
	{
		perror("Retrieving Frame");
		return 1;
	}
	
	if (count < 99) {
		printf("\n %d buf.timestamp sec = %ld.%ld \n", count, buf.timestamp.tv_sec%100, buf.timestamp.tv_usec/1000); 
		if(-1 == xioctl(fd, VIDIOC_QBUF, &buf))
		{
			perror("Query Buffer");
			return 1;
		}
		return 0;
	}
	
	for (i = 0; i < n_buffers; ++i) {
		if (buf.m.userptr == (unsigned long)g_buffers[i].start
			&& buf.length == g_buffers[i].length)
				break;
	}
	
	assert (i < n_buffers);
	printf ("saving image\n");
	IplImage *frame, *grayImg;
	IplImage* m_RGB = cvCreateImage(cvSize(CAM_WIDTH, CAM_HEIGHT), IPL_DEPTH_8U, 3);

	//    CvMat cvmat = cvMat(CAM_HEIGHT, CAM_WIDTH, CV_8UC3, (void*)buffer);
	//    frame = cvDecodeImage(&cvmat, 1);
	
	convert_uyvy_rgb888((unsigned char*)buf.m.userptr,CAM_WIDTH, CAM_HEIGHT,(unsigned char*)m_RGB->imageData);
	//convert_y16_rgb888((unsigned short*)buffer,CAM_WIDTH, CAM_HEIGHT,(unsigned char*)m_RGB->imageData);
	grayImg = cvCreateImage( cvSize(CAM_WIDTH,CAM_HEIGHT), IPL_DEPTH_8U, 1 );
	//    cvSaveImage("image.jpg", m_RGB, 0);
	cvCvtColor(m_RGB , grayImg, CV_BGR2GRAY );
	//    cvCanny(grayImg, grayImg, 50, 150, 3);   
	cvSaveImage("image.jpg", grayImg, 0);
	cvReleaseImage(&grayImg);
	cvReleaseImage(&m_RGB);

	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_USERPTR;
	//buf.index = 0;
	if(-1 == xioctl(fd, VIDIOC_QBUF, &buf))
	{
		perror("Query Buffer");
		return 1;
	}
	return 0;
}

int usercontrol(unsigned int cam_fd,int no_ctrl,struct image *img)
{
	int err = 0;
	
	img->usr_ctrl.id = V4L2_CID_EXPOSURE_AUTO;	// Exposure Mode - V4L2_CID_EXPOSURE_AUTO
	img->usr_ctrl.value = V4L2_EXPOSURE_MANUAL;		// Manual Exposure = 1
	err = ioctl(cam_fd,VIDIOC_S_CTRL,&img->usr_ctrl);
	if(err) {								//Testing if the given value is in the range
		printf("%d\n",errno);
		perror("VIDIOC_S_CTRL");
	}
	
	img->usr_ctrl.id = V4L2_CID_EXPOSURE_ABSOLUTE;	// Manual Mode
	img->usr_ctrl.value = EXP_MANUAL_VAL;		// Manual exp value
	err = ioctl(cam_fd,VIDIOC_S_CTRL,&img->usr_ctrl);
	if(err) {								//Testing if the given value is in the range
		printf("%d\n",errno);
		perror("VIDIOC_S_CTRL");
	}
	printf("\n %d \n", __LINE__);
	return 0;
}

int main()
{
	int fd;
	int i;
	enum v4l2_buf_type type;
	int ctrls = 0;
	struct image img;
	
	fd = open("/dev/video0", O_RDWR);
	if (fd == -1)
	{
		perror("Opening video device");
		return 1;
	}
	if(print_caps(fd))
		return 1;

	if(init_userp(fd, sizeimage)) {
		return 1;
	}
	
	usercontrol(fd, ctrls, &img);

	for (i = 0; i < n_buffers; ++i) {
		struct v4l2_buffer buf;

		CLEAR(buf);
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_USERPTR;
		buf.index = i;
		buf.m.userptr = (unsigned long)g_buffers[i].start;
		buf.length = g_buffers[i].length;
		if (-1 == xioctl(fd, VIDIOC_QBUF, &buf)) {
			perror("Query Buffer");
			return 1;
		}
	}
	
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (-1 == xioctl(fd, VIDIOC_STREAMON, &type)) {
		perror("Start Capture");
		return 1;
	}
	int init = 0;
	for(i = 0; i < 100; i++)
	{
		if(capture_image(fd, i, init))
			return 1;
		init++;
	}
	
	if (-1 == xioctl(fd, VIDIOC_STREAMOFF, &type)) {
		perror("Stop Capture");
		return 1;
	}
	
	for (i = 0; i < n_buffers; ++i)
		free(g_buffers[i].start);
	
	free(g_buffers);
	close(fd);
	return 0;
}

