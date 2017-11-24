/** * @file camera.c
 * @brief camera capture apis
 * @version 1.0
 * @date 2015-09-24
 */

#include "cam1_mipi.h"
#define DEBUG

#if defined (USE_CUDA_BASE)
	#include <cuda.h>
	#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
	#include <helper_functions.h>
	#include <helper_cuda.h>
#elif defined (USE_ARM_NEON)
	#include <arm_neon.h>
#endif

struct buf_info capture_buff_info_mipi1[CAPTURE_MAX_BUFFER_MIPI1];

/**
 * @brief : open camera device node
 *
 * @param camera_node: camera device node name
 *
 * @return : return camer file descriptor number 
 */
static int open_camera_node(char *camera_node)
{
	int fd;
	if((fd = open(camera_node, O_RDWR)) < 0) {
		fprintf(stderr, "cannot open the camera node\n");
	}
	return fd;
}

/**
 * @brief: wrapper for start stredaming
 *
 * @param[in]: fd, file descriptor of the v4l2 device 
 * @param[in]: buf_type, type of buffer mmap or user pointer
 *
 * @return: on success zero will be returned
 */
static int start_streaming(int fd, int buf_type) 
{	
#ifdef __PRINT_FUNC_TIME_BUDGET__
	struct timeval t0 = {0}, t1 = {0};
	long long func_time = 0;
	gettimeofday(&t0, NULL);
#endif
	if(ioctl(fd, VIDIOC_STREAMON, &buf_type) < 0) {
		perror("VIDIOC_STREAMON");
		return -1;
	}

#ifdef __PRINT_FUNC_TIME_BUDGET__
	gettimeofday(&t1, NULL);
	func_time = (t1.tv_sec - t0.tv_sec)*1000000LL + t1.tv_usec - t0.tv_usec;
	printf("%s time = %lld\n", __func__, func_time);
#endif
	return 0;
}

#if 0
/**
 * @brief: wrapper for stop streaming
 *
 * @param fd: file descriptor for the v4l2 devcie
 * @param buf_type: type of buffer mmap or user pointer
 *
 * @return: on success zero will be returned
 */
static int stop_streaming(int fd, int buf_type)
{
#ifdef __PRINT_FUNC_TIME_BUDGET__
	struct timeval t0 = {0}, t1 = {0};
	long long func_time = 0;
	gettimeofday(&t0, NULL);
#endif

	if(ioctl(fd, VIDIOC_STREAMOFF, &buf_type) < 0) {
		perror("VIDIOC_STREAMOFF");
		return -1;
	}

#ifdef __PRINT_FUNC_TIME_BUDGET__
	gettimeofday(&t1, NULL);
	func_time = (t1.tv_sec - t0.tv_sec)*1000000LL + t1.tv_usec - t0.tv_usec;
	printf("%s time = %lld\n", __func__, func_time);
#endif
	return 0;
}
#endif
/**
 * @brief: check the capability of the v4l2 device 
 *
 * @param fd: file descriptor of the v4l2_device node  
 * @param specific_capability: capability to check for
 *
 * @return: if the capability is there then 1 will be returned else zero will be returned
 */
static int check_capability(int fd, int specific_capability)
{
	struct v4l2_capability capability;
#ifdef __PRINT_FUNC_TIME_BUDGET__
	struct timeval t0 = {0}, t1 = {0};
	long long func_time = 0;
	gettimeofday(&t0, NULL);
#endif
	if(ioctl(fd, VIDIOC_QUERYCAP, &capability) < 0) {
		perror("VIDIOC_QUERYCAP");
		exit(errno);
	}
	if(capability.capabilities & specific_capability) {
		return 1;
	}
#ifdef __PRINT_FUNC_TIME_BUDGET__
	gettimeofday(&t1, NULL);
	func_time = (t1.tv_sec - t0.tv_sec)*1000000LL + t1.tv_usec - t0.tv_usec;
	printf("%s time = %lld\n", __func__, func_time);
#endif
	return 0;
}


/**
 * @brief: sethe the width height and pixem format for capture or display 
 *
 * @param fd: fd, file descriptor of the v4l2_device node
 * @param type: type, type of the device display or capture
 * @param format: format, pixel format to be set to the device
 * @param height: height, height of the image frame
 * @param width: width, width of the image frame
 *
 * @return: on succes this function will return 0 
 */
static int set_format_info(int fd, int type, int format, int height, int width)
{
	struct v4l2_format fmt = {0};
	struct v4l2_streamparm cam_parm = {0};
	struct v4l2_frmivalenum frminterval = {0};



#ifdef __PRINT_FUNC_TIME_BUDGET__
	struct timeval t0 = {0}, t1 = {0};
	long long func_time = 0;
	gettimeofday(&t0, NULL);
#endif
	fmt.type = type;
	if(ioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
		perror("ioctl error VIDIO_G_FMT");
		exit(errno);
	}

	fmt.fmt.pix.pixelformat = format;
	fmt.fmt.pix.height      = height;
	fmt.fmt.pix.width       = width;
	fmt.fmt.pix.sizeimage   = width * height * 2;

	frminterval.discrete.numerator = 1;
	frminterval.discrete.denominator = 60;
	cam_parm.type  = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	cam_parm.parm.capture.timeperframe.numerator   = frminterval.discrete.numerator;
	cam_parm.parm.capture.timeperframe.denominator = frminterval.discrete.denominator;

	printf("n:d %d:%d \n", frminterval.discrete.numerator, frminterval.discrete.denominator);
#if 0
	if(ioctl(fd, VIDIOC_S_PARM, &cam_parm) < 0) {
		perror("CAPTURE: VIDIOC_S_PARM");
	}
#endif
	if(ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
		perror("ioctl error VIDIOC_S_FMT");
		exit(errno);
	} 

	if(ioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
		perror("ioctl error VIDIOC_G_FMT");
		exit(errno);
	}

#ifdef DEBUG
	fprintf(stdout, "3) Type %d Format: %d YUYV: %d Width: %d Height: %d\n", 
			fmt.type, fmt.fmt.pix.pixelformat, format, fmt.fmt.pix.width, 
									fmt.fmt.pix.height);
#endif

	if((width != fmt.fmt.pix.width) || 
		(height != fmt.fmt.pix.height)) {

		fprintf(stdout, "device changed requested resolution "
			"(%d x %d) to (%dx%d)\n", width, height, fmt.fmt.pix.width, 
								fmt.fmt.pix.height);
	}
	if(fmt.fmt.pix.pixelformat != format) {
		fprintf(stderr, "requested format not supported\n");
		exit(errno);
	}

#ifdef __PRINT_FUNC_TIME_BUDGET__
	gettimeofday(&t1, NULL);
	func_time = (t1.tv_sec - t0.tv_sec)*1000000LL + t1.tv_usec - t0.tv_usec;
	printf("%s time = %lld\n", __func__, func_time);
#endif

	return 0;
}

/**
 * @brief: allocates and initialize the buffers for v4l2 device 
 *
 * @param fd: the opened v4l2 device node
 * @param type: type of v4l2 devcie , input or output
 * @param memory_type: the memorty type to use , eg: mmpa, usrpointer
 *
 * @return: on succes this function will return 0 
 */
static int alloc_buffs(int fd, int count, int type, int memory_type, int width, int height)
{
	struct v4l2_requestbuffers reqbuf = {0};
	struct v4l2_buffer buf = {0};
	int numbuffs = count;
	int i;

#ifdef __PRINT_FUNC_TIME_BUDGET__
	struct timeval t0 = {0}, t1 = {0};
	long long func_time = 0;
	gettimeofday(&t0, NULL);
#endif

	reqbuf.count  = count;
	reqbuf.type   = type;
	reqbuf.memory = memory_type;

	if(ioctl(fd, VIDIOC_REQBUFS, &reqbuf) < 0) {
		perror("ioctl error VIDIOC_REQBUFS");
		exit(errno);
	}
	printf("VIDIOC_REQBUFS done %d \n",reqbuf.count);
	if (reqbuf.memory == V4L2_MEMORY_USERPTR) {
		struct buf_info *buff_info = NULL;
		if(type == V4L2_BUF_TYPE_VIDEO_CAPTURE) {
			buff_info = capture_buff_info_mipi1;
		} else {
			printf("invalid buffer type\n");
        	} 

		for(i = 0; i < numbuffs; i++) {
#if 0//defined (USE_CUDA_BASE)
        		checkCudaErrors(cudaMallocHost((void **)&buff_info[i].buf.m.userptr,  width*height*2));

#else
			if (posix_memalign((void **)&buff_info[i].buf.m.userptr, getpagesize(), width*height*2) != 0) {
				perror("Memory alloc failed posix_memalign \n");
			}
#endif
			buff_info[i].buf.type = reqbuf.type;
			buff_info[i].buf.index = i;
			buff_info[i].buf.memory = reqbuf.memory;
			buff_info[i].buf.length = width*height*2;

			buff_info[i].length = buff_info[i].buf.length;
			buff_info[i].index   = i;
			buff_info[i].start   = (char *)buff_info[i].buf.m.userptr;
			buff_info[i].phy = (char *)buff_info[i].buf.m.userptr;
			printf("Buffer allocate done %d \n", i);

			if(ioctl(fd, VIDIOC_QBUF, &buff_info[i].buf) < 0) {
        	    		perror("VIDIOC_QBUF");
			}
		}
	}else if (reqbuf.memory == V4L2_MEMORY_MMAP) { 
		memset(&buf, 0, sizeof(buf));
		printf("numbuffs = %d \n", numbuffs);
		for(i = 0; i < numbuffs; i++) {
			buf.type = reqbuf.type;
			buf.index = i;
			buf.memory = reqbuf.memory;
			if(ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
				perror("VIDIOC_QUERYBUF");
				numbuffs = i;
			}

			struct buf_info *buff_info = NULL;
			if(type == V4L2_BUF_TYPE_VIDEO_CAPTURE) {
				buff_info = capture_buff_info_mipi1;
			} else {
				printf("invalid buffer type\n");
        		} 

	        	buff_info[i].length  = buf.length;
        		buff_info[i].index   = i;
        		buff_info[i].start   = (char *) mmap(NULL, buf.length, 
                						PROT_READ|
								PROT_WRITE, 
								MAP_SHARED, 
								fd, 
								buf.m.offset);
			buff_info[i].phy = (char *)buf.m.offset;
        		if(buff_info[i].start == MAP_FAILED) {
	            		fprintf(stderr, "Cannot mmap = %d buffer\n", i);
            			perror("mmap");
        	    		numbuffs = i;
        		}
	        	//memset((void *)buff_info[i].start, 0xCD, buff_info[i].length);
			if(ioctl(fd, VIDIOC_QBUF, &buf) < 0) {
            			perror("VIDIOC_QBUF");
            			numbuffs = i + 1;
			}
		}
	}

#ifdef __PRINT_FUNC_TIME_BUDGET__
	gettimeofday(&t1, NULL);
	func_time = (t1.tv_sec - t0.tv_sec)*1000000LL + t1.tv_usec - t0.tv_usec;
	printf("%s time = %lld\n", __func__, func_time);
#endif
	return 0;
}

int cam1_buf_tv_usec = -1, cam2_buf_tv_usec = -1; 
#define DELTA(x,y) ((x-y)>0?(x-y):(y-x))
#define IN_RANGE(x,y,z,round)  (DELTA(x,y)>(round-z))?								\
				((x>y)?((x>(y+round))?((x>(y+round+z))?0:1):((x>=(y+round-z))?1:0)): 		\
				(((x+round)>(y))?(((x+round)>(y+z))?0:1):(((x+round)>=(y-z))?1:0))):		\
				(x==y)?1:((x>y)?((x>(y+z))?0:1):((x>=(y-z))?1:0))


#define NEED_SKIP(x,y,z,round) (DELTA(x,y)>(round-z))?((x>y)?(x>(y+round)):((x+round)>y)):x>y
/**
 *@brief: dequeue a v4l2 buffer
 *@param[in]: fd, file descriptor of a v4l2device node
 *@param[out]: buffer, v4l2_buffer will be filled to this parameter
 *@return: on success value 0 will be returned on fail value < 0 will be returned
 */
int dequeue_buffer_mipi1(int fd, struct v4l2_buffer *buffer)
{
	if(ioctl(fd, VIDIOC_DQBUF, buffer) < 0) {
		perror("VIDIOC_DQBUF");
		fprintf(stderr, "Problem occured in deque type: %d\n", buffer->type);
		return -1;
	}
	return 0;
}

/**
 *@brief: enqueue a v4l2 buffer
 *@param[in]: fd, file descriptor of a v4l2 device node
 *@param[in]: buffer, v4l2_buffer will be queued back to v4l2 subsystem
 *@return: on success value 0 will be returned on fail value < 0 will be returned
 */
int enqueue_buffer_mipi1(int fd, struct v4l2_buffer *buffer)
{
	if(ioctl(fd, VIDIOC_QBUF, buffer)) {
		perror("VIDIOIC_QBUF");
		fprintf(stderr, "problem occured in enqueue type: %d\n", buffer->type);
		return -1;
	}
	return 0;
}


/**
 * @brief: open and initialize a capture device
 *
 * @param height: height of the image to be captured
 * @parami width: width of the image to be captured
 *
 * @return: on success this function will return 0
 */
int init_camera_mipi1(char *device_node, int width, int height, int fmt)
{

	int fd;
#ifdef __PRINT_FUNC_TIME_BUDGET__
	struct timeval t0 = {0}, t1 = {0};
	long long func_time = 0;
	gettimeofday(&t0, NULL);
#endif
	fd = open_camera_node(device_node);

	if(check_capability(fd, V4L2_CAP_STREAMING)) {
		fprintf(stdout, "Device is capable for streaming\n");
	} else {
		fprintf(stdout, "Device is not Capable for streaing\n");
		exit(errno);
	}
	int format_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    	set_format_info(fd, format_type, fmt, height, width);
	alloc_buffs(fd, CAPTURE_MAX_BUFFER_MIPI1, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP, width, height);

	start_streaming(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE);

#ifdef __PRINT_FUNC_TIME_BUDGET__
	gettimeofday(&t1, NULL);
	func_time = (t1.tv_sec - t0.tv_sec)*1000000LL + t1.tv_usec - t0.tv_usec;
	printf("%s time = %lld\n", __func__, func_time);
#endif
	return fd;
}

/**
 * @brief : get capture buffer virtual address
 *
 * @param capture_buff: v4l2 capture buffer
 *
 * @return : virtual address of the captured frame
 */
void *get_frame_virt_mipi1(struct v4l2_buffer *capture_buff)
{
	return (void *)capture_buff_info_mipi1[capture_buff->index].start;
}

/**
 * @brief : get capture buffer physical address
 *
 * @param capture_buff: v4l2 capture buffer
 *
 * @return : physical address of the captured frame
 */
void *get_frame_phy_mipi1(struct v4l2_buffer *capture_buff)
{
	return (void *)capture_buff_info_mipi1[capture_buff->index].phy;
}

