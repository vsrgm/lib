#include "v4l2.h"
#include "fn_prototype.h"

static int open_cam(char *dev)
{
	int fd;
	if((fd = open(dev, O_RDWR)) < 0) {
		fprintf(stderr, "cannot open the camera node\n");
	}
	return fd;
}

static int check_capability(int fd, int specific_capability)
{
	struct v4l2_capability capability;
	if(ioctl(fd, VIDIOC_QUERYCAP, &capability) < 0) {
		perror("VIDIOC_QUERYCAP");
		exit(errno);
	}
	if(capability.capabilities & specific_capability) {
		return 1;
	}
	return 0;
}

int 
get_capture_mode (int width, int height)
{
       if ((width == 640) && (height == 480)) {
               return 0;
       }else if ((width == 1280) && (height == 720)) {
               return 1;
       }else if ((width == 1920) && (height == 1080)) {
               return 2;
       }else if ((width == 3264) && (height == 2448)) {
               return 3;
       }
       return 0;
}

static int set_format_info(int fd, int type, int format, int width, int height)
{
	struct v4l2_format fmt = {0};
	struct v4l2_streamparm cam_parm = {0};
	struct v4l2_frmivalenum frminterval = {0};

	fmt.type = type;
	if(ioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
		perror("ioctl error VIDIO_G_FMT");
		return -1;
	}

	frminterval.discrete.numerator = 1;
	frminterval.discrete.denominator = 30;
	cam_parm.type  = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	cam_parm.parm.capture.timeperframe.numerator   = frminterval.discrete.numerator;
	cam_parm.parm.capture.timeperframe.denominator = frminterval.discrete.denominator;
	cam_parm.parm.capture.capturemode = get_capture_mode(width, height);

	printf("n:d %d:%d \n", frminterval.discrete.numerator, frminterval.discrete.denominator);
	if(ioctl(fd, VIDIOC_S_PARM, &cam_parm) < 0) {
		perror("CAPTURE: VIDIOC_S_PARM");
	}

	fmt.type = type;
	fmt.fmt.pix.pixelformat = format;
	fmt.fmt.pix.height = height;
	fmt.fmt.pix.width = width;
	fmt.fmt.pix.sizeimage = width * height * 2;

	if(ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
		perror("ioctl error VIDIOC_S_FMT");
		return -1;
	} 

	if(ioctl(fd, VIDIOC_G_FMT, &fmt) < 0) {
		perror("ioctl error VIDIOC_G_FMT");
		return -1;
	}

	if((width != fmt.fmt.pix.width) || 
		(height != fmt.fmt.pix.height)) {

		fprintf(stdout, "device changed requested resolution "
			"(%d x %d) to (%dx%d)\n", width, height, fmt.fmt.pix.width, 
								fmt.fmt.pix.height);
	}
	if(fmt.fmt.pix.pixelformat != format) {
		fprintf(stderr, "requested format not supported\n");
		return -1;
	}

	return 0;
}

static int alloc_buffs(int fd, struct buf_info *buff_info, 
			int count, int type, int memory_type, int width, int height)
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

	if(!(type == V4L2_BUF_TYPE_VIDEO_CAPTURE)) {		
		printf("invalid buffer type\n");
		return -1;
	} 

	if (reqbuf.memory == V4L2_MEMORY_USERPTR) {

		for(i = 0; i < numbuffs; i++) {
			if (posix_memalign((void **)&buff_info[i].buf.m.userptr, 
					getpagesize(), width*height*2) != 0) {
				perror("Memory alloc failed posix_memalign \n");
			}
			buff_info[i].buf.type = reqbuf.type;
			buff_info[i].buf.index = i;
			buff_info[i].buf.memory = reqbuf.memory;
			buff_info[i].buf.length = width * height * 2;

			buff_info[i].length = buff_info[i].buf.length;
			buff_info[i].index   = i;
			buff_info[i].start   = (char *)buff_info[i].buf.m.userptr;
			buff_info[i].phy = (char *)buff_info[i].buf.m.userptr;
			printf("buff_info[%d].phy %p buff_info[%d].start %p Memory got as %p \n",i, buff_info[i].phy, i, buff_info[i].start, buff_info[i].buf.m.userptr);

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

	        	buff_info[i].length  = buf.length;
        		buff_info[i].index   = i;
        		buff_info[i].start   = (char *) mmap(NULL, buf.length, 
                						PROT_READ|
								PROT_WRITE, 
								MAP_SHARED, 
								fd, 
								buf.m.offset);
			if(ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
				perror("VIDIOC_QUERYBUF");
				numbuffs = i;
			}
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

	return 0;
}

static int start_streaming(int fd, int buf_type) 
{	
	if(ioctl(fd, VIDIOC_STREAMON, &buf_type) < 0) {
		perror("VIDIOC_STREAMON");
		return -1;
	}
	return 0;
}

int dequeue_buffer(int fd, struct v4l2_buffer *buffer)
{
	if(ioctl(fd, VIDIOC_DQBUF, buffer) < 0) {
		perror("VIDIOC_DQBUF");
		fprintf(stderr, "Problem occured in deque type: %d\n", buffer->type);
		return -1;
	}
	return 0;
}
int enqueue_buffer(int fd, struct v4l2_buffer *buffer)
{
	if(ioctl(fd, VIDIOC_QBUF, buffer)) {
		perror("VIDIOIC_QBUF");
		fprintf(stderr, "problem occured in enqueue type: %d\n", buffer->type);
		return -1;
	}
	return 0;
}

void *get_frame_virt(struct cam_info *cam, struct v4l2_buffer *capture_buff)
{
	return (void *)cam->buf[capture_buff->index].start;
}

void *get_frame_phy(struct cam_info *cam, struct v4l2_buffer *capture_buff)
{
	return (void *)cam->buf[capture_buff->index].phy;
}

int init_cam(char *dev, struct buf_info *bufinfo, int width, int height)
{
	int fd = -1;
	int format_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	int ret; 
	
	fd = open_cam(dev);
	if (fd < 0) {
		return -1;
	}
	
	/* Check the device is capable of streaming */
	if(check_capability(fd, V4L2_CAP_STREAMING)) {
		fprintf(stdout, "%s Device is capable for streaming\n", dev);
	} else {
		fprintf(stdout, "%s Device is not Capable for streaing\n", dev);
		return -1;
	}
	ret = set_format_info(fd, format_type, V4L2_PIX_FMT_YUYV, width, height);
	if (ret < 0) {
		return ret;
	}

	ret = alloc_buffs(fd, bufinfo, MAX_CAM_BUFFERS, 
			V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP, 
			width, height);
	if (ret < 0) {
		return ret;
	}
	
	ret = start_streaming(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE);
	if (ret < 0) {
		return ret;
	}
				
	return fd;
}
