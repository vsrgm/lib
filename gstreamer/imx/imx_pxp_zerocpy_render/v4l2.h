#include <stdio.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/queue.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#include <gst/video/video.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>

/* camera Width and height */
#define CAM_WIDTH 1920 
#define CAM_HEIGHT 1080

#define MAX_CAM 1
#define MAX_CAM_LEN 20
#define MAX_CAM_BUFFERS 5


struct buf_info {
	int index;
	unsigned int length;
	char *start;
	char *phy;
	struct v4l2_buffer buf;
};

struct cam_info {
	int fd;
	int width;
	int height;
	struct buf_info buf[MAX_CAM_BUFFERS];
	char *obuf;
};

struct cam_video_info {
	int width;
	int height;
	GstElement *camsink[MAX_CAM];
	int ping_pong_select;
	char *sink_ping, *sink_pong;
};
void *get_frame_virt(struct cam_info *cam, struct v4l2_buffer *capture_buff);

