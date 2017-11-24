/**
 * @file camera.h
 * @brief camera proto types 
 * @version v0.1
 * @date 2015-10-01
 */

#ifndef __CAM1_MIPI_H__
#define __CAM1_MIPI_H__

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

struct buf_info {
	int index;
	unsigned int length;
	char *start;
	char *phy;
	struct v4l2_buffer buf;
};

#define CAPTURE_MAX_BUFFER_MIPI1 6

/**
 *@brief: dequeue a v4l2 buffer
 *@param[in]: fd, file descriptor of a v4l2device node
 *@param[out]: buffer, v4l2_buffer will be filled to this parameter
 *@return: on success value 0 will be returned on fail value < 0 will be returned
 */
int dequeue_buffer_mipi1(int fd,struct v4l2_buffer *buffer);

/**
 *@brief: enqueue a v4l2 buffer
 *@param[in]: fd, file descriptor of a v4l2 device node
 *@param[in]: buffer, v4l2_buffer will be queued back to v4l2 subsystem
 *@return: on success value 0 will be returned on fail value < 0 will be returned
 */
int enqueue_buffer_mipi1(int fd,struct v4l2_buffer *buffer);

/**
 * @brief: open and initialize a capture device
 *
 * @param height: height of the image to be captured
 * @parami width: width of the image to be captured
 *
 * @return: on success this function will return 0
 */
int init_camera_mipi1(char *device_node, int width,int height, int fmt);


/**
 * @brief : get frame from capture buffer
 *
 * @param capture_buff: v4l2 capture buffer
 *
 * @return 
 */
void *get_frame_virt_mipi1(struct v4l2_buffer *capture_buff);

void *get_frame_phy_mipi1(struct v4l2_buffer *capture_buff);

#endif
