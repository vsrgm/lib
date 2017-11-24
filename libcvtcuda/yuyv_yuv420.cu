#include <stdio.h>
#include <stdlib.h>
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

__host__ __device__ 
int yuyv_yuv420_int_color_convert(	int blockid, int threadid, unsigned int *des_buf, 
					unsigned int *src_buf, int width, int height, 
					int blockid_count, int threadid_count) {

	int src_height, src_width, src_buf_index, des_buf_index;
	unsigned int Y1U1Y2V1;
	unsigned int Y3U2Y4V2;
	unsigned int Y5U3Y6V3;
	unsigned int Y7U4Y8V4;

	unsigned char Y1, U1, Y2, V1;
	unsigned char Y3, U2, Y4, V2;
	unsigned char Y5, U3, Y6, V3;
	unsigned char Y7, U4, Y8, V4;

	unsigned int *u_des_buf = des_buf + (width * height/4);
	unsigned int *v_des_buf = u_des_buf + (width * height)/16;

	for (src_height = threadid; src_height < height; src_height+=threadid_count) {
		for (src_width = (blockid<<2); src_width < (width*2); src_width+=(blockid_count<<2)) {
			src_buf_index = src_height*(width/2) + src_width;
			Y1U1Y2V1 = src_buf[src_buf_index];
			V1 = 0xFF & (Y1U1Y2V1 >> 24);
			Y2  = 0xFF & (Y1U1Y2V1 >> 16);
			U1 = 0xFF & (Y1U1Y2V1 >> 8);
			Y1  = 0xFF & (Y1U1Y2V1);

			Y3U2Y4V2 = src_buf[src_buf_index +1];
			V2 = 0xFF & (Y3U2Y4V2 >> 24);
			Y4  = 0xFF & (Y3U2Y4V2 >> 16);
			U2 = 0xFF & (Y3U2Y4V2 >> 8);
			Y3  = 0xFF & (Y3U2Y4V2);

			Y5U3Y6V3 = src_buf[src_buf_index +2];
			V3 = 0xFF & (Y5U3Y6V3 >> 24);
			Y6  = 0xFF & (Y5U3Y6V3 >> 16);
			U3 = 0xFF & (Y5U3Y6V3 >> 8);
			Y5  = 0xFF & (Y5U3Y6V3);

			Y7U4Y8V4 = src_buf[src_buf_index +3];
			V4 = 0xFF & (Y7U4Y8V4 >> 24);
			Y8  = 0xFF & (Y7U4Y8V4 >> 16);
			U4 = 0xFF & (Y7U4Y8V4 >> 8);
			Y7  = 0xFF & (Y7U4Y8V4);
				
			des_buf_index = src_height*(width/4) + src_width/2;
			des_buf[des_buf_index]	 = Y4 <<24 | Y3 <<16 | Y2 <<8 | Y1;
			des_buf[des_buf_index+1] = Y8 <<24 | Y7 <<16 | Y6 <<8 | Y5;

			des_buf_index = (src_height*width)/16 + src_width/4;
			u_des_buf[des_buf_index] = U4 << 24 | U3 <<16 | U2 <<8 | U1;
			v_des_buf[des_buf_index] = V4 << 24 | V3 <<16 | V2 <<8 | V1;
		}
	}
	return 0;
}

__host__ __device__ 
int yuyv_yuv420_color_convert(	int blockid, int threadid, unsigned char *des_buf, 
				unsigned char *src_buf, int width, int height, 
				int blockid_count, int threadid_count) {
	unsigned char *u_des_buf = des_buf + (width * height);
	unsigned char *v_des_buf = des_buf + (width * height) + (width * (height>>2));

	int width_inc, height_inc;
	int sindex, src_index;
	int duvindex;

	for (height_inc = blockid; height_inc < height ;height_inc+=blockid_count) {
		sindex = height_inc * width;
		for (width_inc =threadid; width_inc < width ;width_inc+=threadid_count) {
			src_index = (sindex + width_inc)<<1;
			des_buf [sindex + width_inc]= src_buf[src_index];
			if (height_inc%2==0) {
				duvindex = (height_inc * width)>>2;
				if ((src_index+1)%4 == 1) {
					u_des_buf[duvindex + (width_inc>>1)] = src_buf[src_index+1];
				}else {
					v_des_buf[duvindex + (width_inc>>1)] = src_buf[src_index+1];
				}
			}
		}
	}
	return 0;
}
