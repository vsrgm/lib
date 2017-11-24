#ifndef __YUV_YUV420_H
#define __YUV_YUV420_H
__host__ __device__ 
int yuyv_yuv420_color_convert(	int blockid, int threadid, unsigned char *des_buf, 
				unsigned char *src_buf, int width, int height, 
				int blockid_count, int threadid_count);
#endif

