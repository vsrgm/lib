#include <stdio.h>
#include <stdlib.h>

int yuyv_to_yuv420_opt(unsigned char *des_buf, unsigned char *src_buf, int width, int height)
{
	unsigned char *u_des_buf = des_buf + (width * height);
	unsigned char *v_des_buf = des_buf + (width * height) + (width * (height>>2));
	int height_inc, sindex, duvindex, width_inc, src_index;
	for (height_inc = 0; height_inc < height ;height_inc++) {
		sindex = height_inc * width;
		for (width_inc =0; width_inc < width ;width_inc++) {
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
