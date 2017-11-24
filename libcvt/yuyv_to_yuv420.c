#include <stdio.h>
#include <stdlib.h>

int yuyv_to_yuv420(unsigned char *des_buf, unsigned char *src_buf, int width, int height)
{
	int src_inc, des_inc, src_height, src_width_index, src_width;
	for (src_inc =0,des_inc=0; src_inc < (width * height*2);src_inc+=2) {
		des_buf [des_inc++]= src_buf[src_inc];
	}

	for (src_height =0; src_height <height; src_height+=2) {
		src_width_index = src_height*width*2;
		for (src_width =1; src_width <(width*2); src_width+=4) 
			des_buf [des_inc++] = (src_buf [src_width_index + src_width] + src_buf [src_width_index + width*2 + src_width])>>1;
	}

	for (src_height =0; src_height <height; src_height+=2) {
		src_width_index = src_height*width*2;
		for (src_width =3; src_width <(width*2); src_width+=4) 
			des_buf [des_inc++] = (src_buf [src_width_index + src_width] + src_buf [src_width_index + width*2 + src_width])>>1;
	}
	return 0;
}
