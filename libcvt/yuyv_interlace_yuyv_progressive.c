#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int yuyv_interlace_yuv_progressive(unsigned char *des_buf1,unsigned char *des_buf2, unsigned char *src_buf, int src_width, int src_height)
{
	int src_inc, des_inc, src_width_index;
	for (src_inc =0; src_inc < src_height; ) {
		memcpy(&des_buf1[(src_inc/2)*src_width*2], &src_buf[src_inc*src_width*2], src_width*2);
		src_inc++;
		memcpy(&des_buf2[(src_inc/2)*src_width*2], &src_buf[src_inc*src_width*2], src_width*2);
		src_inc++;
	}
	return 0;
}
