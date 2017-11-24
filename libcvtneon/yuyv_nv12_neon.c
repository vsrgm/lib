#include <arm_neon.h>

int 
yuyv_nv12(unsigned char *des_buf, unsigned char *src_buf, int width, int height) {
	unsigned char *uv_des_buf = des_buf + (width * height);
	int src_inc, des_inc, uv_count,uv_des_buf_count;

	for (src_inc =0, des_inc=0, uv_count=0, uv_des_buf_count=0; src_inc < (width *height *2); src_inc+=16) {

		uint8x8x2_t temp = vld2_u8(src_buf + src_inc);
		uint8x8_t y1 = temp.val[0];
		uint8x8_t uv = temp.val[1];
    		vst1_u8 (des_buf + des_inc, y1);
		des_inc+=8;

		if (uv_count < (width >> 3)) {
			// (1 line = 1920*2 = 240)
	    		vst1_u8 (uv_des_buf + uv_des_buf_count, uv);
			uv_des_buf_count+=8;
		}

		uv_count++;
		if (uv_count >= (width >>2)) {
			uv_count=0;
		}
	}
	return 0;
}
