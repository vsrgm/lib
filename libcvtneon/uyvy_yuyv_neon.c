#include <arm_neon.h>
int 
uyvy_yuyv_neon(unsigned char *des_buf, unsigned char *src_buf, int width, int height) {
	int inc = 0;
	for (inc = 0; inc < (width*height*2); inc+=8) {
		uint8x8_t rev = vld1_u8(src_buf + inc);
		rev = vrev16_u8(rev);
		vst1_u8(des_buf + inc, rev);
	}
	return 0;
}
