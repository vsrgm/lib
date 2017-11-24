#include <arm_neon.h>
#include <stdio.h>
int yuyv_yuv420_color_convert(unsigned char *des_buf, unsigned char *src_buf, int width, int height) {
	int src_inc, des_inc, uv_count, uv_des_buf_count;
	unsigned char *u_des_buf = des_buf + (width*height);
	unsigned char *v_des_buf = des_buf + (width*height) + (width*height)/4;

	for (src_inc =0, des_inc=0, uv_count=0, uv_des_buf_count=0; src_inc < (width*height *2); src_inc+=32) {
		uint8x8x4_t temp = vld4_u8(src_buf + src_inc);
		uint8x8_t u = temp.val[1];
		uint8x8_t v = temp.val[3];
		uint8x8x2_t y1y2;
		y1y2.val[0]=temp.val[0];
		y1y2.val[1]=temp.val[2];
    		vst2_u8 (des_buf, y1y2);
		des_buf+=16;
		if (uv_count < (width >> 4)) { //width*2/32
  	 		vst1_u8 (u_des_buf, u);
			u_des_buf+=8;
   			vst1_u8 (v_des_buf, v);
			v_des_buf+=8;
		}
		uv_count++;
		if (uv_count >=(width >> 3)) { //width*4/32
			uv_count=0;
		}
	}
	return 0;
}

int yuyv_yuv420_sidebyside_color_convert(unsigned char *des_buf, unsigned char *src_buf, int width, int height, int stride) {
	int src_inc, des_inc, uv_count, sbs_uv_count, sbs_width_count, uv_des_buf_count;
	unsigned char *u_des_buf = des_buf + (width*height*2);
	unsigned char *v_des_buf = u_des_buf + (width*height)/2;
	unsigned char *base = des_buf;

	des_buf +=stride;
	u_des_buf+=(stride >> 1);
	v_des_buf+=(stride >> 1);
//	printf("des_buf %u u_des_buf %u v_des_buf %u \n",des_buf-base, u_des_buf-base, v_des_buf-base);

	for (	src_inc =0, des_inc=0, uv_count=0, 
		sbs_uv_count=0, sbs_width_count=0,
		uv_des_buf_count=0; src_inc < (width*height *2); src_inc+=32) {

		uint8x8x4_t temp = vld4_u8(src_buf + src_inc);
		uint8x8_t u = temp.val[1];
		uint8x8_t v = temp.val[3];
		uint8x8x2_t y1y2;
		y1y2.val[0]=temp.val[0];
		y1y2.val[1]=temp.val[2];
    		vst2_u8 (des_buf, y1y2);
		des_buf+=16;

		if (uv_count < (width >> 4)) { //width*2/32
			vst1_u8 (u_des_buf, u);
			u_des_buf+=8;
   			vst1_u8 (v_des_buf, v);
			v_des_buf+=8;
		}

		uv_count++;
		sbs_width_count++;
		sbs_uv_count++;
		if (sbs_width_count == (width >> 4)) {
			des_buf+=width;		
			sbs_width_count = 0;
		}
	
		if (sbs_uv_count == (width >> 3)) {
			u_des_buf+=(width >> 1);
			v_des_buf+=(width >> 1);
			sbs_uv_count = 0;
		}

		if (uv_count >=(width >> 3)) { //width*4/32
			uv_count=0;
		}
	}
	return 0;
}
