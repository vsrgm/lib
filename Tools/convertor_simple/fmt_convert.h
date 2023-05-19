#ifndef FMT_CONVERT_H
#define FMT_CONVERT_H

int convert_y8_rgb888(unsigned char* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height);
int convert_y16_rgb888(unsigned short* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height);
int convert_bayer12_bayer8(unsigned char *src_buffer, unsigned char *dest_buffer, int width, int height);

int convert_bayer8_rgb24(unsigned char *src_buffer, unsigned char *dest_buffer, int width, int height, unsigned char start_with_b);
int convert_yuyv_rgb888(unsigned char* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height, int start_with);
int convert_bmp_565_bmp_888(char *src_buffer, char *des_buffer, int width, int height);
int convert_bayer_gen_rgb24(unsigned short *src_buffer, unsigned char *dest_buffer, int width, int height, int start_with, int shift);
int convert_yuy422p_rgb888(unsigned char* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height);
int convert_rgb555_888(unsigned char* rgb565,unsigned char* rgb888, unsigned int width, unsigned int height, int start_with);
int convert_rgb565_888(unsigned char* rgb565,unsigned char* rgb888, unsigned int width, unsigned int height, int start_with);
int convert_yuy420p_rgb888(unsigned char* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height);
int convert_nv12_rgb888(unsigned char* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height);
int convert_argb32_rgb(unsigned char *src_buffer, unsigned char *des_buffer, int width, int height);
int convert_bayer10_packed_rgbir(unsigned char *src_buffer, unsigned char *des_buffer, int width, int height);
int save_asyuv(unsigned char *des_buffer, unsigned int width, unsigned int height);
int perform_equalize_rgb24 (unsigned char *ptr, unsigned int width, unsigned int height);
int save_ir_asyuv(unsigned char *des_buffer, unsigned int width, unsigned int height);
int extract_bayer10_packed_ir(unsigned char *src_buffer, unsigned char *dest_buffer, int width, int height);
int save_buffer(unsigned char *des_buffer, unsigned int size);

#endif // FMT_CONVERT_H
