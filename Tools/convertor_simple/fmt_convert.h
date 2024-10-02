#ifndef FMT_CONVERT_H
#define FMT_CONVERT_H

int32_t convert_bgr888_rgb888(uint8_t *src_buffer, uint8_t *dest_buffer,
        uint32_t width, uint32_t height);
int32_t convert_y8_rgb888(uint8_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height);
int32_t convert_y16_rgb888(uint16_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height);
int32_t convert_bayer12_bayer8(uint8_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height);
int32_t convert_rccg_rgb24(
        uint8_t *src_buffer, uint8_t *dest_buffer,
        int32_t width, int32_t height, uint8_t pc,
        uint32_t bpp, uint32_t shift);
int32_t convert_bayer_rgb24(uint8_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height, uint8_t start_with_b, uint32_t bpp, uint32_t shift);
int32_t convert_yuyv_rgb888(uint8_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height, int32_t start_with);
int32_t convert_bmp_565_bmp_888(char *src_buffer, char *des_buffer, int32_t width, int32_t height);
int32_t convert_bayer_gen_rgb24(uint16_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height, int32_t start_with, int32_t shift);
int32_t convert_yuy422p_rgb888(uint8_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height);
int32_t convert_rgb555_888(uint8_t* rgb565,uint8_t* rgb888, uint32_t width, uint32_t height, int32_t start_with);
int32_t convert_rgb565_888(uint8_t* rgb565,uint8_t* rgb888, uint32_t width, uint32_t height, int32_t start_with);
int32_t convert_yuy420p_rgb888(uint8_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height);
int32_t convert_nv12_rgb888(uint8_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height);
int32_t convert_argb32_rgb(uint8_t *src_buffer, uint8_t *des_buffer, int32_t width, int32_t height);
int32_t convert_bayer10_packed_rgbir(uint8_t *src_buffer, uint8_t *des_buffer, int32_t width, int32_t height);
int32_t save_asyuv(uint8_t *des_buffer, uint32_t width, uint32_t height);
int32_t perform_equalize_rgb24 (uint8_t *ptr, uint32_t width, uint32_t height);
int32_t save_ir_asyuv(uint8_t *des_buffer, uint32_t width, uint32_t height);
int32_t extract_bayer10_packed_ir(uint8_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height);
int32_t save_buffer(uint8_t *des_buffer, uint32_t size);
int32_t convert_RGBIR16_bayer8(uint8_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height);
int32_t extract_RGBIR16_IR8(uint8_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height);
int32_t perform_equalize_y8 (uint8_t *ptr, uint32_t width, uint32_t height);
int32_t convert_bit16_bit8(uint16_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height);
int32_t convert_yuy420_rgb888(uint8_t* yuyv_buffer, uint8_t* rgb888,
       uint32_t width, uint32_t height);
int32_t perform_crop(uint8_t *dptr, uint8_t *sptr, uint32_t x, uint32_t y,
                 uint32_t width, uint32_t height, uint32_t bpp,
                 uint32_t srcwidth, uint32_t srcheight);
int32_t perform_stride_correction(uint8_t *dptr, uint8_t *sptr,
        uint32_t dwidth, uint32_t dheight,
        uint32_t swidth, uint32_t sheight, uint32_t bpp);


#endif // FMT_CONVERT_H
