#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <utility>
//#include <arpa/inet.h>
#define CLIP(x) (((x) > 0xFF) ? 0xFF : (((x) < 0) ? 0 :(uint8_t)(x)))

int32_t convert_bgr888_rgb888(uint8_t *src_buffer, uint8_t *dest_buffer,
        uint32_t width, uint32_t height)
{
    for (uint32_t hidx = 0; hidx < height; hidx++)
    {
        for (uint32_t widx = 0; widx < width; widx++)
        {
            uint32_t offset = ((hidx * width) + widx) * 3;
            dest_buffer[offset]     = src_buffer[offset + 2];
            dest_buffer[offset + 1] = src_buffer[offset + 1];
            dest_buffer[offset + 2] = src_buffer[offset];
        }
    }
    return 0;
}

int32_t convert_yuyv420_rgb888(uint8_t* yuyv_buffer,uint8_t* rgb888,
       uint32_t width, uint32_t height, int32_t start_with)
{
    /* 
     *    start_with
     *    YUYV = 0
     *    YVYU = 1
     *    UYVY = 2
     *    VYUY = 3
     */
    int32_t hidx = 0, widx = 0;
    uint8_t* rgb_buffer;
    int32_t temp;
    int32_t yuvcount = 0;
    //int32_t rgbcount = 0;
    float u_val, v_val, y1_val, y2_val;

    rgb_buffer = rgb888;

    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'

    for (hidx = 0; hidx < height; hidx++)
    {
        for (widx = 0; widx < width; widx+= 2)
        {
            switch (start_with)
            {
                case 0: {
                            y1_val = (float)yuyv_buffer[yuvcount++];
                            u_val = (float)yuyv_buffer[yuvcount++];
                            y2_val = (float)yuyv_buffer[yuvcount++];
                            v_val = (float)yuyv_buffer[yuvcount++];
                        }break;

                case 1: {
                            y1_val = (float)yuyv_buffer[yuvcount++];
                            v_val = (float)yuyv_buffer[yuvcount++];
                            y2_val = (float)yuyv_buffer[yuvcount++];
                            u_val = (float)yuyv_buffer[yuvcount++];
                        }break;

                case 2: {
                            u_val = (float)yuyv_buffer[yuvcount++];
                            y1_val = (float)yuyv_buffer[yuvcount++];
                            v_val = (float)yuyv_buffer[yuvcount++];
                            y2_val = (float)yuyv_buffer[yuvcount++];
                        }break;

                case 3: {
                            v_val = (float)yuyv_buffer[yuvcount++];
                            y1_val = (float)yuyv_buffer[yuvcount++];
                            u_val = (float)yuyv_buffer[yuvcount++];
                            y2_val = (float)yuyv_buffer[yuvcount++];
                        }break;
            }

            u_val = u_val - 128;
            v_val = v_val - 128;        

            temp = (int32_t)(y1_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +0] = CLIP(temp);

            temp = (int32_t)(y1_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +1] = CLIP(temp);

            temp = (int32_t)(y1_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +2] = CLIP(temp);

            temp = (int32_t)(y2_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +3] = CLIP(temp);

            temp = (int32_t)(y2_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +4] = CLIP(temp);

            temp = (int32_t)(y2_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +5] = CLIP(temp);
        }
    }
    return 0;
}

int32_t convert_nv12_rgb888(uint8_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height)
{
    /* 
     *    start_with
     *    YUYV = 0
     *    YVYU = 1
     *    UYVY = 2
     *    VYUY = 3
     */
    uint8_t* rgb_buffer;
    int32_t temp;
    int32_t yuvcount = 0;
    int32_t u_count = (width * height);
    int32_t v_count = u_count + (u_count/4);
    //    int32_t rgbcount = 0;
    //    float u_val, v_val, y1_val, y2_val,y3_val, y4_val;
    uint32_t time;

    rgb_buffer = rgb888;
    uint8_t *y_channel = yuyv_buffer;
    uint8_t *u_channel = yuyv_buffer + u_count;
    uint8_t *v_channel = yuyv_buffer + v_count;

    int32_t r,g,b;
    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'
    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            int32_t yy = y_channel[(y * width) + x];
            int32_t uu = u_channel[(((y / 2) * (width / 2)) + (x / 2))*2];
            int32_t vv = u_channel[(((y / 2) * (width / 2)) + (x / 2))*2 +1];

            r = 1.164 * (yy - 16) + 1.596 * (vv - 128);
            g = 1.164 * (yy - 16) - 0.813 * (vv - 128) - 0.391 * (uu - 128);
            b = 1.164 * (yy - 16) + 2.018 * (uu - 128);
            *rgb_buffer++ = CLIP(r);
            *rgb_buffer++ = CLIP(g);
            *rgb_buffer++ = CLIP(b);
        }
    }
    return 0;
}

int32_t convert_yuy420p_rgb888(uint8_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height)
{
    /* 
     *    start_with
     *    YUYV = 0
     *    YVYU = 1
     *    UYVY = 2
     *    VYUY = 3
     */
    uint8_t* rgb_buffer;
    int32_t temp;
    int32_t u_count = (width * height);
    int32_t v_count = u_count + (u_count/4);
    int32_t rgbcount = 0;

    rgb_buffer = rgb888;
    uint8_t *y_channel = yuyv_buffer;
    uint8_t *u_channel = yuyv_buffer + u_count;
    uint8_t *v_channel = yuyv_buffer + v_count;

    int32_t r,g,b;
    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'
    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            int32_t yy = y_channel[(y * width) + x];
            int32_t uu = u_channel[((y / 2) * (width / 2)) + (x / 2)];
            int32_t vv = v_channel[((y / 2) * (width / 2)) + (x / 2)];

            r = 1.164 * (yy - 16) + 1.596 * (vv - 128);
            g = 1.164 * (yy - 16) - 0.813 * (vv - 128) - 0.391 * (uu - 128);
            b = 1.164 * (yy - 16) + 2.018 * (uu - 128);
            *rgb_buffer++ = CLIP(r);
            *rgb_buffer++ = CLIP(g);
            *rgb_buffer++ = CLIP(b);
        }
    }
    return 0;
}

int32_t convert_yuy420_rgb888(uint8_t *yuyv_buffer, uint8_t *rgb888,
        uint32_t width, uint32_t height)
{
    return convert_yuy420p_rgb888(yuyv_buffer, rgb888, width, height);
}

int32_t convert_yuy422p_rgb888(uint8_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height)
{
    /* 
     *    start_with
     *    YUYV = 0
     *    YVYU = 1
     *    UYVY = 2
     *    VYUY = 3
     */
    int32_t widx = 0, hidx = 0;
    uint8_t* rgb_buffer;
    int32_t temp;
    int32_t yuvcount = 0;
    int32_t u_count = (width * height);
    int32_t v_count = u_count + (u_count/2);
    int32_t rgbcount = 0;
    float u_val, v_val, y1_val, y2_val;

    rgb_buffer = rgb888;

    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'

    for (hidx = 0; hidx < height; hidx++)
    {
        for (widx = 0; widx < width; widx+= 2)
        {
            y1_val = (float)yuyv_buffer[yuvcount++];
            u_val = (float)yuyv_buffer[u_count++];
            y2_val = (float)yuyv_buffer[yuvcount++];
            v_val = (float)yuyv_buffer[v_count++];

            u_val = u_val - 128;
            v_val = v_val - 128;        

            temp = (int32_t)(y1_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +0] = CLIP(temp);

            temp = (int32_t)(y1_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +1] = CLIP(temp);

            temp = (int32_t)(y1_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +2] = CLIP(temp);

            temp = (int32_t)(y2_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +3] = CLIP(temp);

            temp = (int32_t)(y2_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +4] = CLIP(temp);

            temp = (int32_t)(y2_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +5] = CLIP(temp);
        }
    }
    return 0;
}
int32_t convert_y8_rgb888(uint8_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height)
{
    int32_t hidx = 0, widx = 0;
    uint8_t* rgb_buffer;
    int32_t yuvcount = 0;
    float y_val;
    uint32_t time;

    rgb_buffer = rgb888;

    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'

    for (hidx = 0; hidx < height; hidx++)
    {
        for (widx = 0; widx < width; widx++)
        {
            y_val = (float)yuyv_buffer[yuvcount++];

            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +0] = y_val;
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +1] = y_val; 
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +2] = y_val;
        }
    }
    return 0;
}

int32_t convert_y16_rgb888(uint16_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height)
{
    int32_t hidx = 0, widx = 0;
    uint8_t* rgb_buffer;
    int32_t yuvcount = 0;
    uint16_t y_val;
    uint32_t time;

    rgb_buffer = rgb888;

    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'

    for (hidx = 0; hidx < height; hidx++)
    {
        for (widx = 0; widx < width; widx++)
        {
            y_val = (uint16_t)yuyv_buffer[yuvcount++];

            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +0] = y_val>>3;
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +1] = y_val>>3;
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +2] = y_val>>3;
        }
    }
    return 0;
}

int32_t convert_yuyv_rgb888(uint8_t* yuyv_buffer,uint8_t* rgb888, uint32_t width, uint32_t height, int32_t start_with)
{
    /* 
     *    start_with
     *    YUYV = 0
     *    YVYU = 1
     *    UYVY = 2
     *    VYUY = 3
     */
    int32_t hidx = 0, widx = 0;
    uint8_t* rgb_buffer;
    int32_t temp;
    int32_t yuvcount = 0;
    int32_t rgbcount = 0;
    float u_val, v_val, y1_val, y2_val;
    uint32_t time;

    rgb_buffer = rgb888;

    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'

    for (hidx = 0; hidx < height; hidx++)
    {
        for (widx = 0; widx < width; widx+= 2)
        {
            switch (start_with)
            {
                case 0: {
                            y1_val = (float)yuyv_buffer[yuvcount++];
                            u_val = (float)yuyv_buffer[yuvcount++];
                            y2_val = (float)yuyv_buffer[yuvcount++];
                            v_val = (float)yuyv_buffer[yuvcount++];
                        }break;

                case 1: {
                            y1_val = (float)yuyv_buffer[yuvcount++];
                            v_val = (float)yuyv_buffer[yuvcount++];
                            y2_val = (float)yuyv_buffer[yuvcount++];
                            u_val = (float)yuyv_buffer[yuvcount++];
                        }break;

                case 2: {
                            u_val = (float)yuyv_buffer[yuvcount++];
                            y1_val = (float)yuyv_buffer[yuvcount++];
                            v_val = (float)yuyv_buffer[yuvcount++];
                            y2_val = (float)yuyv_buffer[yuvcount++];
                        }break;

                case 3: {
                            v_val = (float)yuyv_buffer[yuvcount++];
                            y1_val = (float)yuyv_buffer[yuvcount++];
                            u_val = (float)yuyv_buffer[yuvcount++];
                            y2_val = (float)yuyv_buffer[yuvcount++];
                        }break;
            }

            u_val = u_val - 128;
            v_val = v_val - 128;        

            temp = (int32_t)(y1_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +0] = CLIP(temp);

            temp = (int32_t)(y1_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +1] = CLIP(temp);

            temp = (int32_t)(y1_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +2] = CLIP(temp);

            temp = (int32_t)(y2_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +3] = CLIP(temp);

            temp = (int32_t)(y2_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +4] = CLIP(temp);

            temp = (int32_t)(y2_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-hidx) * width * 3) + widx*3 +5] = CLIP(temp);
        }
    }
    return 0;
}

int32_t perform_equalize_y8 (uint8_t *ptr, uint32_t width, uint32_t height)
{
    if (ptr == NULL || width == 0 || height == 0)
        return -1;

    uint8_t Y;
    uint32_t Y_count[256];
    memset(Y_count, 0x00, sizeof(Y_count));
    uint32_t Y_sum = 0;
    uint32_t cdf_Y_count[256];
    uint32_t cdf_Y_min = 0;

    uint8_t hv_Y[256];

    uint32_t Y_total_count = width * height;

    for(uint32_t hidx=0;hidx<(width*height);hidx++) {
        Y = ptr[hidx];

        /* perform Histogram */
        Y_count[Y]++;
    }
    /* compute cdf and h(v) */
    for (uint32_t cdf_idx=0; cdf_idx <256; cdf_idx++)
    {
        Y_sum += Y_count[cdf_idx];

        cdf_Y_min = (cdf_Y_min > 0)?cdf_Y_min:Y_sum;

        cdf_Y_count[cdf_idx] = Y_sum;
        /* round((cdf(v)-cdf(min)/(MxN)-cdf(min)x(L-1) */
        hv_Y[cdf_idx] = (Y_total_count == cdf_Y_min) ? static_cast<uint8_t>(cdf_idx) :
            CLIP(((cdf_Y_count[cdf_idx] - cdf_Y_min) * 255) / (Y_total_count - cdf_Y_min));
    }

    /* Equalize */
    for(uint32_t hidx=0;hidx<(width*height);hidx++) {
        ptr[hidx] = hv_Y[ptr[hidx]];
    }
    return 0;
}

int32_t perform_stride_correction(uint8_t *dptr, uint8_t *sptr,
        uint32_t dwidth, uint32_t dheight,
        uint32_t swidth, uint32_t sheight, uint32_t bpp)
{
    if ((dwidth > swidth) || (dheight > sheight))
    {
        return -1;
    }

    for (uint32_t hidx = 0; hidx < dheight; hidx++)
    {
        memcpy(&dptr[hidx*dwidth*bpp], &sptr[hidx*swidth*bpp], dwidth*bpp);
    }
    return 0;
}


int32_t perform_crop(uint8_t *dptr, uint8_t *sptr, uint32_t x, uint32_t y,
        uint32_t width, uint32_t height, uint32_t bpp,
        uint32_t srcwidth, uint32_t srcheight)
{
    if (dptr == NULL || sptr == NULL || width == 0 || height == 0 || bpp == 0 ||
            width > srcwidth || height > srcheight || x < width / 2 || y < height / 2)
    {
        return -1;
    }
    const uint32_t x_start = x - (width / 2);
    const uint32_t y_start = y - (height / 2);
    if (x_start > srcwidth - width || y_start > srcheight - height)
        return -1;

    for (uint32_t hidx = 0; hidx < height; hidx++)
    {
        memcpy(&dptr[hidx*width*bpp], &sptr[((y_start+hidx)*srcwidth + x_start)*bpp], width*bpp);
    }
    return 0;
}

int32_t perform_equalize_rgb24 (uint8_t *ptr, uint32_t width, uint32_t height)
{
    if (ptr == NULL || width == 0 || height == 0)
        return -1;

    uint8_t R,G,B;
    uint32_t R_count[256];
    uint32_t G_count[256];
    uint32_t B_count[256];
    uint32_t R_sum = 0;
    uint32_t G_sum = 0;
    uint32_t B_sum = 0;
    uint32_t cdf_R_count[256];
    uint32_t cdf_G_count[256];
    uint32_t cdf_B_count[256];
    uint32_t cdf_R_min = 0;
    uint32_t cdf_G_min = 0;
    uint32_t cdf_B_min = 0;

    uint8_t hv_R[256];
    uint8_t hv_G[256];
    uint8_t hv_B[256];

    uint32_t R_total_count = width * height;
    uint32_t G_total_count = R_total_count;
    uint32_t B_total_count = R_total_count;

    memset(R_count, 0x00, sizeof(R_count));
    memset(G_count, 0x00, sizeof(G_count));
    memset(B_count, 0x00, sizeof(B_count));

    for(uint32_t idx=0; idx<(width*height); idx++)
    {
        R = ptr[idx*3+2];
        G = ptr[idx*3+1];
        B = ptr[idx*3+0];

        /* perform Histogram */
        R_count[R]++;
        G_count[G]++;
        B_count[B]++;
    }

    /* compute cdf and h(v) */
    for (uint32_t cdf_idx=0; cdf_idx <256; cdf_idx++)
    {
        R_sum += R_count[cdf_idx];
        G_sum += G_count[cdf_idx];
        B_sum += B_count[cdf_idx];

        cdf_R_min = (cdf_R_min > 0)?cdf_R_min:R_sum;
        cdf_G_min = (cdf_G_min > 0)?cdf_G_min:G_sum;
        cdf_B_min = (cdf_B_min > 0)?cdf_B_min:B_sum;

        cdf_R_count[cdf_idx] = R_sum;
        cdf_G_count[cdf_idx] = G_sum;
        cdf_B_count[cdf_idx] = B_sum;
        /* round((cdf(v)-cdf(min)/(MxN)-cdf(min)x(L-1) */
        hv_R[cdf_idx] = (R_total_count == cdf_R_min) ? static_cast<uint8_t>(cdf_idx) :
            CLIP((((cdf_R_count[cdf_idx] - cdf_R_min) * 255) / (R_total_count - cdf_R_min)));
        hv_G[cdf_idx] = (G_total_count == cdf_G_min) ? static_cast<uint8_t>(cdf_idx) :
            CLIP((((cdf_G_count[cdf_idx] - cdf_G_min) * 255) / (G_total_count - cdf_G_min)));
        hv_B[cdf_idx] = (B_total_count == cdf_B_min) ? static_cast<uint8_t>(cdf_idx) :
            CLIP((((cdf_B_count[cdf_idx] - cdf_B_min) * 255) / (B_total_count - cdf_B_min)));
    }

    /* Equalize */
    for(uint32_t idx=0; idx<(width*height); idx++)
    {
        ptr[idx*3+2] = hv_R[ptr[idx*3+2]];
        ptr[idx*3+1] = hv_G[ptr[idx*3+1]];
        ptr[idx*3+0] = hv_B[ptr[idx*3+0]];
    }
    return 0;
}

int32_t convert_rgb444_888(uint8_t *rgb444, uint8_t *rgb888, uint32_t width, uint32_t height)
{
    if (rgb444 == NULL || rgb888 == NULL)
        return -1;

    const uint32_t pixelCount = width * height;
    for (uint32_t index = 0; index < pixelCount; ++index)
    {
        const uint16_t packed = static_cast<uint16_t>(rgb444[index * 2]) |
                (static_cast<uint16_t>(rgb444[index * 2 + 1]) << 8);
        const uint8_t blue = static_cast<uint8_t>((packed & 0x0f) * 17);
        const uint8_t green = static_cast<uint8_t>(((packed >> 4) & 0x0f) * 17);
        const uint8_t red = static_cast<uint8_t>(((packed >> 8) & 0x0f) * 17);
        rgb888[index * 3] = red;
        rgb888[index * 3 + 1] = green;
        rgb888[index * 3 + 2] = blue;
    }
    return 0;
}

int32_t convert_fisheye_undistort_rgb24(const uint8_t *source, uint8_t *destination,
        uint32_t width, uint32_t height, double fx, double fy, double cx, double cy,
        double k1, double k2, double k3, double k4)
{
    if (source == NULL || destination == NULL || width == 0 || height == 0 ||
            !std::isfinite(fx) || !std::isfinite(fy) || fx <= 0 || fy <= 0 ||
            !std::isfinite(cx) || !std::isfinite(cy) ||
            !std::isfinite(k1) || !std::isfinite(k2) ||
            !std::isfinite(k3) || !std::isfinite(k4))
        return -1;

    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
        {
            const double normalizedX = (static_cast<double>(x) - cx) / fx;
            const double normalizedY = (static_cast<double>(y) - cy) / fy;
            const double radius = std::sqrt(normalizedX * normalizedX + normalizedY * normalizedY);
            const double theta = std::atan(radius);
            const double theta2 = theta * theta;
            const double theta4 = theta2 * theta2;
            const double theta6 = theta4 * theta2;
            const double theta8 = theta4 * theta4;
            const double distortedTheta = theta *
                    (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
            const double scale = radius > 1e-12 ? distortedTheta / radius : 1.0;
            const double sourceX = fx * normalizedX * scale + cx;
            const double sourceY = fy * normalizedY * scale + cy;
            const size_t outputOffset = (static_cast<size_t>(y) * width + x) * 3;

            if (!std::isfinite(sourceX) || !std::isfinite(sourceY) ||
                    sourceX < 0 || sourceY < 0 || sourceX > width - 1.0 || sourceY > height - 1.0)
            {
                destination[outputOffset] = 0;
                destination[outputOffset + 1] = 0;
                destination[outputOffset + 2] = 0;
                continue;
            }

            const uint32_t x0 = static_cast<uint32_t>(std::floor(sourceX));
            const uint32_t y0 = static_cast<uint32_t>(std::floor(sourceY));
            const uint32_t x1 = std::min(x0 + 1, width - 1);
            const uint32_t y1 = std::min(y0 + 1, height - 1);
            const double xWeight = sourceX - x0;
            const double yWeight = sourceY - y0;
            const size_t topLeft = (static_cast<size_t>(y0) * width + x0) * 3;
            const size_t topRight = (static_cast<size_t>(y0) * width + x1) * 3;
            const size_t bottomLeft = (static_cast<size_t>(y1) * width + x0) * 3;
            const size_t bottomRight = (static_cast<size_t>(y1) * width + x1) * 3;

            for (int channel = 0; channel < 3; ++channel)
            {
                const double top = source[topLeft + channel] * (1.0 - xWeight) +
                        source[topRight + channel] * xWeight;
                const double bottom = source[bottomLeft + channel] * (1.0 - xWeight) +
                        source[bottomRight + channel] * xWeight;
                const long value = std::lround(top * (1.0 - yWeight) + bottom * yWeight);
                destination[outputOffset + channel] = static_cast<uint8_t>(
                    std::clamp(value, 0L, 255L));
            }
        }
    return 0;
}

int32_t convert_rgb555_888(uint8_t* inbuf, uint8_t* outbuf, uint32_t width, uint32_t height, int32_t start_with)
{
    uint32_t row_cnt, pix_cnt;
    uint32_t off1 = 0, off2 = 0;
    uint8_t  tbi1, tbi2, R5, G5, B5, R8, G8, B8;

    for (row_cnt = 0; row_cnt < height; row_cnt++) 
    {
        off1 = row_cnt * width * 2;
        off2 = row_cnt * width * 3;
        for(pix_cnt=0; pix_cnt < width; pix_cnt++)
        {
            tbi1 = inbuf[off1 + (pix_cnt * 2)];
            tbi2 = inbuf[off1 + (pix_cnt * 2) + 1];
            B5 = tbi1 & 0x1F;
            G5 = (((tbi1 & 0xE0) >> 5) | ((tbi2 & 0x03) << 3)) & 0x1F;
            R5 = (tbi2 >> 2) & 0x1F;
            R8 = ( R5 * 527 + 23 ) >> 6;
            G8 = ( G5 * 527 + 23 ) >> 6;
            B8 = ( B5 * 527 + 23 ) >> 6;
            outbuf[off2 + (pix_cnt * 3)] = R8;
            outbuf[off2 + (pix_cnt * 3) + 1] = G8;
            outbuf[off2 + (pix_cnt * 3) + 2] = B8;
        }
    }
    return 0;
}

#if 0
int32_t convert_rgb555_888(uint8_t* rgb565,uint8_t* rgb888, uint32_t width, uint32_t height, int32_t start_with)
{

    /*
     * RGGB = 0
     * GBRG = 1
     * BGGR = 2
     * GRBG = 3
     */
    int32_t r,g,b,rg,gb,bg,gr,hidx,widx;

    for (hidx=0; hidx<height; hidx++)
    {
        for(widx=0; widx<width; widx++)
        {
            switch (start_with)
            {
                case 0:
                    {
                        rg = rgb565[(hidx * width*2)+widx*2+0];
                        gb = rgb565[(hidx * width*2)+widx*2+1];
                        r = (0x7C & rg) << 1;
                        g = ((rg & 0x3)<<6) | ((gb&0xE0)>>2);
                        b = (gb & 0x1F) << 2;
                    }break;

                case 1:
                    {
                        gb = rgb565[(hidx * width*2)+widx*2+0];
                        rg = rgb565[(hidx * width*2)+widx*2+1];
                        r = (0x7C & rg) << 1;
                        g = ((rg & 0x3)<<6) | ((gb&0xE0)>>2);
                        b = (gb & 0x1F) << 2;

                    }break;

                case 2:
                    {
                        bg = rgb565[(hidx * width*2)+widx*2+0];
                        gr = rgb565[(hidx * width*2)+widx*2+1];
                        b = (0x7C & bg) << 1;
                        g = ((bg & 0x3)<<6) | ((gr&0xE0)>>2);
                        r = (gr & 0x1F) << 2;

                    }break;

                case 3:
                    {
                        gr = rgb565[(hidx * width*2)+widx*2+0];
                        bg = rgb565[(hidx * width*2)+widx*2+1];

                    }break;
            }

            rgb888[(((height-1)-hidx) * width * 3)+ widx*3 +0] = 0xFF & b;
            rgb888[(((height-1)-hidx) * width * 3)+ widx*3 +1] = 0xFF & g;
            rgb888[(((height-1)-hidx) * width * 3)+ widx*3 +2] = 0xFF & r;
        }
    }
    return 0;
}
#endif

int32_t convert_rgb565_888(uint8_t* rgb565,uint8_t* rgb888, uint32_t width, uint32_t height, int32_t start_with)
{

    /*
     * RGGB = 0
     * GBRG = 1
     * BGGR = 2
     * GRBG = 3
     */
    if (rgb565 == NULL || rgb888 == NULL || start_with < 0 || start_with > 3)
        return -1;

    for (uint32_t hidx = 0; hidx < height; ++hidx)
    {
        for (uint32_t widx = 0; widx < width; ++widx)
        {
            const uint32_t inputOffset = (hidx * width + widx) * 2;
            const bool reverseBytes = (start_with & 1) != 0;
            const uint8_t high = rgb565[inputOffset + (reverseBytes ? 1 : 0)];
            const uint8_t low = rgb565[inputOffset + (reverseBytes ? 0 : 1)];
            uint8_t red = static_cast<uint8_t>((high & 0xf8) | (high >> 5));
            const uint8_t green5 = static_cast<uint8_t>(((high & 0x07) << 3) | (low >> 5));
            uint8_t green = static_cast<uint8_t>((green5 << 2) | (green5 >> 4));
            uint8_t blue = static_cast<uint8_t>(((low & 0x1f) << 3) | ((low & 0x1f) >> 2));
            if (start_with >= 2)
                std::swap(red, blue);

            const uint32_t outputOffset = (hidx * width + widx) * 3;
            rgb888[outputOffset] = red;
            rgb888[outputOffset + 1] = green;
            rgb888[outputOffset + 2] = blue;
        }
    }
    return 0;
}

int32_t convert_bayer12_bayer8(uint8_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height)
{
    int32_t index =0;
    int32_t hindex, windex;
    for (hindex = 0; hindex < height; hindex++)
    {
        for (windex = 0; windex < width/2; windex++)
        {
            dest_buffer[index++] = src_buffer[(int32_t)(hindex*width*1.5)+ (windex * 3) +0];
            dest_buffer[index] = (src_buffer[(int32_t)(hindex*width*1.5)+ (windex * 3) +1] & 0XF)<<4;
            dest_buffer[index++] |= (src_buffer[(int32_t)(hindex*width*1.5)+ (windex * 3) +1] & 0XF0)>>4;
        }
    }
    return 0;
}

int32_t convert_rccg_rgb24(
        uint8_t *src_buffer, uint8_t *dest_buffer,
        int32_t width, int32_t height, uint8_t pc,
        uint32_t bpp, uint32_t shift)
{
    int32_t bayer_step    = width;
    uint32_t idx, width_end_watch;
    /*
     * pc = 0 = BGGR
     * pc = 1 = GBRG
     * pc = 2 = RGGB
     * pc = 3 = GRBG
     */
    int32_t pattern[4][3][2][2]= {
        {
            /* B offset for BGGR */
            {
                {0,        -1,},
                {bayer_step,    bayer_step-1,},
            },
            /* G offset for BGGR */
            {
                {1,        0,},
                {0,        bayer_step,},
            },
            /* R offset for BGGR */
            {
                {bayer_step+1,    bayer_step,},
                {1,        0,},
            },
        },
        {
            /* B offset for GBRG */
            {
                {1,        0,},
                {bayer_step +1,    bayer_step,},
            },
            /* G offset for GBRG */
            {
                {0,        -1,},
                {1,        0,},
            },
            /* R offset for GBRG */
            {
                {bayer_step,    bayer_step-1,},
                {0,        -1,},
            },
        },
        {
            /* B offset for RGGB */
            {
                {bayer_step +1,    bayer_step,},
                {1,        0,},
            },
            /* G offset for RGGB */
            {
                {1,        0,},
                {0,        bayer_step,},
            },
            /* R offset for RGGB */
            {
                {0,        -1,},
                {bayer_step,    bayer_step-1,},
            },
        },
        {
            /* B offset for GRBG */
            {
                {bayer_step,    bayer_step-1,},
                {0,        -1,},
            },
            /* G offset for GRBG */
            {
                {0,        -1,},
                {1,        0,},
            },
            /* R offset for GRBG */
            {
                {1,        0,},
                {bayer_step+1,    bayer_step,},
            },
        },
    };

    if (bpp == 8)
    {
        uint8_t *s_buffer = src_buffer;
        for(idx = 0, width_end_watch = 0; idx < width*(height-1); idx++)
        {
            uint16_t r, c, g, b;
            g = s_buffer[pattern[pc][0][width_end_watch][idx&1] +idx];
            c = s_buffer[pattern[pc][1][width_end_watch][idx&1] +idx];
            r = s_buffer[pattern[pc][2][width_end_watch][idx&1] +idx];

            b = c - (0.7*r)- g;

            dest_buffer[idx*3+2] = b;
            dest_buffer[idx*3+1] = g;
            dest_buffer[idx*3+0] = r;

            if(((idx+1)%width) == 0)
            {
                width_end_watch = width_end_watch?0:1;
            }
        }
    }else
    {
        uint16_t *s_buffer = (uint16_t *)src_buffer;
        for(idx = 0, width_end_watch = 0; idx < width*(height-1); idx++)
        {
            uint16_t rccg_r, rccg_c,rccg_g;
            uint8_t bgr_b, bgr_g, bgr_r;
            int32_t d_rccg_r, d_rccg_c, d_rccg_g;
            uint32_t s_rgb_r, s_rgb_g, s_rgb_b;
            uint32_t idx_g, idx_c, idx_r;
            uint32_t y[]={0, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 20480,
                24576, 32768, 40960, 49152, 57344, 65536, 81920, 98304, 114688,
                131072, 163840, 196608, 262144, 393216, 524288, 786432, 1048576,
                1572864, 2097152, 3145728, 4194304, 6291456, 8388608, 12582912, 16777216};
            uint32_t x[]={0, 1024, 2047, 3071, 3583, 4095, 4607, 5119, 5631,
                6143, 6655, 7167, 7679, 8191, 8703, 9727, 10239, 10751, 11263,
                12287, 13311, 14335, 16383, 18431, 22527, 24575, 28671, 32767,
                36863, 40959, 45055, 49151, 57343, 65535};
 
            rccg_g = s_buffer[pattern[pc][0][width_end_watch][idx&1] +idx];
            rccg_c = s_buffer[pattern[pc][1][width_end_watch][idx&1] +idx];
            rccg_r = s_buffer[pattern[pc][2][width_end_watch][idx&1] +idx];
 
            for (idx_g=1; idx_g < (sizeof(x)/sizeof(x[0])); idx_g++)
            {
                if (x[idx_g] > rccg_g)
                    break;
            }
            
            d_rccg_g = y[idx_g-1] + ((rccg_g - x[idx_g-1]) * (y[idx_g]-y[idx_g-1])/(x[idx_g]-x[idx_g-1]));
            for (idx_c=1; idx_c < (sizeof(x)/sizeof(x[0])); idx_c++)
            {
                if (x[idx_c] > rccg_c)
                    break;
            }
            d_rccg_c = y[idx_c-1] + ((rccg_c - x[idx_c-1]) * (y[idx_c]-y[idx_c-1])/(x[idx_c]-x[idx_c-1]));
 
            for (idx_r=1; idx_r < (sizeof(x)/sizeof(x[0])); idx_r++)
            {
                if (x[idx_r] > rccg_r)
                    break;
            }
            d_rccg_r = y[idx_r-1] + ((rccg_r - x[idx_r-1]) * (y[idx_r]-y[idx_r-1])/(x[idx_r]-x[idx_r-1]));
 
            d_rccg_g = CLIP(d_rccg_g >> shift);
            d_rccg_c = CLIP(d_rccg_c >> shift);
            d_rccg_r = CLIP(d_rccg_r >> shift);

            //b = c - (0.7*r)- g;
            bgr_b = (uint8_t) CLIP((-4.56*d_rccg_g +7.21*d_rccg_c -1.65*d_rccg_r));
            bgr_g = (uint8_t) CLIP((2.45*d_rccg_g -1.16*d_rccg_c  -0.29*d_rccg_r));
            bgr_r = (uint8_t) CLIP((-1.14*d_rccg_g -0.6*d_rccg_c  +2.75*d_rccg_r));

            s_rgb_r = CLIP (+1.08*bgr_r +0.07*bgr_g -0.15*bgr_b);
            s_rgb_g = CLIP (-0.38*bgr_r +1.43*bgr_g -0.05*bgr_b);
            s_rgb_b = CLIP (-0.02*bgr_r -0.15*bgr_g +1.17*bgr_b);

            // bgr_b = CLIP(bgr_b - 30);
            dest_buffer[idx*3+2] = bgr_b;//s_rgb_b;
            dest_buffer[idx*3+1] = bgr_g;//s_rgb_g;
            dest_buffer[idx*3+0] = bgr_r;//s_rgb_r;
 
            if(((idx +1)%width) == 0)
            {
                width_end_watch = width_end_watch?0:1;
            }
        }
    }
    return 0;
}

int32_t convert_bayer_rgb24(
        uint8_t *src_buffer, uint8_t *dest_buffer,
        int32_t width, int32_t height, uint8_t pc,
        uint32_t bpp, uint32_t shift)
{
    int32_t bayer_step    = width;
    uint32_t idx,width_end_watch;
    /*
     * pc = 0 = BGGR
     * pc = 1 = GBRG
     * pc = 2 = RGGB
     * pc = 3 = GRBG
     */
    int32_t pattern[4][3][2][2]= {
        {
            /* B offset for BGGR */
            {    {0,        -1,},
                {bayer_step,    bayer_step-1,},
            },
            /* G offset for BGGR */
            {    {1,        0,},
                {0,        bayer_step,},
            },
            /* R offset for BGGR */
            {    {bayer_step+1,    bayer_step,},
                {1,        0,},
            },
        },

        /* B offset for GBRG */
        {
            {    {1,        0,},
                {bayer_step +1,    bayer_step,},
            },
            /* G offset for GBRG */
            {    {0,        -1,},
                {1,        0,},
            },
            /* R offset for GBRG */
            {    {bayer_step,    bayer_step-1,},
                {0,        -1,},
            },
        },

        {
            /* B offset for RGGB */
            {    {bayer_step +1,    bayer_step,},
                {1,        0,},
            },
            /* G offset for RGGB */
            {    {1,        0,},
                {0,        bayer_step,},
            },
            /* R offset for RGGB */
            {    {0,        -1,},
                {bayer_step,    bayer_step-1,},
            },
        },
        /* B offset for GRBG */
        {
            {    {bayer_step,    bayer_step-1,},
                {0,        -1,},
            },
            /* G offset for GRBG */
            {    {0,        -1,},
                {1,        0,},
            },
            /* R offset for GRBG */
            {    {1,        0,},
                {bayer_step+1,    bayer_step,},
            },
        },
    };

    if (bpp == 8)
    {
        uint8_t *s_buffer = src_buffer;
        for(idx = 0, width_end_watch = 0; idx < width*(height-1); idx++)
        {
            dest_buffer[idx*3+2] = s_buffer[pattern[pc][0][width_end_watch][idx&1] +idx];
            dest_buffer[idx*3+1] = s_buffer[pattern[pc][1][width_end_watch][idx&1] +idx];
            dest_buffer[idx*3+0] = s_buffer[pattern[pc][2][width_end_watch][idx&1] +idx];

            if(((idx+1)%width) == 0)
            {
                width_end_watch = width_end_watch?0:1;
            }
        }
    }else
    {
        uint16_t *s_buffer = (uint16_t *)src_buffer;
        for(idx = 0, width_end_watch = 0; idx < width*(height-1); idx++)
        {
            dest_buffer[idx*3+2] = s_buffer[pattern[pc][0][width_end_watch][idx&1] +idx] >> shift;
            dest_buffer[idx*3+1] = s_buffer[pattern[pc][1][width_end_watch][idx&1] +idx] >> shift;
            dest_buffer[idx*3+0] = s_buffer[pattern[pc][2][width_end_watch][idx&1] +idx] >> shift;

            if(((idx+1)%width) == 0)
            {
                width_end_watch = width_end_watch?0:1;
            }
        }

    }
    return 0;
}

int32_t save_ir_asyuv(uint8_t *des_buffer, uint32_t width, uint32_t height)
{
    if (des_buffer == NULL || width == 0 || height == 0)
        return -1;
    FILE *yuvptr = fopen("sample_ir.yuv", "wb");
    if (yuvptr == NULL)
        return -1;
    const size_t dataSize = static_cast<size_t>(width) * height;
    const size_t written = fwrite(des_buffer, 1, dataSize, yuvptr);
    fclose(yuvptr);
    return written == dataSize ? 0 : -1;
}

int32_t save_buffer(uint8_t *des_buffer, uint32_t size)
{
    if (des_buffer == NULL || size == 0)
        return -1;
    FILE *raw = fopen("sample.raw", "wb");
    if (raw == NULL)
        return -1;
    const size_t written = fwrite(des_buffer, 1, size, raw);
    fclose(raw);
    return written == size ? 0 : -1;
}


int32_t save_asyuv(uint8_t *des_buffer, uint32_t width, uint32_t height)
{
    if (des_buffer == NULL || width == 0 || height == 0 || (width & 1U))
        return -1;
    const size_t pixelCount = static_cast<size_t>(width) * height;
    if (pixelCount > SIZE_MAX / 2)
        return -1;
    uint8_t *yuvbuf = static_cast<uint8_t *>(malloc(pixelCount * 2));
    if (yuvbuf == NULL)
        return -1;

    for (uint32_t row = 0; row < height; ++row)
    {
        for (uint32_t column = 0; column < width; column += 2)
        {
            const size_t first = (static_cast<size_t>(row) * width + column) * 3;
            const size_t second = first + 3;
            const int red1 = des_buffer[first];
            const int green1 = des_buffer[first + 1];
            const int blue1 = des_buffer[first + 2];
            const int red2 = des_buffer[second];
            const int green2 = des_buffer[second + 1];
            const int blue2 = des_buffer[second + 2];
            const int y1 = (299 * red1 + 587 * green1 + 114 * blue1) / 1000;
            const int y2 = (299 * red2 + 587 * green2 + 114 * blue2) / 1000;
            const int u1 = (-169 * red1 - 331 * green1 + 499 * blue1) / 1000 + 128;
            const int u2 = (-169 * red2 - 331 * green2 + 499 * blue2) / 1000 + 128;
            const int v1 = (499 * red1 - 418 * green1 - 81 * blue1) / 1000 + 128;
            const int v2 = (499 * red2 - 418 * green2 - 81 * blue2) / 1000 + 128;
            const size_t output = (static_cast<size_t>(row) * width + column) * 2;
            yuvbuf[output] = CLIP(y1);
            yuvbuf[output + 1] = CLIP((u1 + u2) / 2);
            yuvbuf[output + 2] = CLIP(y2);
            yuvbuf[output + 3] = CLIP((v1 + v2) / 2);
        }
    }

    FILE *yuvptr = fopen("sample.yuv", "wb");
    if (yuvptr == NULL)
    {
        free(yuvbuf);
        return -1;
    }
    const size_t dataSize = pixelCount * 2;
    const size_t written = fwrite(yuvbuf, 1, dataSize, yuvptr);
    fclose(yuvptr);
    free(yuvbuf);
    return written == dataSize ? 0 : -1;
}

int32_t extract_bayer10_packed_ir(uint8_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height)
{
    uint32_t count=0;
    uint32_t widthinc = (uint32_t)(width*1.25);
    uint32_t widx = 0;
    for (uint32_t hidx=1;hidx<height;hidx+=2)
    {
        for (widx=1;widx<widthinc;widx+=2)
        {
            if ((widx%5)==0)
                widx++;

            dest_buffer[count++] = src_buffer[hidx*widthinc + widx];
        }
    }
    printf("Total IR count %d x %d = %d \n", width, height, count);

    return 0;
}

int32_t extract_RGBIR16_IR8(uint8_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height)
{
    uint16_t* img;
    uint32_t widthinc = (uint32_t)(width*2);
    uint32_t count = 0;
    for (uint32_t hidx=1;hidx<height;hidx+=2)
    {
        img = (uint16_t*) &src_buffer[hidx * widthinc];
        for (uint32_t widx=1;widx<(widthinc/2);widx+=2)
        {
            dest_buffer[count++] = (img[widx])>>8;
        }
    }		
    return 0;
}

int32_t convert_bit16_bit8(uint16_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height)
{
    for (uint32_t idx=0; idx<(width*height); idx++)
    {
        dest_buffer[idx] = ((src_buffer[idx])) >>8;//htons
    }
    return 0;
}

int32_t convert_RGBIR16_bayer8(uint8_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height)
{
    /*
     * B G R G (padded bits)  B G R G (padded bits) ...
     * G IR G IR (padded bits) G IR G IR (padded bits) ...
     * R G B G (padded bits)  B G R G (padded bits) ...
     * G IR G IR (padded bits) G IR G IR (padded bits) ...
     */
    uint32_t count = 0;
    uint32_t srccount = 0;

    uint32_t widthinc = (uint32_t)(width*2);
    for (uint32_t hidx = 0; hidx < height; hidx++)
    {
        srccount = 0;
        uint32_t patternidx = hidx%4;
        uint16_t* srcptr_even;
        uint16_t* srcptr_odd;

        if ((hidx%2) == 0)
        {
            srcptr_even = (uint16_t* )&src_buffer[hidx * widthinc];
            srcptr_odd  = (uint16_t* )&src_buffer[(hidx+1) * widthinc];
        }
        else
        {
            srcptr_even = (uint16_t* )&src_buffer[(hidx-1) * widthinc];
            srcptr_odd  = (uint16_t* )&src_buffer[(hidx) * widthinc];
        }

        switch(patternidx)
        {
            case 2:
                for (uint32_t widx = 0; widx < width; widx+=4)
                {
                    dest_buffer[count+ 0] = srcptr_even[srccount + 0]>>8;
                    dest_buffer[count+ 1] = srcptr_even[srccount + 1]>>8;
                    dest_buffer[count+ 2] = srcptr_even[srccount + 0]>>8;
                    dest_buffer[count+ 3] = srcptr_even[srccount + 3]>>8;
                    srccount+=4;
                    count+=4;	
                }
                break;

            case 3:
                for (uint32_t widx = 0; widx < width; widx+=4)
                {
                    dest_buffer[count+ 0] =  srcptr_odd[srccount  +0]>>8;
                    dest_buffer[count+ 1] =  srcptr_even[srccount +2]>>8;
                    dest_buffer[count+ 2] =  srcptr_odd[srccount  +2]>>8;
                    dest_buffer[count+ 3] =  srcptr_even[srccount +2]>>8;
                    srccount+=4;
                    count+=4;		
                }
                break;

            case 0:
                for (uint32_t widx = 0; widx < width; widx+=4)
                {
                    dest_buffer[count+ 0] = srcptr_even[srccount + 2]>>8;
                    dest_buffer[count+ 1] = srcptr_even[srccount + 1]>>8;
                    dest_buffer[count+ 2] = srcptr_even[srccount + 2]>>8;
                    dest_buffer[count+ 3] = srcptr_even[srccount + 3]>>8;
                    srccount+=4;
                    count+=4;	
                }
                break;

            case 1:
                for (uint32_t widx = 0; widx < width; widx+=4)
                {
                    dest_buffer[count+ 0] =  srcptr_odd[srccount  +0]>>8;
                    dest_buffer[count+ 1] =  srcptr_even[srccount +0]>>8;
                    dest_buffer[count+ 2] =  srcptr_odd[srccount  +2]>>8;
                    dest_buffer[count+ 3] =  srcptr_even[srccount +0]>>8;
                    srccount+=4;
                    count+=4;		
                }
                break;
        }
    }

    return 0;
}

int32_t convert_bayer10_packed_rgbir(uint8_t *src_buffer, uint8_t *dest_buffer, int32_t width, int32_t height)
{
    /*
     * B G R G (padded bits)  B G R G (padded bits) ...
     * G IR G IR (padded bits) G IR G IR (padded bits) ...
     * R G B G (padded bits)  B G R G (padded bits) ...
     * G IR G IR (padded bits) G IR G IR (padded bits) ...
     */
    uint32_t count = 0;
    uint32_t srccount = 0;

    uint32_t widthinc = (uint32_t)(width*1.25);
    for (uint32_t hidx = 0; hidx < height; hidx++)
    {
        srccount = 0;
        uint32_t patternidx = hidx%4;
        uint8_t* srcptr_even;
        uint8_t* srcptr_odd;

        if ((hidx%2) == 0)
        {
            srcptr_even = &src_buffer[hidx * widthinc];
            srcptr_odd  = &src_buffer[(hidx+1) * widthinc];
        }
        else
        {
            srcptr_even = &src_buffer[(hidx-1) * widthinc];
            srcptr_odd  = &src_buffer[(hidx) * widthinc];
        }

        switch(patternidx)
        {
            case 2:
                for (uint32_t widx = 0; widx < width; widx+=4)
                {
                    dest_buffer[count+ 0] = srcptr_even[srccount + 0];
                    dest_buffer[count+ 1] = srcptr_even[srccount + 1];
                    dest_buffer[count+ 2] = srcptr_even[srccount + 0];
                    dest_buffer[count+ 3] = srcptr_even[srccount + 3];
                    srccount+=5;
                    count+=4;	
                }
                break;

            case 3:
                for (uint32_t widx = 0; widx < width; widx+=4)
                {
                    dest_buffer[count+ 0] =  srcptr_odd[srccount  +0];
                    dest_buffer[count+ 1] =  srcptr_even[srccount +2];
                    dest_buffer[count+ 2] =  srcptr_odd[srccount  +2];
                    dest_buffer[count+ 3] =  srcptr_even[srccount +2];
                    srccount+=5;
                    count+=4;		
                }
                break;

            case 0:
                for (uint32_t widx = 0; widx < width; widx+=4)
                {
                    dest_buffer[count+ 0] = srcptr_even[srccount + 2];
                    dest_buffer[count+ 1] = srcptr_even[srccount + 1];
                    dest_buffer[count+ 2] = srcptr_even[srccount + 2];
                    dest_buffer[count+ 3] = srcptr_even[srccount + 3];
                    srccount+=5;
                    count+=4;	
                }
                break;

            case 1:
                for (uint32_t widx = 0; widx < width; widx+=4)
                {
                    dest_buffer[count+ 0] =  srcptr_odd[srccount  +0];
                    dest_buffer[count+ 1] =  srcptr_even[srccount +0];
                    dest_buffer[count+ 2] =  srcptr_odd[srccount  +2];
                    dest_buffer[count+ 3] =  srcptr_even[srccount +0];
                    srccount+=5;
                    count+=4;		
                }
                break;
        }
    }

    return 0;
}

int32_t convert_bmp_565_bmp_888(int8_t *src_buffer, int8_t *des_buffer, int32_t width, int32_t height)
{
    int32_t ret_val;
    int32_t r,g,b,rg,gb,hidx,widx;
    uint32_t time;

    for(hidx = 0 ;hidx < height ;hidx++)
    {
        for(widx = 0 ;widx < width ;widx++)
        {
            gb    = src_buffer[(hidx*width*2)+widx*2+0];
            rg    = src_buffer[(hidx*width*2)+widx*2+1];
            r    = (rg & 0xF8);
            g    = ((((rg & 0x7)<<3) | ((gb & 0xE0) >>5)) << 2);
            b    = ((gb & 0x1F) << 3);

            des_buffer[(((height-1)-hidx)*width*3)+widx*3+0]    =0xFF & b;
            des_buffer[(((height-1)-hidx)*width*3)+widx*3+1]    =0xFF & g;
            des_buffer[(((height-1)-hidx)*width*3)+widx*3+2]    =0xFF & r;
        }
    }
    return 0;
}

int32_t convert_argb32_rgb(uint8_t *src_buffer, uint8_t *des_buffer, int32_t width, int32_t height)
{
    int32_t width_idx;
    int32_t height_idx;
    int32_t count = 0;

    for (height_idx = 0; height_idx < height; height_idx++)
        for (width_idx = 0;width_idx < width; width_idx++)
        {
            des_buffer[count++] = src_buffer[height_idx*(width*4) + (width_idx*4) + 1];
            des_buffer[count++] = src_buffer[height_idx*(width*4) + (width_idx*4) + 2];
            des_buffer[count++] = src_buffer[height_idx*(width*4) + (width_idx*4) + 3];
        }

    return 0;
}

int32_t convert_bayer_gen_rgb24(uint16_t *src_buffer, uint8_t *dest_buffer, int32_t sx, int32_t sy, int32_t start_with, int32_t shift)
{
    uint16_t *bayer,*fbayer;
    uint8_t *rgb;
    int32_t bayer_step = sx;
    int32_t rgbStep = 3 * sx;
    int32_t width = sx;
    int32_t height = sy;
    int32_t blue = -1;    //1;
    int32_t start_with_green = 1;

    int32_t idx, imax, iinc;

    switch(start_with)
    {
        case 0: { // BGGR
                    blue = 1;
                    start_with_green = 0;
                }break;

        case 1: { // GBRG
                    blue = -1;
                    start_with_green = 1;
                }break;

        case 2: { // RGGB
                    blue = -1;
                    start_with_green = 0;
                }break;

        case 3: {// GRBG
                    blue = 1;
                    start_with_green = 1;
                }break;
    }

    bayer = src_buffer;
    fbayer = bayer;    

    rgb = dest_buffer;

    /* add black border */
    imax = sx * sy * 3;

    for (idx = sx * (sy - 1) * 3; idx < imax; idx++) {
        rgb[idx] = 0;
    }

    iinc = (sx - 1) * 3;
    for (idx = (sx - 1) * 3; idx < imax; idx += iinc) {
        rgb[idx++] = 0;
        rgb[idx++] = 0;
        rgb[idx++] = 0;
    }

    rgb += 1;
    width -= 1;
    height -= 1;


    for (; height--; bayer += bayer_step, rgb += rgbStep) {
        //int32_t t0, t1;
        const uint16_t *bayer_end = bayer + width;

        if (start_with_green) {
            rgb[-blue] = (bayer[1] >> shift);
            rgb[0] = (bayer[bayer_step + 1] >> shift);
            rgb[blue] = (bayer[bayer_step] >> shift);
            bayer++;
            rgb += 3;
        }

        if (blue > 0) {
            for (; bayer <= bayer_end - 2; bayer += 2, rgb += 6) {
                rgb[-1] = (bayer[0] >> shift);
                rgb[0] = (bayer[1] >> shift);
                rgb[1] = (bayer[bayer_step + 1] >> shift);

                rgb[2] = (bayer[2] >> shift);
                rgb[3] = (bayer[bayer_step + 2] >> shift);
                rgb[4] = (bayer[bayer_step + 1] >> shift);
            }
        } 
        else {
            for (; bayer <= bayer_end - 2; bayer += 2, rgb += 6) {
                rgb[1] = (bayer[0] >> shift);
                rgb[0] = (bayer[1] >> shift);
                rgb[-1] = (bayer[bayer_step + 1] >> shift);

                rgb[4] = (bayer[2] >> shift);
                rgb[3] = (bayer[bayer_step + 2] >> shift);
                rgb[2] = (bayer[bayer_step + 1] >> shift);
            }
        }

        if (bayer < bayer_end) {
            rgb[-blue] = (bayer[0] >> shift);
            rgb[0] = (bayer[1] >> shift);
            rgb[blue] = (bayer[bayer_step + 1] >> shift);
            bayer++;
            rgb += 3;
        }

        bayer -= width;
        rgb -= width * 3;
        blue = -blue;
        start_with_green = !start_with_green;
    }
    return 0;
}
