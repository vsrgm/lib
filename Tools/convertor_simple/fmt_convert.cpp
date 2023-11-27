#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BYTE_CLAMP(temp) ((temp > 255) ? 255 : ((temp < 0) ? 0 :(unsigned char)temp))

int convert_yuyv420_rgb888(unsigned char* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height, int start_with)
{
/* 
 *    start_with
 *    YUYV = 0
 *    YVYU = 1
 *    UYVY = 2
 *    VYUY = 3
*/
    int i = 0, j = 0;
    unsigned char* rgb_buffer;
    int temp;
    int yuvcount = 0;
    int rgbcount = 0;
    float u_val, v_val, y1_val, y2_val;
    unsigned int time;

    rgb_buffer = rgb888;

    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j+= 2)
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

            temp = (int)(y1_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +0] = BYTE_CLAMP(temp);

            temp = (int)(y1_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +1] = BYTE_CLAMP(temp);
                
            temp = (int)(y1_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +2] = BYTE_CLAMP(temp);
    
            temp = (int)(y2_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +3] = BYTE_CLAMP(temp);
    
            temp = (int)(y2_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +4] = BYTE_CLAMP(temp);
                
            temp = (int)(y2_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +5] = BYTE_CLAMP(temp);
        }
    }
    return 0;
}
int convert_nv12_rgb888(unsigned char* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height)
{
/* 
 *    start_with
 *    YUYV = 0
 *    YVYU = 1
 *    UYVY = 2
 *    VYUY = 3
*/
    int i = 0, j = 0;
    unsigned char* rgb_buffer;
    int temp;
    int yuvcount = 0;
    int u_count = (width * height);
    int v_count = u_count + (u_count/4);
//    int rgbcount = 0;
//    float u_val, v_val, y1_val, y2_val,y3_val, y4_val;
    unsigned int time;

    rgb_buffer = rgb888;
    unsigned char *y_channel = yuyv_buffer;
    unsigned char *u_channel = yuyv_buffer + u_count;
    unsigned char *v_channel = yuyv_buffer + v_count;

    int r,g,b;
    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'
    #define CLAMP(x) x>0xFF?0xFF:(x<0)?0:x
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int yy = y_channel[(y * width) + x];
            int uu = u_channel[(((y / 2) * (width / 2)) + (x / 2))*2];
            int vv = u_channel[(((y / 2) * (width / 2)) + (x / 2))*2 +1];

            r = 1.164 * (yy - 16) + 1.596 * (vv - 128);
            g = 1.164 * (yy - 16) - 0.813 * (vv - 128) - 0.391 * (uu - 128);
            b = 1.164 * (yy - 16) + 2.018 * (uu - 128);
            *rgb_buffer++ = CLAMP(r);
            *rgb_buffer++ = CLAMP(g);
            *rgb_buffer++ = CLAMP(b);
        }
    }
    return 0;
}

int convert_yuy420p_rgb888(unsigned char* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height)
{
/* 
 *    start_with
 *    YUYV = 0
 *    YVYU = 1
 *    UYVY = 2
 *    VYUY = 3
*/
    int i = 0, j = 0;
    unsigned char* rgb_buffer;
    int temp;
    int yuvcount = 0;
    int u_count = (width * height);
    int v_count = u_count + (u_count/4);
    int rgbcount = 0;
    float u_val, v_val, y1_val, y2_val,y3_val, y4_val;
    unsigned int time;

    rgb_buffer = rgb888;
    unsigned char *y_channel = yuyv_buffer;
    unsigned char *u_channel = yuyv_buffer + u_count;
    unsigned char *v_channel = yuyv_buffer + v_count;

    int r,g,b;
    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'
    #define CLAMP(x) x>0xFF?0xFF:(x<0)?0:x
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int yy = y_channel[(y * width) + x];
            int uu = u_channel[((y / 2) * (width / 2)) + (x / 2)];
            int vv = v_channel[((y / 2) * (width / 2)) + (x / 2)];

            r = 1.164 * (yy - 16) + 1.596 * (vv - 128);
            g = 1.164 * (yy - 16) - 0.813 * (vv - 128) - 0.391 * (uu - 128);
            b = 1.164 * (yy - 16) + 2.018 * (uu - 128);
            *rgb_buffer++ = CLAMP(r);
            *rgb_buffer++ = CLAMP(g);
            *rgb_buffer++ = CLAMP(b);
        }
    }
    return 0;
}

int convert_yuy422p_rgb888(unsigned char* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height)
{
/* 
 *    start_with
 *    YUYV = 0
 *    YVYU = 1
 *    UYVY = 2
 *    VYUY = 3
*/
    int i = 0, j = 0;
    unsigned char* rgb_buffer;
    int temp;
    int yuvcount = 0;
    int u_count = (width * height);
    int v_count = u_count + (u_count/2);
    int rgbcount = 0;
    float u_val, v_val, y1_val, y2_val;
    unsigned int time;

    rgb_buffer = rgb888;

    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j+= 2)
        {
            y1_val = (float)yuyv_buffer[yuvcount++];
            u_val = (float)yuyv_buffer[u_count++];
            y2_val = (float)yuyv_buffer[yuvcount++];
            v_val = (float)yuyv_buffer[v_count++];

            u_val = u_val - 128;
            v_val = v_val - 128;        

            temp = (int)(y1_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +0] = BYTE_CLAMP(temp);

            temp = (int)(y1_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +1] = BYTE_CLAMP(temp);
                
            temp = (int)(y1_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +2] = BYTE_CLAMP(temp);
    
            temp = (int)(y2_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +3] = BYTE_CLAMP(temp);
    
            temp = (int)(y2_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +4] = BYTE_CLAMP(temp);
                
            temp = (int)(y2_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +5] = BYTE_CLAMP(temp);
        }
    }
    return 0;
}
int convert_y8_rgb888(unsigned char* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height)
{
    int i = 0, j = 0;
    unsigned char* rgb_buffer;
    int yuvcount = 0;
    float y_val;
    unsigned int time;

    rgb_buffer = rgb888;

    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            y_val = (float)yuyv_buffer[yuvcount++];

            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +0] = y_val;
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +1] = y_val;                
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +2] = y_val;
        }
    }
    return 0;
}

int convert_y16_rgb888(unsigned short* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height)
{
    int i = 0, j = 0;
    unsigned char* rgb_buffer;
    int yuvcount = 0;
    unsigned short y_val;
    unsigned int time;

    rgb_buffer = rgb888;

    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            y_val = (unsigned short)yuyv_buffer[yuvcount++];

            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +0] = y_val>>3;
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +1] = y_val>>3;                
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +2] = y_val>>3;
        }
    }
    return 0;
}

int convert_yuyv_rgb888(unsigned char* yuyv_buffer,unsigned char* rgb888, unsigned int width, unsigned int height, int start_with)
{
/* 
 *    start_with
 *    YUYV = 0
 *    YVYU = 1
 *    UYVY = 2
 *    VYUY = 3
*/
    int i = 0, j = 0;
    unsigned char* rgb_buffer;
    int temp;
    int yuvcount = 0;
    int rgbcount = 0;
    float u_val, v_val, y1_val, y2_val;
    unsigned int time;

    rgb_buffer = rgb888;

    // memset(rgb_buffer, 0x00, (still_height * still_width * 3));
    // R = Y + 1.403V'
    // G = Y - 0.344U' - 0.714V'    
    // B = Y + 1.770U'

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j+= 2)
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

            temp = (int)(y1_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +0] = BYTE_CLAMP(temp);

            temp = (int)(y1_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +1] = BYTE_CLAMP(temp);
                
            temp = (int)(y1_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +2] = BYTE_CLAMP(temp);
    
            temp = (int)(y2_val + (1.770 * u_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +3] = BYTE_CLAMP(temp);
    
            temp = (int)(y2_val - (0.344 * u_val) - (0.714 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +4] = BYTE_CLAMP(temp);
                
            temp = (int)(y2_val + (1.403 * v_val));
            rgb_buffer[(((height-1)-i) * width * 3) + j*3 +5] = BYTE_CLAMP(temp);
        }
    }
    return 0;
}

int perform_equalize_rgb24 (unsigned char *ptr, unsigned int width, unsigned int height)
{
	unsigned char R,G,B;
	unsigned int R_count[256];
	unsigned int G_count[256];
	unsigned int B_count[256];
    memset(R_count, 0x00, sizeof(R_count));
	memset(G_count, 0x00, sizeof(G_count));
	memset(B_count, 0x00, sizeof(B_count));
	unsigned int R_sum = 0;
	unsigned int G_sum = 0;
	unsigned int B_sum = 0;
	unsigned int cdf_R_count[256];
	unsigned int cdf_G_count[256];
	unsigned int cdf_B_count[256];
	unsigned int cdf_R_min = 0;
	unsigned int cdf_G_min = 0;
	unsigned int cdf_B_min = 0;

	unsigned char hv_R[256];
	unsigned char hv_G[256];
	unsigned char hv_B[256];

	unsigned int R_total_count = width * height;
	unsigned int G_total_count = R_total_count;
	unsigned int B_total_count = R_total_count;

    for(unsigned int i=0;i<(width*height);i++) {
        R = ptr[i*3+2];
        G = ptr[i*3+1];
        B = ptr[i*3+0];

	/* perform Histogram */
		R_count[R]++;
		G_count[G]++;
		B_count[B]++;
    }
	#define CEIL(x) ((x)>255?255:((x)<0?0:(x)))
	/* compute cdf and h(v) */
	for (unsigned int cdf_idx=0; cdf_idx <256; cdf_idx++)
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
		hv_R[cdf_idx] = CEIL(((cdf_R_count[cdf_idx] - cdf_R_min) * 255) / (R_total_count - cdf_R_min));
		hv_G[cdf_idx] = CEIL(((cdf_G_count[cdf_idx] - cdf_G_min) * 255) / (G_total_count - cdf_G_min));
		hv_B[cdf_idx] = CEIL(((cdf_B_count[cdf_idx] - cdf_B_min) * 255) / (B_total_count - cdf_B_min));
	}

	/* Equalize */
	for(unsigned int i=0;i<(width*height);i++) {
		ptr[i*3+2] = (ptr[i*3+2] > 240) ? ptr[i*3+2] : hv_R[ptr[i*3+2]];
		ptr[i*3+1] = (ptr[i*3+1] > 240) ? ptr[i*3+1] : hv_G[ptr[i*3+1]];
		ptr[i*3+0] = (ptr[i*3+0] > 240) ? ptr[i*3+0] : hv_B[ptr[i*3+0]];
	}
	return 0;
}

int convert_rgb555_888(unsigned char* inbuf, unsigned char* outbuf, unsigned int width, unsigned int height, int start_with)
{     unsigned int row_cnt, pix_cnt;     
      unsigned int off1 = 0, off2 = 0;
      unsigned char  tbi1, tbi2, R5, G5, B5, R8, G8, B8;

      for (row_cnt = 0; row_cnt <= height; row_cnt++) 
      {     off1 = row_cnt * width * 2;
            off2 = row_cnt * width * 3;
            for(pix_cnt=0; pix_cnt < width; pix_cnt++)
            {    tbi1 = inbuf[off1 + (pix_cnt * 2)];
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
int convert_rgb555_888(unsigned char* rgb565,unsigned char* rgb888, unsigned int width, unsigned int height, int start_with)
{

/*
 * RGGB = 0
 * GBRG = 1
 * BGGR = 2
 * GRBG = 3
 */
    int r,g,b,rg,gb,bg,gr,i,j;

    for (i=0; i<height; i++)
    {
        for(j=0; j<width; j++)
        {
            switch (start_with)
            {
                case 0:
                {
                    rg = rgb565[(i * width*2)+j*2+0];
                    gb = rgb565[(i * width*2)+j*2+1];
                    r = (0x7C & rg) << 1;
                    g = ((rg & 0x3)<<6) | ((gb&0xE0)>>2);
                    b = (gb & 0x1F) << 2;
                }break;

                case 1:
                {
                    gb = rgb565[(i * width*2)+j*2+0];
                    rg = rgb565[(i * width*2)+j*2+1];
                    r = (0x7C & rg) << 1;
                    g = ((rg & 0x3)<<6) | ((gb&0xE0)>>2);
                    b = (gb & 0x1F) << 2;

                }break;

                case 2:
                {
                    bg = rgb565[(i * width*2)+j*2+0];
                    gr = rgb565[(i * width*2)+j*2+1];
                    b = (0x7C & bg) << 1;
                    g = ((bg & 0x3)<<6) | ((gr&0xE0)>>2);
                    r = (gr & 0x1F) << 2;

                }break;

                case 3:
                {
                    gr = rgb565[(i * width*2)+j*2+0];
                    bg = rgb565[(i * width*2)+j*2+1];
                
                }break;
            }

            rgb888[(((height-1)-i) * width * 3)+ j*3 +0] = 0xFF & b;
            rgb888[(((height-1)-i) * width * 3)+ j*3 +1] = 0xFF & g;
            rgb888[(((height-1)-i) * width * 3)+ j*3 +2] = 0xFF & r;
        }
    }
    return 0;
}
#endif

int convert_rgb565_888(unsigned char* rgb565,unsigned char* rgb888, unsigned int width, unsigned int height, int start_with)
{

/*
 * RGGB = 0
 * GBRG = 1
 * BGGR = 2
 * GRBG = 3
 */
    int r,g,b,rg,gb,bg,gr,i,j;

    for (i=0; i<height; i++)
    {
        for(j=0; j<width; j++)
        {
            switch (start_with)
            {
                case 0:
                {
                    rg = rgb565[(i * width*2)+j*2+0];
                    gb = rgb565[(i * width*2)+j*2+1];
                    r = (0xF7 & rg);
                    g = (((rg&0x7)<<3) | ((gb&0xE0)>>5)) << 2;
                    b = ((gb&0x1F) << 3);
                }break;

                case 1:
                {
                    gb = rgb565[(i * width*2)+j*2+0];
                    rg = rgb565[(i * width*2)+j*2+1];
                    r = (0xF7 & rg);
                    g = (((rg&0x7)<<3) | ((gb&0xE0)>>5)) << 2;
                    b = ((gb&0x1F) << 3);
                }break;

                case 2:
                {
                    bg = rgb565[(i * width*2)+j*2+0];
                    gr = rgb565[(i * width*2)+j*2+1];
                    r = (gr & 0x1F) << 3;
                    g = ((bg & 0x7) << 3 | (gr & 0xE0)>>5) << 2;
                    b = (0xF7 & gb);
                }break;

                case 3:
                {
                    gr = rgb565[(i * width*2)+j*2+0];
                    bg = rgb565[(i * width*2)+j*2+1];
                    b = (0xF7 & bg);
                    g = (((bg&0x7)<<3) | ((gr&0xE0)>>5)) << 2;
                    r = ((gr&0x1F) << 3);

                }break;
            }

            rgb888[(((height-1)-i) * width * 3)+ j*3 +0] = 0xFF & b;
            rgb888[(((height-1)-i) * width * 3)+ j*3 +1] = 0xFF & g;
            rgb888[(((height-1)-i) * width * 3)+ j*3 +2] = 0xFF & r;
        }
    }
    return 0;
}

int convert_bayer12_bayer8(unsigned char *src_buffer, unsigned char *dest_buffer, int width, int height)
{
    int index =0;
    int hindex, windex;
    for (hindex = 0; hindex < height; hindex++) {
        for (windex = 0; windex < width/2; windex++) {
            dest_buffer[index++] = src_buffer[(int)(hindex*width*1.5)+ (windex * 3) +0];
            dest_buffer[index] = (src_buffer[(int)(hindex*width*1.5)+ (windex * 3) +1] & 0XF)<<4;
            dest_buffer[index++] |= (src_buffer[(int)(hindex*width*1.5)+ (windex * 3) +1] & 0XF0)>>4;
        }
    }
    return 0;
}

int convert_bayer8_rgb24(unsigned char *src_buffer, unsigned char *dest_buffer, int width, int height, unsigned char pc)
{
    int bayer_step    = width;
    unsigned int i,width_end_watch;
/*
 * pc = 0 = BGGR
 * pc = 1 = GBRG
 * pc = 2 = RGGB
 * pc = 3 = GRBG
 */
    int pattern[4][3][2][2]= {
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
            
    for(i=0,width_end_watch=0;i<width*(height-1);i++) {
        dest_buffer[i*3+2] = src_buffer[pattern[pc][0][width_end_watch][i&1] +i];
        dest_buffer[i*3+1] = src_buffer[pattern[pc][1][width_end_watch][i&1] +i];
        dest_buffer[i*3+0] = src_buffer[pattern[pc][2][width_end_watch][i&1] +i];

        if((i%width) == 0) {
            width_end_watch = width_end_watch?0:1;
        }
    }
    return 0;
}

int save_ir_asyuv(unsigned char *des_buffer, unsigned int width, unsigned int height)
{
	/* Convert into YUV file and SAVE it */
	FILE *yuvptr = fopen("sample_ir.yuv", "wb");
	fwrite(des_buffer,1, width * height, yuvptr);
	fclose(yuvptr);
}

int save_buffer(unsigned char *des_buffer, unsigned int size)
{
	/* Convert into YUV file and SAVE it */
	FILE *raw = fopen("sample.raw", "wb");
	fwrite(des_buffer,1, size, raw);
    fclose(raw);
}


int save_asyuv(unsigned char *des_buffer, unsigned int width, unsigned int height)
{
	/* Convert into YUV file and SAVE it */
	FILE *yuvptr = fopen("sample.yuv", "wb");
	unsigned char *yuvbuf = (unsigned char *)calloc(width * height*2, 1);

	unsigned int widthinc, heightinc;
	for (heightinc =0; heightinc < height; heightinc++)
	{
		for(widthinc = 0; widthinc < width; widthinc+=2)
		{
			unsigned char y1,u1,v1;
			unsigned char y2,u2,v2;
			unsigned char R1,G1,B1;
			unsigned char R2,G2,B2;
			R1 = des_buffer[(heightinc*width*3)+ widthinc*3 + 0];
			G1 = des_buffer[(heightinc*width*3)+ widthinc*3 + 1];
			B1 = des_buffer[(heightinc*width*3)+ widthinc*3 + 2];

			R2 = des_buffer[(heightinc*width*3)+ widthinc*3 + 3];
			G2 = des_buffer[(heightinc*width*3)+ widthinc*3 + 4];
			B2 = des_buffer[(heightinc*width*3)+ widthinc*3 + 5];

#define CLAMP(x) (x>255)?255:(x<0?0:x);
			y1 = CLAMP((299*R1 +587*G1 +114*B1)/1000);
			u1 = CLAMP(((-169*R1 -331*G1 +499*B1)/1000)+128);
			v1 = CLAMP(((499*R1 -418*G1 -81*B1)/1000)+128);

			y2 = CLAMP((299*R2 +587*G2 +114*B2)/1000);
			u1 = CLAMP(((-169*R2 -331*G2 +499*B2)/1000)+128);
			v1 = CLAMP(((499*R2 -418*G2 -81*B2)/1000)+128);

			yuvbuf[(heightinc*width*2) + widthinc*2 +0] = y1;
			yuvbuf[(heightinc*width*2) + widthinc*2 +1] = u1;
			yuvbuf[(heightinc*width*2) + widthinc*2 +2] = y2;
			yuvbuf[(heightinc*width*2) + widthinc*2 +3] = v1;
		}
	}
	fwrite(yuvbuf,1, width * height*2, yuvptr);
	fclose(yuvptr);
	free(yuvbuf);
}

int extract_bayer10_packed_ir(unsigned char *src_buffer, unsigned char *dest_buffer, int width, int height)
{
	unsigned int count=0;
	unsigned int widthinc = (unsigned int)(width*1.25);
	unsigned int widx = 0;
	for (unsigned int hidx=1;hidx<height;hidx+=2)
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

int extract_RGBIR16_IR8(unsigned char *src_buffer, unsigned char *dest_buffer, int width, int height)
{
	unsigned short* img;
	unsigned int widthinc = (unsigned int)(width*2);
	unsigned int count = 0;
	for (unsigned int hidx=1;hidx<height;hidx+=2)
	{
		img = (unsigned short*) &src_buffer[hidx * widthinc];
		for (unsigned int widx=1;widx<(widthinc/2);widx+=2)
		{
			dest_buffer[count++] = img[widx]>>8;
			printf("count = %d \n",count);
		}
	}		
	return 0;
}

int convert_RGBIR16_bayer8(unsigned char *src_buffer, unsigned char *dest_buffer, int width, int height)
{
	/*
	 * B G R G (padded bits)  B G R G (padded bits) ...
	 * G IR G IR (padded bits) G IR G IR (padded bits) ...
	 * R G B G (padded bits)  B G R G (padded bits) ...
	 * G IR G IR (padded bits) G IR G IR (padded bits) ...
	 */
	unsigned int count = 0;
	unsigned int srccount = 0;

	unsigned int widthinc = (unsigned int)(width*2);
	for (unsigned int hidx = 0; hidx < height; hidx++)
	{
		srccount = 0;
		unsigned int patternidx = hidx%4;
		unsigned short* srcptr_even;
		unsigned short* srcptr_odd;

		if ((hidx%2) == 0)
		{
			srcptr_even = (unsigned short* )&src_buffer[hidx * widthinc];
			srcptr_odd  = (unsigned short* )&src_buffer[(hidx+1) * widthinc];
		}
		else
		{
			srcptr_even = (unsigned short* )&src_buffer[(hidx-1) * widthinc];
			srcptr_odd  = (unsigned short* )&src_buffer[(hidx) * widthinc];
		}

		switch(patternidx)
		{
			case 2:
			for (unsigned int widx = 0; widx < width; widx+=4)
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
			for (unsigned int widx = 0; widx < width; widx+=4)
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
			for (unsigned int widx = 0; widx < width; widx+=4)
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
			for (unsigned int widx = 0; widx < width; widx+=4)
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
int convert_bayer10_packed_rgbir(unsigned char *src_buffer, unsigned char *dest_buffer, int width, int height)
{
	/*
	 * B G R G (padded bits)  B G R G (padded bits) ...
	 * G IR G IR (padded bits) G IR G IR (padded bits) ...
	 * R G B G (padded bits)  B G R G (padded bits) ...
	 * G IR G IR (padded bits) G IR G IR (padded bits) ...
	 */
	unsigned int count = 0;
	unsigned int srccount = 0;

	unsigned int widthinc = (unsigned int)(width*1.25);
	for (unsigned int hidx = 0; hidx < height; hidx++)
	{
		srccount = 0;
		unsigned int patternidx = hidx%4;
		unsigned char* srcptr_even;
		unsigned char* srcptr_odd;

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
			for (unsigned int widx = 0; widx < width; widx+=4)
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
			for (unsigned int widx = 0; widx < width; widx+=4)
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
			for (unsigned int widx = 0; widx < width; widx+=4)
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
			for (unsigned int widx = 0; widx < width; widx+=4)
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

int convert_bmp_565_bmp_888(char *src_buffer, char *des_buffer, int width, int height)
{
    int ret_val;
    int r,g,b,rg,gb,i,j;
    unsigned int time;

    for(i = 0 ;i < height ;i++)
    {
        for(j = 0 ;j < width ;j++)
        {
            gb    = src_buffer[(i*width*2)+j*2+0];
            rg    = src_buffer[(i*width*2)+j*2+1];
            r    = (rg & 0xF8);
            g    = ((((rg & 0x7)<<3) | ((gb & 0xE0) >>5)) << 2);
            b    = ((gb & 0x1F) << 3);
            
            des_buffer[(((height-1)-i)*width*3)+j*3+0]    =0xFF & b;
            des_buffer[(((height-1)-i)*width*3)+j*3+1]    =0xFF & g;
            des_buffer[(((height-1)-i)*width*3)+j*3+2]    =0xFF & r;
        }
    }
    return 0;
}

int convert_argb32_rgb(unsigned char *src_buffer, unsigned char *des_buffer, int width, int height)
{
    int width_idx;
    int height_idx;
    int count = 0;

    for (height_idx = 0; height_idx < height; height_idx++)
        for (width_idx = 0;width_idx < width; width_idx++)
        {
            des_buffer[count++] = src_buffer[height_idx*(width*4) + (width_idx*4) + 1];
            des_buffer[count++] = src_buffer[height_idx*(width*4) + (width_idx*4) + 2];
            des_buffer[count++] = src_buffer[height_idx*(width*4) + (width_idx*4) + 3];
        }

    return 0;
}
int convert_bayer_gen_rgb24(unsigned short *src_buffer, unsigned char *dest_buffer, int sx, int sy, int start_with, int shift)
{
    unsigned short *bayer,*fbayer;
    unsigned char *rgb;
        int bayer_step = sx;
        int rgbStep = 3 * sx;
        int width = sx;
        int height = sy;
        int blue = -1;    //1;
        int start_with_green = 1;

    int i, imax, iinc;

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

    for (i = sx * (sy - 1) * 3; i < imax; i++) {
        rgb[i] = 0;
        }

        iinc = (sx - 1) * 3;
        for (i = (sx - 1) * 3; i < imax; i += iinc) {
        rgb[i++] = 0;
        rgb[i++] = 0;
        rgb[i++] = 0;
        }

        rgb += 1;
        width -= 1;
        height -= 1;


        for (; height--; bayer += bayer_step, rgb += rgbStep) {
        //int t0, t1;
        const unsigned short *bayer_end = bayer + width;

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

