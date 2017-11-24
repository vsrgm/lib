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
//	int rgbcount = 0;
//	float u_val, v_val, y1_val, y2_val,y3_val, y4_val;
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

int convert_bayer8_rgb24(unsigned char *src_buffer, unsigned char *dest_buffer, int width, int height, unsigned char pc)
{
	int bayer_step	= width;
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
			{	{0,		-1,},
				{bayer_step,	bayer_step-1,},
			},
/* G offset for BGGR */
			{	{1,		0,},
				{0,		bayer_step,},
			},
/* R offset for BGGR */
			{	{bayer_step+1,	bayer_step,},
				{1,		0,},
			},
		},

/* B offset for GBRG */
		{
			{	{1,		0,},
				{bayer_step +1,	bayer_step,},
			},
/* G offset for GBRG */
			{	{0,		-1,},
				{1,		0,},
			},
/* R offset for GBRG */
			{	{bayer_step,	bayer_step-1,},
				{0,		-1,},
			},
		},

		{
/* B offset for RGGB */
			{	{bayer_step +1,	bayer_step,},
				{1,		0,},
			},
/* G offset for RGGB */
			{	{1,		0,},
				{0,		bayer_step,},
			},
/* R offset for RGGB */
			{	{0,		-1,},
				{bayer_step,	bayer_step-1,},
			},
		},
/* B offset for GRBG */
		{
			{	{bayer_step,	bayer_step-1,},
				{0,		-1,},
			},
/* G offset for GRBG */
			{	{0,		-1,},
				{1,		0,},
			},
/* R offset for GRBG */
			{	{1,		0,},
				{bayer_step+1,	bayer_step,},
			},
		},
	};
			
	for(i=0,width_end_watch=0;i<width*(height-1);i++) {
		dest_buffer[i*3+2] = src_buffer[pattern[pc][0][width_end_watch][i%2] +i];
		dest_buffer[i*3+1] = src_buffer[pattern[pc][1][width_end_watch][i%2] +i];
		dest_buffer[i*3+0] = src_buffer[pattern[pc][2][width_end_watch][i%2] +i];

		if((i%width) == 0) {
			width_end_watch = width_end_watch?0:1;
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
			gb	= src_buffer[(i*width*2)+j*2+0];
			rg	= src_buffer[(i*width*2)+j*2+1];
			r	= (rg & 0xF8);
			g	= ((((rg & 0x7)<<3) | ((gb & 0xE0) >>5)) << 2);
			b	= ((gb & 0x1F) << 3);
			
			des_buffer[(((height-1)-i)*width*3)+j*3+0]	=0xFF & b;
			des_buffer[(((height-1)-i)*width*3)+j*3+1]	=0xFF & g;
			des_buffer[(((height-1)-i)*width*3)+j*3+2]	=0xFF & r;
		}
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
    	int blue = -1;	//1;
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

