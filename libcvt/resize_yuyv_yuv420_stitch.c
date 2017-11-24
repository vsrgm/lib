#include <stdio.h>
#include <stdlib.h>

void resize_yuyv_yuv420_stitch_fast(int* input1, int* input2, int* output, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight) 
{	
	int x, y, index;
	float x_ratio = ((float)(sourceWidth - 1)) / (targetWidth >>1);
	float y_ratio = ((float)(sourceHeight - 1)) / targetHeight;
	int i, j;
	unsigned int YUYV1, YUYV2;
	unsigned char *dest = (unsigned char*)output;
	unsigned short *output_y = dest;
	unsigned char *output_u =  dest + (targetWidth*targetHeight);
	unsigned char *output_v = output_u + (targetWidth*targetHeight)/4;
	for (i = 0; i < targetHeight; i++) 
	{
		for (j = 0; j < (targetWidth >>2); j++) 
		{
			x = (int)(x_ratio * j) ;
			y = (int)(y_ratio * i) ;
			index = (y * sourceWidth/2 + x);				
//			output [(i*(targetWidth>>1) + j)] = input1[index];
//			output [(targetWidth>>2)+(i*(targetWidth>>1) + j)] = input2[index];	
			YUYV1 = input1[index];
			output_y[(i*(targetWidth>>1) + j)] = ((YUYV1 & 0xFF)) | ((YUYV1 >> 8) & (0xFF00));

			YUYV2 = input2[index];
			output_y[(targetWidth>>2)+(i*(targetWidth>>1) + j)] = ((YUYV2 & 0xFF)) | ((YUYV2 >> 8) & (0xFF00));

			if (i%2 == 0) {
				if (j >= (targetWidth >>3)) {
					output_u[(i*(targetWidth>>2) + j)] = 0xFF & (YUYV1>>8);
					output_v[(i*(targetWidth>>2) + j)] = 0xFF & (YUYV1>>24);
					output_u[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = 0xFF & (YUYV2>>8);
					output_v[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = 0xFF & (YUYV2>>24);
				}
				else {
					output_u[((i*(targetWidth>>2)) + j)] = 0xFF & (YUYV1>>8);
					output_v[((i*(targetWidth>>2)) + j)] = 0xFF & (YUYV1>>24);
					output_u[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = 0xFF & (YUYV2>>8);
					output_v[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = 0xFF & (YUYV2>>24);
				}
			}
		}
	}
}

void resize_yuyv_yuv420_stitch(int* input1, int* input2, int* output, int sourceWidth, int sourceHeight, int targetWidth, int targetHeight) 
{	
	int a, b, c, d, x, y, index;
	float x_ratio = ((float)(sourceWidth - 1)) / (targetWidth>>1);
	float y_ratio = ((float)(sourceHeight - 1)) / targetHeight;
	float x_diff, y_diff, Y1, U1, Y2, V1, U2, V2;
	int i, j;

	unsigned char *dest = (unsigned char*)output;
	unsigned short *output_y = dest;
	unsigned char *output_u =  dest + (targetWidth*targetHeight);
	unsigned char *output_v = output_u + (targetWidth*targetHeight)/4;

	for (i = 0; i < targetHeight; i++) 
	{
		for (j = 0; j < (targetWidth >> 2); j++) 
		{
			x = (int)(x_ratio * j) ;
			y = (int)(y_ratio * i) ;
			x_diff = (x_ratio * j) - x ;
			y_diff = (y_ratio * i) - y ;
			index = (y * sourceWidth/2 + x) ;				
			a = input1[index] ;
			b = input1[index + 1] ;
			c = input1[index + sourceWidth/2] ;
			d = input1[index + sourceWidth/2 + 1] ;

			// Y element
			Y1 = (a&0xff)*(1-x_diff)*(1-y_diff) + (b&0xff)*(x_diff)*(1-y_diff) +
				   (c&0xff)*(y_diff)*(1-x_diff)   + (d&0xff)*(x_diff*y_diff);

			// U element
			U1 = ((a>>8)&0xff)*(1-x_diff)*(1-y_diff) + ((b>>8)&0xff)*(x_diff)*(1-y_diff) +
					((c>>8)&0xff)*(y_diff)*(1-x_diff)   + ((d>>8)&0xff)*(x_diff*y_diff);

			// Y element
			Y2 = ((a>>16)&0xff)*(1-x_diff)*(1-y_diff) + ((b>>16)&0xff)*(x_diff)*(1-y_diff) +
				  ((c>>16)&0xff)*(y_diff)*(1-x_diff)   + ((d>>16)&0xff)*(x_diff*y_diff);

			// V element
			V1 = ((a>>24)&0xff)*(1-x_diff)*(1-y_diff) + ((b>>24)&0xff)*(x_diff)*(1-y_diff) +
				  ((c>>24)&0xff)*(y_diff)*(1-x_diff)   + ((d>>24)&0xff)*(x_diff*y_diff);

//			output [(i*(targetWidth>>1) + j)] = 
//					(((((int)V)  << 24)	& 0xff000000) |
//					((((int)Y2) << 16)	& 0x00ff0000) |
//					((((int)U)  << 8) 	& 0x0000ff00) |
//					((int)Y1)		& 0x000000ff);
			output_y[(i*(targetWidth>>1) + j)] = (((unsigned int)Y1 & 0xFF)) | (((unsigned int)Y2 << 8) & (0xFF00));
			
			x = (int)(x_ratio * j) ;
			y = (int)(y_ratio * i) ;
			x_diff = (x_ratio * j) - x ;
			y_diff = (y_ratio * i) - y ;
			index = (y * sourceWidth/2 + x) ;				
			a = input2[index] ;
			b = input2[index + 1] ;
			c = input2[index + sourceWidth/2] ;
			d = input2[index + sourceWidth/2 + 1] ;

			// Y element
			Y1 = (a&0xff)*(1-x_diff)*(1-y_diff) + (b&0xff)*(x_diff)*(1-y_diff) +
				   (c&0xff)*(y_diff)*(1-x_diff)   + (d&0xff)*(x_diff*y_diff);

			// U element
			U2 = ((a>>8)&0xff)*(1-x_diff)*(1-y_diff) + ((b>>8)&0xff)*(x_diff)*(1-y_diff) +
					((c>>8)&0xff)*(y_diff)*(1-x_diff)   + ((d>>8)&0xff)*(x_diff*y_diff);

			// Y element
			Y2 = ((a>>16)&0xff)*(1-x_diff)*(1-y_diff) + ((b>>16)&0xff)*(x_diff)*(1-y_diff) +
				  ((c>>16)&0xff)*(y_diff)*(1-x_diff)   + ((d>>16)&0xff)*(x_diff*y_diff);

			// V element
			V2 = ((a>>24)&0xff)*(1-x_diff)*(1-y_diff) + ((b>>24)&0xff)*(x_diff)*(1-y_diff) +
				  ((c>>24)&0xff)*(y_diff)*(1-x_diff)   + ((d>>24)&0xff)*(x_diff*y_diff);

//			output [(targetWidth>>2)+(i*(targetWidth>>1) + j)] = 
//					(((((int)V)  << 24)	& 0xff000000) |
//					((((int)Y2) << 16)	& 0x00ff0000) |
//					((((int)U)  << 8) 	& 0x0000ff00) |
//					((int)Y1)		& 0x000000ff);
			output_y[(targetWidth>>2)+(i*(targetWidth>>1) + j)] = (((unsigned int)Y1 & 0xFF)) | (((unsigned int)Y2 << 8) & (0xFF00));
			
			if (i%2 == 0) {
				if (j >= (targetWidth >>3)) {
					output_u[(i*(targetWidth>>2) + j)] = (unsigned char)U1;//0xFF & (YUYV1>>8);
					output_v[(i*(targetWidth>>2) + j)] = (unsigned char)V1;//0xFF & (YUYV1>>24);
					output_u[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = (unsigned char)U2;//0xFF & (YUYV2>>8);
					output_v[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = (unsigned char)V2;//0xFF & (YUYV2>>24);
				}
				else {
					output_u[((i*(targetWidth>>2)) + j)] = (unsigned char)U1;//0xFF & (YUYV1>>8);
					output_v[((i*(targetWidth>>2)) + j)] = (unsigned char)V1;//0xFF & (YUYV1>>24);
					output_u[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = (unsigned char)U2;//0xFF & (YUYV2>>8);
					output_v[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = (unsigned char)V2;//0xFF & (YUYV2>>24);
				}
			}
		}
	}
}
