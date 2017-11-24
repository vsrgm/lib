#include <stdio.h>
#include <stdlib.h>
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


__host__ __device__ void
resize_yuyv_yuv420_stitch( int blockid, int threadid, int* input1, int* input2, int* output, 
	int sourceWidth, int sourceHeight, int targetWidth,
	int targetHeight, int blockid_count, int threadid_count) 
{	
	int a, b, c, d, x, y, index;
	float x_ratio = ((float)(sourceWidth - 1)) / (targetWidth >> 1);
	float y_ratio = ((float)(sourceHeight - 1)) / targetHeight;
	float x_diff, y_diff, Y1, U1, Y2, V1, U2, V2;
	int i, j;
	float xy_1mx1my,xy_x1my,xy_y1mx,xy_xy;

	unsigned char *dest = (unsigned char*)output;
	unsigned short *output_y = (unsigned short *)dest;
	unsigned char *output_u =  dest + (targetWidth*targetHeight);
	unsigned char *output_v = output_u + (targetWidth*targetHeight)/4;

//  printf("Thread %d Block %d \n",threadIdx.x,blockIdx.x);
	for (i = blockid; i < targetHeight; i+=blockid_count) 
	{
		for (j = threadid; j < (targetWidth >>2); j+=threadid_count) 
		{
			x = (int)(x_ratio * j) ;
			y = (int)(y_ratio * i) ;
			x_diff = (x_ratio * j) - x ;
			y_diff = (y_ratio * i) - y ;
			index = (y * (sourceWidth>>1) + x) ;				
			a = input1[index] ;
			b = input1[index + 1] ;
			c = input1[index + (sourceWidth>>1)] ;
			d = input1[index + (sourceWidth>>1) + 1] ;

			xy_1mx1my	= (1-x_diff)*(1-y_diff);
			xy_x1my	= (x_diff)*(1-y_diff);
			xy_y1mx	= (y_diff)*(1-x_diff);
			xy_xy	= (x_diff*y_diff);
	
			// Y element
			Y1 = (a&0xff)*xy_1mx1my + (b&0xff)*xy_x1my +
				   (c&0xff)*xy_y1mx   + (d&0xff)*xy_xy;

			// U element
			U1 = ((a>>8)&0xff)*xy_1mx1my + ((b>>8)&0xff)*xy_x1my +
					((c>>8)&0xff)*xy_y1mx   + ((d>>8)&0xff)*xy_xy;

			// Y element
			Y2 = ((a>>16)&0xff)*xy_1mx1my + ((b>>16)&0xff)*xy_x1my +
				  ((c>>16)&0xff)*xy_y1mx   + ((d>>16)&0xff)*xy_xy;

			// V element
			V1 = ((a>>24)&0xff)*xy_1mx1my + ((b>>24)&0xff)*xy_x1my +
				  ((c>>24)&0xff)*xy_y1mx   + ((d>>24)&0xff)*xy_xy;

//			output [i*(targetWidth) + j] = 
//					((int)Y1) & 0x000000ff | 
//					((((int)V)   << 24)&0xff000000) |
//					((((int)Y2) << 16)&0xff0000) |
//					((((int)U)  << 8)&0xff00);
			output_y[(i*(targetWidth>>1) + j)] = (((unsigned int)Y1 & 0xFF)) | (((unsigned int)Y2 << 8) & (0xFF00));

			x = (int)(x_ratio * j) ;
			y = (int)(y_ratio * i) ;
			x_diff = (x_ratio * j) - x ;
			y_diff = (y_ratio * i) - y ;
			index = (y * (sourceWidth>>1) + x) ;				
			a = input2[index] ;
			b = input2[index + 1] ;
			c = input2[index + (sourceWidth>>1)] ;
			d = input2[index + (sourceWidth>>1) + 1] ;

			xy_1mx1my	= (1-x_diff)*(1-y_diff);
			xy_x1my	= (x_diff)*(1-y_diff);
			xy_y1mx	= (y_diff)*(1-x_diff);
			xy_xy	= (x_diff*y_diff);
	
			// Y element
			Y1 = (a&0xff)*xy_1mx1my + (b&0xff)*xy_x1my +
				   (c&0xff)*xy_y1mx   + (d&0xff)*xy_xy;

			// U element
			U2 = ((a>>8)&0xff)*xy_1mx1my + ((b>>8)&0xff)*xy_x1my +
					((c>>8)&0xff)*xy_y1mx   + ((d>>8)&0xff)*xy_xy;

			// Y element
			Y2 = ((a>>16)&0xff)*xy_1mx1my + ((b>>16)&0xff)*xy_x1my +
				  ((c>>16)&0xff)*xy_y1mx   + ((d>>16)&0xff)*xy_xy;

			// V element
			V2 = ((a>>24)&0xff)*xy_1mx1my + ((b>>24)&0xff)*xy_x1my +
				  ((c>>24)&0xff)*xy_y1mx   + ((d>>24)&0xff)*xy_xy;

//			output [(targetWidth>>1)+i*(targetWidth) + j] = 
//					((int)Y1) & 0x000000ff | 
//					((((int)V)   << 24)&0xff000000) |
//					((((int)Y2) << 16)&0xff0000) |
//					((((int)U)  << 8)&0xff00);
			output_y[(targetWidth>>2)+(i*(targetWidth>>1) + j)] = (((unsigned int)Y1 & 0xFF)) | (((unsigned int)Y2 << 8) & (0xFF00));
			if (i%2 == 0) {
				if (j >= (targetWidth >>3)) {
					output_u[(i*(targetWidth>>2) + j)] = (unsigned char)U1;
					output_v[(i*(targetWidth>>2) + j)] = (unsigned char)V1;
					output_u[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = (unsigned char)U2;
					output_v[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = (unsigned char)V2;
				}
				else {
					output_u[((i*(targetWidth>>2)) + j)] = (unsigned char)U1;
					output_v[((i*(targetWidth>>2)) + j)] = (unsigned char)V1;
					output_u[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = (unsigned char)U2;
					output_v[(targetWidth>>2)+(i*(targetWidth>>2) + j)] = (unsigned char)V2;
				}
			}
			
		}
	}
}

__host__ __device__ void
resize_yuyv_yuv420_stitch_simple( int blockid, int threadid, int* input1, int* input2, int* output, 
	int sourceWidth, int sourceHeight, int targetWidth,
	int targetHeight, int blockid_count, int threadid_count) 
{	
	int x, y, index;
	float x_ratio = ((float)(sourceWidth - 1)) / (targetWidth >> 1);
	float y_ratio = ((float)(sourceHeight - 1)) / targetHeight;
	int i, j;

	unsigned int YUYV1, YUYV2;
	unsigned char *dest = (unsigned char*)output;
	unsigned short *output_y = (unsigned short *)dest;
	unsigned char *output_u =  dest + (targetWidth*targetHeight);
	unsigned char *output_v = output_u + (targetWidth*targetHeight)/4;


//  printf("Thread %d Block %d \n",threadIdx.x,blockIdx.x);
	for (i = blockid; i < targetHeight; i+=blockid_count) 
	{
		for (j = threadid; j < (targetWidth >> 2); j+=threadid_count) 
		{
			x = (int)(x_ratio * j) ;
			y = (int)(y_ratio * i) ;
			index = (y * (sourceWidth>>1) + x) ;				
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
