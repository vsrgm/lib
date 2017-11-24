#ifndef __RESIZE_H
#define __RESIZE_H
__host__ __device__ void
resize( int blockid, int threadid, int* input, int* output, 
	int sourceWidth, int sourceHeight, int targetWidth,
	int targetHeight, int blockid_count, int threadid_count);

__host__ __device__ void
resize_stitch( int blockid, int threadid, int* input1, int* input2, int* output, 
	int sourceWidth, int sourceHeight, int targetWidth,
	int targetHeight, int blockid_count, int threadid_count);

__host__ __device__ void
resize_stitch_simple( int blockid, int threadid, int* input1, int* input2, int* output, 
	int sourceWidth, int sourceHeight, int targetWidth,
	int targetHeight, int blockid_count, int threadid_count);

__host__ __device__ void
resize_yuyv_yuv420_stitch( int blockid, int threadid, int* input1, int* input2, int* output, 
	int sourceWidth, int sourceHeight, int targetWidth,
	int targetHeight, int blockid_count, int threadid_count);

__host__ __device__ void
resize_yuyv_yuv420_stitch_simple( int blockid, int threadid, int* input1, int* input2, int* output, 
	int sourceWidth, int sourceHeight, int targetWidth,
	int targetHeight, int blockid_count, int threadid_count);


#endif

