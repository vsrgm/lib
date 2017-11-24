#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "../resize.h"

__global__ void
resize_kernel(int* input1, int* input2, int* output, 
	int sourceWidth, int sourceHeight, int targetWidth,
	int targetHeight, int blockid_count, int threadid_count)
{
	resize_yuyv_yuv420_stitch(blockIdx.x, threadIdx.x, input1, input2, output, 
		sourceWidth, sourceHeight, targetWidth,
		targetHeight, blockid_count, threadid_count);
	return;
}

int
main(int argc, char **argv) 
{
	int count=100;
	static struct timeval cur_tv,prev_tv;

	char *input1_image = argv[1];
	char *input2_image = argv[2];
	int in_width = atoi(argv[3]);
	int in_height = atoi(argv[4]);

	char *output_image = argv[5];
	int out_width = atoi(argv[6]);
	int out_height = atoi(argv[7]);

	char *src1_buffer;
	char *src2_buffer;
	char *des_buffer;

	char *src1_buffer_cu;
	char *src2_buffer_cu;
	char *des_buffer_cu;

	FILE *fp;

#if 0
	checkCudaErrors(cudaHostAlloc((void **)&src1_buffer, in_width*in_height*2, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc((void **)&src2_buffer, in_width*in_height*2, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc((void **)&des_buffer, out_width*out_height*1.5*2, cudaHostAllocMapped));

	checkCudaErrors(cudaHostGetDevicePointer((void **)&src1_buffer_cu, (void *)src1_buffer, 0));
	checkCudaErrors(cudaHostGetDevicePointer((void **)&src2_buffer_cu, (void *)src2_buffer, 0));
	checkCudaErrors(cudaHostGetDevicePointer((void **)&des_buffer_cu, (void *)des_buffer, 0));
#else
//	posix_memalign((void **)&src1_buffer,getpagesize(),in_width*in_height*2);
//	posix_memalign((void **)&src2_buffer,getpagesize(),in_width*in_height*2);
//	posix_memalign((void **)&des_buffer,getpagesize(),out_width*out_height*1.5*2);
	src1_buffer = (char *)malloc(in_width*in_height*2);
	src2_buffer = (char *)malloc(in_width*in_height*2);
	des_buffer = (char *)malloc( out_width*out_height*1.5*2);
	cudaMalloc(&src1_buffer_cu, in_width*in_height*2);
	cudaMalloc(&src2_buffer_cu, in_width*in_height*2);
	cudaMalloc(&des_buffer_cu,  out_width*out_height*1.5*2);
#endif
	fp = fopen(input1_image,"r+");
	fread(src1_buffer,1,in_width*in_height*2,fp);
	fclose(fp);

	fp = fopen(input2_image,"r+");
	fread(src2_buffer,1,in_width*in_height*2,fp);
	fclose(fp);


	printf("in_width, %d  in_height, %d out_width, %d out_height %d \n", in_width, in_height, out_width, out_height);

while (count--) {
	gettimeofday(&prev_tv, NULL);
	cudaMemcpy(src1_buffer_cu,src1_buffer,in_width*in_height*2,cudaMemcpyHostToDevice);
	gettimeofday(&cur_tv, NULL);

	printf("src1       Time prev_tv %llu.%06llu cur_tv %llu.%06llu = %llu\n",(unsigned long long int)prev_tv.tv_sec,
								(unsigned long long int)prev_tv.tv_usec,
								(unsigned long long int)cur_tv.tv_sec,
								(unsigned long long int)cur_tv.tv_usec,
								(unsigned long long int)((cur_tv.tv_sec-prev_tv.tv_sec)*1000000)
											+cur_tv.tv_usec-prev_tv.tv_usec);
	gettimeofday(&prev_tv, NULL);
	cudaMemcpy(src2_buffer_cu,src2_buffer,in_width*in_height*2,cudaMemcpyHostToDevice);
	gettimeofday(&cur_tv, NULL);

	printf("src2       Time prev_tv %llu.%06llu cur_tv %llu.%06llu = %llu\n",(unsigned long long int)prev_tv.tv_sec,
								(unsigned long long int)prev_tv.tv_usec,
								(unsigned long long int)cur_tv.tv_sec,
								(unsigned long long int)cur_tv.tv_usec,
								(unsigned long long int)((cur_tv.tv_sec-prev_tv.tv_sec)*1000000)
											+cur_tv.tv_usec-prev_tv.tv_usec);
	gettimeofday(&prev_tv, NULL);

	resize_kernel<<<192, 192>>>(	(int*) src1_buffer_cu, (int*) src2_buffer_cu, 
					(int*) des_buffer_cu, in_width, in_height,
					out_width, out_height, 192, 192);
	checkCudaErrors(cudaDeviceSynchronize());
	gettimeofday(&cur_tv, NULL);

	printf("resize     Time prev_tv %llu.%06llu cur_tv %llu.%06llu = %llu\n",(unsigned long long int)prev_tv.tv_sec,
								(unsigned long long int)prev_tv.tv_usec,
								(unsigned long long int)cur_tv.tv_sec,
								(unsigned long long int)cur_tv.tv_usec,
								(unsigned long long int)((cur_tv.tv_sec-prev_tv.tv_sec)*1000000)
											+cur_tv.tv_usec-prev_tv.tv_usec);
	gettimeofday(&prev_tv, NULL);
	cudaMemcpy(des_buffer,des_buffer_cu,out_width*out_height*1.5*2,cudaMemcpyDeviceToHost);
	gettimeofday(&cur_tv, NULL);

	printf("des_buffer Time prev_tv %llu.%06llu cur_tv %llu.%06llu = %llu\n",(unsigned long long int)prev_tv.tv_sec,
								(unsigned long long int)prev_tv.tv_usec,
								(unsigned long long int)cur_tv.tv_sec,
								(unsigned long long int)cur_tv.tv_usec,
								(unsigned long long int)((cur_tv.tv_sec-prev_tv.tv_sec)*1000000)
											+cur_tv.tv_usec-prev_tv.tv_usec);
}
	fp = fopen(output_image,"w+");
	fwrite(des_buffer,1,out_width*out_height*1.5*2,fp);
	fclose(fp);

//	checkCudaErrors(cudaFreeHost(src1_buffer));
//	checkCudaErrors(cudaFreeHost(src2_buffer));
//       checkCudaErrors(cudaFreeHost(des_buffer));
}

