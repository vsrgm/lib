#include <stdio.h>
#include <stdlib.h>
// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include "../yuv_yuv420.h"
__global__ void
color_convert_kernel(unsigned char *output, unsigned char *input, 
	int sourceWidth, int sourceHeight, int blockid_count, int threadid_count)
{
	yuyv_yuv420_color_convert(blockIdx.x, threadIdx.x, output, input, 
		sourceWidth, sourceHeight, blockid_count, threadid_count);
	return;
}

int
main(int argc, char **argv) 
{
	int count=1;
	static struct timeval cur_tv,prev_tv;

	char *input_image = argv[1];
	int in_width = atoi(argv[2]);
	int in_height = atoi(argv[3]);

	char *output_image = argv[4];
	int out_width = atoi(argv[5]);
	int out_height = atoi(argv[6]);

	char *src_buffer;
	char *des_buffer;

	char *src_buffer_cu;
	char *des_buffer_cu;

	FILE *fp;

	checkCudaErrors(cudaHostAlloc((void **)&src_buffer, in_width*in_height*2, cudaHostAllocMapped));
	checkCudaErrors(cudaHostAlloc((void **)&des_buffer, out_width*out_height*1.5, cudaHostAllocMapped));

	checkCudaErrors(cudaHostGetDevicePointer((void **)&src_buffer_cu, (void *)src_buffer, 0));
	checkCudaErrors(cudaHostGetDevicePointer((void **)&des_buffer_cu, (void *)des_buffer, 0));

	fp = fopen(input_image,"r+");
	fread(src_buffer,1,in_width*in_height*2,fp);
	fclose(fp);

	printf("in_width, %d  in_height, %d out_width, %d out_height %d \n", in_width, in_height, out_width, out_height);

while (count--) {
	gettimeofday(&prev_tv, NULL);
	color_convert_kernel<<<192, 192>>>( (unsigned char *)des_buffer_cu, (unsigned char *) src_buffer_cu, in_width, in_height, 192, 192);
	checkCudaErrors(cudaDeviceSynchronize());
	gettimeofday(&cur_tv, NULL);
	printf("Time prev_tv %lu.%06lu cur_tv %lu.%06lu = %lu\n",(unsigned long int)prev_tv.tv_sec,
								(unsigned long int)prev_tv.tv_usec,
								(unsigned long int)cur_tv.tv_sec,
								(unsigned long int)cur_tv.tv_usec,
								(unsigned long int)(((cur_tv.tv_sec-prev_tv.tv_sec)*1000000)
								+cur_tv.tv_usec-prev_tv.tv_usec));
}
	fp = fopen(output_image,"w+");
	fwrite(des_buffer,1,out_width*out_height*1.5,fp);
	fclose(fp);

	checkCudaErrors(cudaFreeHost(src_buffer));
        checkCudaErrors(cudaFreeHost(des_buffer));
}

