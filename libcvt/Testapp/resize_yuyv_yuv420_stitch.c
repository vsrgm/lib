#include <stdio.h>
#include <stdlib.h>

int resize_yuyv_yuv420_stitch_fast(int* src1_buffer,int* src2_buffer, int* des_buffer,int  in_width,int in_height,int out_width,int out_height);

int
main(int argc, char **argv) 
{
	char *input1_image = argv[1];
	char *input2_image = argv[2];
	int in_width = atoi(argv[3]);
	int in_height = atoi(argv[4]);

	char *output_image = argv[5];
	int out_width = atoi(argv[6]);
	int out_height = atoi(argv[7]);

	char *src1_buffer = calloc(in_width*in_height*2,1);
	char *src2_buffer = calloc(in_width*in_height*2,1);
	char *des_buffer = calloc(out_width*out_height*2,1);
	FILE *fp;

	fp = fopen(input1_image,"r+");
	fread(src1_buffer,1,in_width*in_height*2,fp);
	fclose(fp);

	fp = fopen(input2_image,"r+");
	fread(src2_buffer,1,in_width*in_height*2,fp);
	fclose(fp);

	printf("in_width, %d  in_height, %d out_width, %d out_height %d \n", in_width, in_height, out_width, out_height);
	resize_yuyv_yuv420_stitch((int*) src1_buffer,(int*) src2_buffer, (int*) des_buffer, in_width, in_height, out_width, out_height);
	 
	fp = fopen(output_image,"w+");
	fwrite(des_buffer,1,out_width*out_height*2,fp);
	fclose(fp);
}
