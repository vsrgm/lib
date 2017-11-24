#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char **argv) 
{
	char *input_image = argv[1];
	int in_width = atoi(argv[2]);
	int in_height = atoi(argv[3]);

	char *output_image = argv[4];
	int out_width = atoi(argv[5]);
	int out_height = atoi(argv[6]);

	char *src_buffer = malloc(in_width*in_height*2);
	char *des_buffer = malloc(out_width*out_height*2*2);
	FILE *fp;

	fp = fopen(input_image,"r+");
	fread(src_buffer,1,in_width*in_height*2,fp);
	fclose(fp);
	printf("in_width, %d  in_height, %d out_width, %d out_height %d \n", in_width, in_height, out_width, out_height);

	resize_stitch((int*) src_buffer,(int*) src_buffer, (int*) des_buffer, in_width, in_height, out_width, out_height);
	 
	fp = fopen(output_image,"w+");
	fwrite(des_buffer,1,out_width*out_height*2*2,fp);
	fclose(fp);
}
