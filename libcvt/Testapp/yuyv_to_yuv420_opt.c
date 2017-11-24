#include <stdio.h>
#include <stdlib.h>

main(int argc, char **argv) {
	int width = 1920;
	int height = 1080;
	unsigned char *src_buf = (char *)malloc (width*height*2);
	unsigned char *des_buf = (char *)malloc (width*height*1.5);
	int file_length;
	FILE *fp;
	int count=0;

	fp = fopen(argv[1],"r+");
	fseek(fp, 0L, SEEK_END);
	file_length = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
	fread(src_buf,1,width*height*2,fp);
	fclose(fp);
 	memset(des_buf, 0x80, width*height*1.5);

	yuyv_to_yuv420_opt(des_buf, src_buf, width, height);
	
	fp = fopen(argv[2],"w+");
	fwrite(des_buf,1,width*height*1.5,fp);
	fclose(fp);
	free(src_buf);
	free(des_buf);

	return 0;
}
