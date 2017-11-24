#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <string.h>

#define WIDTH 640
#define HEIGHT 480
main(int argc, char **argv) {
	int count=0;
	unsigned char *src_buf = (char *)malloc (WIDTH*HEIGHT*2);
	unsigned char *des_buf = (char *)malloc (WIDTH*HEIGHT*1.5*2);
	int file_length, src_inc, des_inc, src_height, src_width_index, src_width, uv_count,uv_des_buf_count;
	FILE *fp;
	struct timeval t0 = {0},t1 = {0};

	fp = fopen(argv[1],"r+");
	fseek(fp, 0L, SEEK_END);
	file_length = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
	fread(src_buf,1,WIDTH*HEIGHT*2,fp);
	fclose(fp);
 	memset(des_buf, 0x80, WIDTH*HEIGHT*1.5*2);

	gettimeofday(&t0,NULL);
	yuyv_yuv420_sidebyside_color_convert(des_buf, src_buf, WIDTH, HEIGHT,0);
	yuyv_yuv420_sidebyside_color_convert(des_buf, src_buf, WIDTH, HEIGHT, WIDTH);

	gettimeofday(&t1,NULL);
	printf("convertion time %lld \n",(t1.tv_sec - t0.tv_sec)*1000000LL + t1.tv_usec - t0.tv_usec);
	fp = fopen(argv[2],"w+");
	fwrite(des_buf,1,WIDTH*HEIGHT*1.5*2,fp);
	fclose(fp);
	free(src_buf);
	free(des_buf);

	return 0;
}
