#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <string.h>

main(int argc, char **argv) {
	int count=0;
	unsigned char *src_buf = (char *)malloc (3840*2160*2);
	unsigned char *des_buf = (char *)malloc (3840*2160*1.5);
	unsigned char *u_des_buf = des_buf + (3840*2160);
	unsigned char *v_des_buf = des_buf + (3840*2160) + (3840*2160)/4;
	unsigned char *input_buf = src_buf;
	unsigned char *res_buf = des_buf;
	int file_length, src_inc, des_inc, src_height, src_width_index, src_width, uv_count,uv_des_buf_count;
	FILE *fp;
	struct timeval t0 = {0},t1 = {0};

	fp = fopen(argv[1],"r+");
	fseek(fp, 0L, SEEK_END);
	file_length = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
	fread(src_buf,1,3840*2160*2,fp);
	fclose(fp);
 	memset(des_buf, 0x80, 3840*2160*1.5);

	src_buf = input_buf;
	des_buf = res_buf;
	u_des_buf = des_buf + (3840*2160);
	v_des_buf = des_buf + (3840*2160) + (3840*2160)/4;

	gettimeofday(&t0,NULL);
	yuyv_yuv420_color_convert(des_buf, src_buf, 3840, 2160);
	gettimeofday(&t1,NULL);
	printf("convertion time %lld \n",(t1.tv_sec - t0.tv_sec)*1000000LL + t1.tv_usec - t0.tv_usec);
	fp = fopen(argv[2],"w+");
	fwrite(res_buf,1,3840*2160*1.5,fp);
	fclose(fp);
	free(input_buf);
	free(res_buf);

	return 0;
}
