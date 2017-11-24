#include <stdio.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <string.h>

main(int argc, char **argv) {
	int count=0;
	unsigned char *src_buf = (char *)malloc (1920*1080*2);
	unsigned char *des_buf = (char *)malloc (1920*1080*1.5);
	unsigned char *uv_des_buf = des_buf + (1920*1080);

	int file_length, src_inc, des_inc, src_height, src_width_index, src_width, uv_count,uv_des_buf_count;
	FILE *fp;

	fp = fopen(argv[1],"r+");
	fseek(fp, 0L, SEEK_END);
	file_length = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
	fread(src_buf,1,1920*1080*2,fp);
	fclose(fp);
 	memset(des_buf, 0x80, 1920*1080*1.5);

for (count =0;count<1000;count++) {
	yuyv_nv12 (des_buf, src_buf, 1920, 1080);
}
	fp = fopen(argv[2],"w+");
	fwrite(des_buf,1,1920*1080*1.5,fp);
	fclose(fp);
	free(src_buf);
	free(des_buf);

	return 0;
}
