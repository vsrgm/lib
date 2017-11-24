#include <stdio.h>
#include <stdlib.h>
#include <string.h>

main(int argc, char **argv) {
	int count;
	int width = 1920;
	int height = 1080;
	char dest_string[15];

	unsigned char *src_buf = (char *)malloc (width*height*2);
	unsigned char *des_buf1 = (char *)malloc (width*height);
	unsigned char *des_buf2 = (char *)malloc (width*height);

	long long int file_length;
	FILE *fp, *fp1;

	fp = fopen(argv[1],"r+");
	fseek(fp, 0L, SEEK_END);
	file_length = ftell(fp);
	printf("File length is %d count %d \n",file_length,file_length/(width*height*2));
	fseek(fp, 0L, SEEK_SET);

	for(count=0; file_length; count++) {
		fread(src_buf,1,width*height*2,fp);
		yuyv_interlace_yuv_progressive(des_buf1, des_buf2, src_buf, width, height);
		file_length-=(width*height*2);

		memset(dest_string, 0x00, sizeof(dest_string));
		sprintf(dest_string, "%s_%d.yuv",argv[2],count);

		fp1 = fopen(dest_string,"w+");
		fwrite(des_buf1,1,width*height,fp1);
		fclose(fp1);

		memset(dest_string, 0x00, sizeof(dest_string));
		sprintf(dest_string, "%s_%d.yuv",argv[3],count);
		fp1 = fopen(dest_string,"w+");
		fwrite(des_buf2,1,width*height,fp1);
		fclose(fp1);
		sync();
		printf("File write completed \n");
	}
	fclose(fp);


	free(src_buf);
	free(des_buf1);
	free(des_buf2);

	return 0;
}
