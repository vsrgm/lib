#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

/*
 * x2 = (cos(Angle) * (x1 - x0)) - (sin(Angle) *(y1 - y0)) + x0
 * y2 = (sin(Angle) * (x1 - x0)) + (cos(Angle) *(y1 - y0)) + y0
 */

/*
 * Limitation : 
 *    1. Consider YUYV as one unit for image manipulation
 */

void rotate_simple (unsigned int *src_buf, unsigned int *des_buf, int width, int height, float angle)
{
	unsigned int dest_img_size = width * height * 2;
	width >>= 1;
	int x0 = (width >> 1);
	int y0 = (height >> 1);
	int inc_width, inc_height;
	int x2, y2;

	if (!src_buf) {
		printf("src_buf is NULL\n");
		return;
	}

	if (des_buf) {
		memset(des_buf, 0x00, dest_img_size);
	}else {
		printf("des_buf is NULL\n");
		return;
	}
	printf("%f %f \n", cos(90 * (22.0/7.0)/180), sin(90 * (22.0/7.0)/180));

	for (inc_height = 0; inc_height < height; inc_height++) {
		for (inc_width = 0; inc_width < width; inc_width++) {
			x2 = (cos(angle) * (inc_width - x0)) - (sin(angle) * (inc_height - y0)) + x0;
			y2 = (sin(angle) * (inc_width - x0)) + (cos(angle) * (inc_height - y0)) + y0;
			if ((y2 < 0) || (x2 < 0) || (x2 > width) || (y2 > height)) {
//				printf ("BUGGY Logic \n");
				continue;
			}
//			printf("y2 = %d x2 = %d \n", y2, x2);
			des_buf[y2*width + x2] = *src_buf++;
		}
	}
	return ;
}

int main(int argc, char **argv)
{
	FILE *srcptr, *desptr;
	int srclen = 0;
	char *srcbuf, *desbuf;
	int width, height;

	srcptr = fopen(argv[1], "rb");
	if (srcptr) {
		fseek (srcptr, 0L, SEEK_END);
		srclen = ftell(srcptr);
		fseek (srcptr, 0L, SEEK_SET);
	}
	width = atoi(argv[2]);
	height = atoi(argv[3]);
	if (srclen) {
		srcbuf = malloc(width * height * 2);
		if (srcbuf == NULL) {
			printf("Allocation failed \n");
		}
		desbuf = malloc((width*width+height*height)*2);
		if (desbuf == NULL) {
			printf("Allocation failed \n");
		}
		printf("Allocated size %d \n",(width*width+height*height)*2);
	} else {
		printf("File content is empty");
		exit(0);
	}
	fread(srcbuf, 1, width*height*2, srcptr);
	fclose(srcptr);
	srcptr = NULL;
	printf("%f \n", (22.0/7.0)/180);
	rotate_simple((unsigned int *)srcbuf, (unsigned int *)desbuf, width, height, (((float)atoi(argv[4]) * (22.0/7.0)/180)));
	desptr = fopen(argv[5], "wb");
	if (desptr) {
		fwrite(desbuf, 1, (width*width+height*height)*2, desptr);
		fclose (desptr);
		desptr = NULL;
	}

	return 0;
}
