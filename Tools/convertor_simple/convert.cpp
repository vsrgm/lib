#include "convert.h"
#include "ui_convert.h"
#include "fmt_convert.h"
#include <QMessageBox>
#include <stdio.h>

struct
{
	enum pix_fmt fmt;
	char fmt_name[20];
	int bpp;
}img_pix_fmt[] = {
	{Y8, "Y8", 8}, {Y16, "Y16", 16}, {UYVY, "UYVY", 16}, {YVYU, "YVYU", 16},
	{YUYV, "YUYV", 16}, {VYUY, "VYUY", 16}, {YUV422P_YUV, "YUV422P_YUV", 16},
	{YUV420, "YUV420", 16}, {NV12, "NV12", 16},

	{RGB444_BGGR, "RGB444_BGGR", 16}, {RGB555_RGGB, "RGB555_RGGB", 16},
	{RGB565_GBRG, "RGB565_GBRG", 16}, {RGB565_RGGB, "RGB565_RGGB", 16},
	{RGB565_BGGR, "RGB565_BGGR", 16}, {RGB565_GRBG, "RGB565_GRBG", 16},

	{BAYER8_BGGR, "BAYER8_BGGR", 8}, {BAYER8_GBRG, "BAYER8_GBRG", 8},
	{BAYER8_GRBG, "BAYER8_GRBG", 8}, {BAYER8_RGGB, "BAYER8_RGGB", 8},

	{BAYER10_BGGR, "BAYER10_BGGR", 10}, {BAYER10_GBRG, "BAYER10_GBRG", 10},
	{BAYER10_GRBG, "BAYER10_GRBG", 10}, {BAYER10_RGGB, "BAYER10_RGGB", 10},
	{BMP24_BGR, "BMP24_BGR", 24}, {BMP24_RGB, "BMP24_RGB", 24}
};

convert::convert(QWidget *parent) :
	QMainWindow(parent),
	ui(new Ui::convert)
{
	int i;
	ui->setupUi(this);
	ui->pixel_fmt->clear();
	for(i = 0;i<(sizeof(img_pix_fmt)/sizeof(img_pix_fmt[0]));i++)
	{
		if(img_pix_fmt[i].fmt_name)
			ui->pixel_fmt->addItem(img_pix_fmt[i].fmt_name);
	}
}

convert::~convert()
{
	delete ui;
}

void convert::changeEvent(QEvent *e)
{
	QMainWindow::changeEvent(e);
	switch (e->type()) {
	case QEvent::LanguageChange:
	ui->retranslateUi(this);
	break;
	default:
	break;
	}
}
int convert_yuy420_rgb888(unsigned char* yuyv_buffer, unsigned char* rgb888, unsigned int width, unsigned int height);

void convert::paintimage()
{
	/* Take image data from file */
	FILE *fp;
	unsigned int width, height, bpp;
	enum pix_fmt pix_fmt;
	unsigned char *src_buffer, *des_buffer;
	unsigned int file_length;
	unsigned int rm_header  = 0;
	char tmp_bufer[50];
	unsigned int in_img_frame_count;
	int frame_stride, size;

	width = atoi(ui->width->text().toLocal8Bit().data());
	height = atoi(ui->height->text().toLocal8Bit().data());
	frame_stride = atoi(ui->frame_stride->text().toLocal8Bit().data());

	pix_fmt = (enum pix_fmt)ui->pixel_fmt->currentIndex();
	bpp = (img_pix_fmt[pix_fmt].bpp & 0x7)?((img_pix_fmt[pix_fmt].bpp/8) +1)*8:img_pix_fmt[pix_fmt].bpp;

	src_buffer = (unsigned char*)calloc(width * height * (bpp/8), 1);
	des_buffer = (unsigned char*)calloc(width * height * 3, 1);

	fp = fopen((char*)(ui->file_path->text().toLocal8Bit().data()), "r+");
	{
		char* str = ui->file_path->text().toLocal8Bit().data();
		char* str_match;
		char src_file[300];
		int count = 0;
		memset(src_file, 0x00, 300);

		if(str_match = strstr(str, "file://")) {
			str_match += strlen("file://");
			while (((str_match[count] >= 32) && (str_match[count] <= 126))) {
				src_file[count] = str_match[count];
				count ++;
				src_file[count] = '\0';
			}
			fp = fopen((char*)(src_file), "r+");
		}
	}

	if(fp == NULL)
	{
		QMessageBox msgBox;
		msgBox.setText("Please mention the input file name.");
		msgBox.exec();
		goto exit;
	}

	fseek(fp, 0L, SEEK_END);
	file_length = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
	sprintf(tmp_bufer, "%d", file_length);
	ui->file_size->setText(tmp_bufer);

	if (frame_stride) {
       		in_img_frame_count = file_length/frame_stride;
	}else {
		in_img_frame_count = ((file_length-rm_header)/(height*width*(bpp/8)) +
					(((file_length-rm_header)%(height*width*(bpp/8)))?1:0));
	}
	sprintf(tmp_bufer, "%d", in_img_frame_count);

	ui->Source_img_integrity->setText(((file_length-rm_header)%(height*width*(bpp/8)))?"Fail":"Pass");
	ui->src_img_count->setMaximum(in_img_frame_count-1);
	ui->num_frames->setText(tmp_bufer);

	if (frame_stride) {
		fseek(fp, frame_stride * ui->src_img_count->value(), SEEK_CUR);
	}else {
		fseek(fp, (height*width*(bpp/8)*ui->src_img_count->value()), SEEK_CUR);
	}
	size = fread(src_buffer, 1, width * height * (bpp/8), fp);

	switch(pix_fmt) {
		case Y8: {
			convert_y8_rgb888(src_buffer, des_buffer, width, height);
		}break;

		case Y16: {
			convert_y16_rgb888((unsigned short *)src_buffer, des_buffer, width, height);
		}break;

		case UYVY: {
			convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 2);
		}break;

		case YVYU: {
			convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 1);
		}break;

		case YUYV: {
			convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 0);
		}break;

		case VYUY: {
			convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 3);
		}break;

		case YUV422P_YUV: {
			convert_yuy422p_rgb888(src_buffer, des_buffer, width, height);
		}break;

		case YUV420: {
			convert_yuy420p_rgb888(src_buffer, des_buffer, width, height);
		}break;

		case NV12: {
			convert_nv12_rgb888(src_buffer, des_buffer, width, height);
		}break;

		case RGB444_BGGR: {
		}break;

		case RGB555_RGGB: {
			convert_rgb555_888(src_buffer, des_buffer, width, height, 0);
		}break;

		case RGB565_RGGB: {
			convert_rgb565_888(src_buffer, des_buffer, width, height, 0);
		}break;

		case RGB565_GBRG: {
			convert_rgb565_888(src_buffer, des_buffer, width, height, 1);
		}break;

		case RGB565_BGGR: {
			convert_rgb565_888(src_buffer, des_buffer, width, height, 2);
		}break;

		case RGB565_GRBG: {
			convert_rgb565_888(src_buffer, des_buffer, width, height, 3);
		}break;

		case BAYER8_BGGR: {
			convert_bayer8_rgb24(src_buffer, des_buffer, width, height, 0);
		}break;

		case BAYER8_GBRG: {
			convert_bayer8_rgb24(src_buffer, des_buffer, width, height, 1);
		}break;

		case BAYER8_RGGB: {
			convert_bayer8_rgb24(src_buffer, des_buffer, width, height, 2);
		}break;

		case BAYER8_GRBG: {
			convert_bayer8_rgb24(src_buffer, des_buffer, width, height, 3);
		}break;

		case BAYER10_BGGR: {
			convert_bayer_gen_rgb24((short unsigned int*)src_buffer, des_buffer,
						width, height, 0, img_pix_fmt[pix_fmt].bpp -8);
		}break;

		case BAYER10_GBRG: {
			convert_bayer_gen_rgb24((short unsigned int*)src_buffer, des_buffer, width,
						height, 1, img_pix_fmt[pix_fmt].bpp -8);
		}break;

		case BAYER10_RGGB: {
			convert_bayer_gen_rgb24((short unsigned int*)src_buffer, des_buffer,
						width, height, 2, img_pix_fmt[pix_fmt].bpp -8);
		}break;

		case BAYER10_GRBG: {
			convert_bayer_gen_rgb24((short unsigned int*)src_buffer, des_buffer,
						width, height, 3, img_pix_fmt[pix_fmt].bpp -8);
		}break;

		case BMP24_BGR: {
		}break;

		case BMP24_RGB: {
		}break;
	}

	imageObject = new QImage(width, height, QImage::Format_RGB888);
	memcpy(imageObject->bits(), des_buffer, width * height *3);
	ui->draw_window->setPixmap(QPixmap::fromImage(*imageObject));
	ui->draw_window->update();

	delete imageObject;
exit:
	free(src_buffer);
	free(des_buffer);
}

void convert::on_src_img_count_valueChanged(int )
{
	paintimage();
}

void convert::on_pixel_fmt_currentIndexChanged(int index)
{
//	ui->src_img_count->setValue(0);
	paintimage();
}

void convert::on_height_lostFocus()
{
    paintimage();
}
