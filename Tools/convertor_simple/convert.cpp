#include "convert.h"
#include "ui_convert.h"
#include "fmt_convert.h"
#include <QMessageBox>
#include <stdio.h>
#include <bayer.h>
#include <string.h>
struct
{
    enum pix_fmt fmt;
    char fmt_name[20];
    float bpp;
    int bpp_padded;
}img_pix_fmt[] = {
    {Y8, "Y8", 8, 0},
    {Y16, "Y16", 16, 0 },

    {UYVY, "UYVY", 16, 0 }, {YVYU, "YVYU", 16, 0 },
    {YUYV, "YUYV", 16, 0 }, {VYUY, "VYUY", 16, 0 },

    {YUV422P_YUV, "YUV422P_YUV", 16, 0 },
    {YUV420, "YUV420", 16, 0 }, {NV12, "NV12", 16, 0 },

    {RGB444_BGGR, "RGB444_BGGR", 16, 0 }, {RGB555_RGGB, "RGB555_RGGB", 16, 0 },
    {RGB565_GBRG, "RGB565_GBRG", 16, 0 }, {RGB565_RGGB, "RGB565_RGGB", 16, 0 },
    {RGB565_BGGR, "RGB565_BGGR", 16, 0 }, {RGB565_GRBG, "RGB565_GRBG", 16, 0 },

    {BAYER8_BGGR, "BAYER8_BGGR", 8, 0 }, {BAYER8_GBRG, "BAYER8_GBRG", 8, 0 },
    {BAYER8_GRBG, "BAYER8_GRBG", 8, 0 }, {BAYER8_RGGB, "BAYER8_RGGB", 8, 0 },

    {BAYER10_BGGR, "BAYER10_BGGR", 10, 6 }, {BAYER10_GBRG, "BAYER10_GBRG", 10, 6 },
    {BAYER10_GRBG, "BAYER10_GRBG", 10, 6 }, {BAYER10_RGGB, "BAYER10_RGGB", 10, 6 },

    {BAYER12_BGGR, "BAYER12_BGGR", 12, 0 }, {BAYER12_GBRG, "BAYER12_GBRG", 12, 0 },
    {BAYER12_GRBG, "BAYER12_GRBG", 12, 0 }, {BAYER12_RGGB, "BAYER12_RGGB", 12, 0 },

    {BMP24_BGR, "BMP24_BGR", 24, 0 },
    {BMP24_RGB, "BMP24_RGB", 24, 0 },

    {ABMP32_RGB, "BMP32_RGB", 32, 0 },
    {BAYER10_PACKED, "BAYER10_PACKED", 16, 0}

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
    unsigned int width, height;
    unsigned int bpp;
    enum pix_fmt pix_fmt;
    unsigned char *src_buffer, *des_buffer;
    unsigned int file_length;
    unsigned int rm_header  = 0;
    char tmp_bufer[50];
    unsigned int in_img_frame_count;
    int frame_stride, size;
    unsigned int malloc_length;

    width = atoi(ui->width->text().toLocal8Bit().data());
    height = atoi(ui->height->text().toLocal8Bit().data());
    frame_stride = atoi(ui->frame_stride->text().toLocal8Bit().data());
    pix_fmt = (enum pix_fmt)ui->pixel_fmt->currentIndex();
    bpp = img_pix_fmt[pix_fmt].bpp + atoi(ui->bpp_pad->text().toLocal8Bit().data());
    malloc_length = width * height * (bpp / 8.0f);
    src_buffer = (unsigned char*)calloc(malloc_length, 1);
    des_buffer = (unsigned char*)calloc(width * height * 3, 1);

    fp = fopen((char*)(ui->file_path->text().toStdString().c_str()), "r+");
    {
        const char* str = ui->file_path->text().toStdString().c_str();

        char* str_match;
        char src_file[300];
        int count = 0;
        memset(src_file, 0x00, 300);
        fp = fopen((char*)ui->file_path->text().toStdString().c_str(), "r+");
#if 0
        if(str_match = strstr(str, "file://")) {
            str_match += strlen("file://");
            while (((str_match[count] >= 32) && (str_match[count] <= 126))) {
                src_file[count] = str_match[count];
                count ++;
                src_file[count] = '\0';
            }
            fp = fopen((char*)(src_file), "r+");
        }
#endif
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
        in_img_frame_count = ((file_length-rm_header)/ (unsigned int)(height*width*(bpp/8.0f)) +
                    (((file_length-rm_header)%(unsigned int)(height*width*(bpp/8.0f)))?1:0));
    }
    sprintf(tmp_bufer, "%d", in_img_frame_count);

    ui->Source_img_integrity->setText(
                                ((file_length-rm_header)%
                                (unsigned int)(height*width*(bpp/8.0f)))?"Fail":"Pass");
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
			save_asyuv(des_buffer,width, height);
        }break;

        case YVYU: {
            convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 1);
            save_buffer(des_buffer, width*height*3);

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
            //convert_bayer8_rgb24(src_buffer, des_buffer, width, height, 0);
			dc1394_bayer_decoding_8bit(src_buffer, des_buffer,  width, height, DC1394_COLOR_FILTER_BGGR, DC1394_BAYER_METHOD_EDGESENSE);
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
        case BAYER12_BGGR: {
            unsigned char *src_buffer1 = (unsigned char *)calloc((width * height), 1);
            convert_bayer12_bayer8(src_buffer, src_buffer1, width, height);
            convert_bayer8_rgb24(src_buffer1, des_buffer, width, height, 0);
            free(src_buffer1);
        }break;

        case BAYER12_GBRG: {
            unsigned char *src_buffer1 = (unsigned char *)calloc((width * height), 1);
            convert_bayer12_bayer8(src_buffer, src_buffer1, width, height);
            convert_bayer8_rgb24(src_buffer1, des_buffer, width, height, 1);
            free(src_buffer1);
        }break;

        case BAYER12_RGGB: {
            unsigned char *src_buffer1 = (unsigned char *)calloc((width * height), 1);
            convert_bayer12_bayer8(src_buffer, src_buffer1, width, height);
            convert_bayer8_rgb24(src_buffer1, des_buffer, width, height, 2);
            free(src_buffer1);
        }break;

        case BAYER12_GRBG: {
            unsigned char *src_buffer1 = (unsigned char *)calloc((width * height), 1);
            convert_bayer12_bayer8(src_buffer, src_buffer1, width, height);
            convert_bayer8_rgb24(src_buffer1, des_buffer, width, height, 3);
            free(src_buffer1);
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
            memcpy(des_buffer, src_buffer, width*height*3);
        }break;

        case BMP24_RGB: {
        }break;

        case ABMP32_RGB: {
            convert_argb32_rgb(src_buffer, des_buffer, width, height);
        }break;

        case BAYER10_PACKED: {
            unsigned char *src_buffer1 = (unsigned char *)calloc(width * height, 1);
            convert_bayer10_packed_rgbir(src_buffer, src_buffer1, width, height);
            convert_bayer8_rgb24(src_buffer1, des_buffer, width, height, 3);
			//dc1394_bayer_decoding_8bit(src_buffer1, des_buffer,  width, height, DC1394_COLOR_FILTER_BGGR, DC1394_BAYER_METHOD_SIMPLE);
			//perform_equalize_rgb24 (des_buffer, width, height);			
			save_asyuv(des_buffer,width, height);
            free(src_buffer1);

            unsigned char *src_ir = (unsigned char *)calloc((width * height)/4, 1);
            extract_bayer10_packed_ir(src_buffer, src_ir, width, height);
			save_ir_asyuv(src_ir,width/2, height/2);            
            free(src_ir);

            break;
        }
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
    enum pix_fmt pix_fmt;
    char tmp_bufer[50];
    pix_fmt = (enum pix_fmt)ui->pixel_fmt->currentIndex();
    sprintf(tmp_bufer, "%d", img_pix_fmt[pix_fmt].bpp_padded);
    ui->bpp_pad->setText(QString::fromStdString(tmp_bufer));
    paintimage();
}

void convert::on_height_lostFocus()
{
    paintimage();
}

void convert::on_bpp_pad_textChanged(const QString &arg1)
{
}

void convert::on_bpp_pad_editingFinished()
{

}

void convert::on_bpp_pad_returnPressed()
{
    paintimage();
}
