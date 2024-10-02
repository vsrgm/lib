#include "convert.h"
#include "ui_convert.h"
#include "fmt_convert.h"
#include <QMessageBox>
#include <stdio.h>
#include <bayer.h>
#include <string.h>
#include <QPainter>
#include <QImage>
#include <QPen>
#include <pthread.h>

struct
{
    enum pix_fmt fmt;
    char fmt_name[20];
    int32_t bpp_container;
    int32_t bpp;
    int32_t bpp_shift;
}img_pix_fmt[] = {
{ Y8, "Y8", 8, 8, 0},
{ Y16, "Y16", 16, 16, 0},

{ UYVY, "UYVY", 16, 16, 0},
{ YVYU, "YVYU", 16, 16, 0},
{ YUYV, "YUYV", 16, 16, 0},
{ VYUY, "VYUY", 16, 16, 0},

{ YUV422P_YUV, "YUV422P_YUV", 16, 16, 0},
{ YUV420, "YUV420", 16, 16, 0},
{ NV12, "NV12", 16, 16, 0},

{ RGB444_BGGR, "RGB444_BGGR", 16, 16, 0},
{ RGB555_RGGB, "RGB555_RGGB", 16, 16, 0},
{ RGB565_GBRG, "RGB565_GBRG", 16, 16, 0},
{ RGB565_RGGB, "RGB565_RGGB", 16, 16, 0},
{ RGB565_BGGR, "RGB565_BGGR", 16, 16, 0},
{ RGB565_GRBG, "RGB565_GRBG", 16, 16, 0},

{ BAYER_BGGR, "BAYER_BGGR", 8, 8, 0},
{ BAYER_GBRG, "BAYER_GBRG", 8, 8, 0},
{ BAYER_GRBG, "BAYER_GRBG", 8, 8, 0},
{ BAYER_RGGB, "BAYER_RGGB", 8, 8, 0},
{ RCCG, "RCCG", 16, 16, 8},

{ BMP24_BGR, "BMP24_BGR", 24, 24, 0},
{ BMP24_RGB, "BMP24_RGB", 24, 24, 0},

{ ABMP32_RGB, "BMP32_RGB", 32, 32, 0},
{ BAYER10_PACKED, "BAYER10_PACKED", 16, 16, 0},
{ RGBIR16, "RGBIR", 16, 16, 0},
{ BGGR16, "BGGR16", 16, 16, 0}

};
pthread_mutex_t lock;

convert::convert(QWidget *parent) :
    QMainWindow(parent),
    src_buffer(NULL),
    des_buffer(NULL),
    ui(new Ui::convert)
{
    int32_t i;
    ui->setupUi(this);
    ui->pixel_fmt->clear();
    for(i = 0;i<(sizeof(img_pix_fmt)/sizeof(img_pix_fmt[0]));i++)
    {
        if(img_pix_fmt[i].fmt_name)
            ui->pixel_fmt->addItem(img_pix_fmt[i].fmt_name);
    }
    width = atoi(ui->width->text().toLocal8Bit().data());
    height = atoi(ui->height->text().toLocal8Bit().data());
    frame_stride = atoi(ui->frame_stride->text().toLocal8Bit().data());
    pix_fmt = (enum pix_fmt)ui->pixel_fmt->currentIndex();
    bpp = atoi(ui->bpp->text().toLocal8Bit().data());
    bpp_container = atoi(ui->bpp_container->text().toLocal8Bit().data());
    bpp_shift = atoi(ui->bpp_shift->text().toLocal8Bit().data());

    if (pthread_mutex_init(&lock, NULL) != 0)
    { 
        printf("\n mutex init has failed\n"); 
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

void convert::paintimage()
{
    pthread_mutex_lock(&lock); 
    /* Take image data from file */
    FILE *fp;
    int32_t file_length;
    int32_t rm_header  = 0;
    char tmp_bufer[50];
    int32_t in_img_frame_count;
    int32_t malloc_length;

    malloc_length = width * height * (bpp_container / 8.0f);    
    src_buffer = (uint8_t*)calloc(malloc_length, 1);
    if (src_buffer == NULL)
    {
        printf("Source Buffer allocation failed of size = %d\n", malloc_length);
    }

    if (des_buffer)
    {
        free(des_buffer);
        des_buffer = NULL;
    }

    des_buffer = (uint8_t*)calloc(width * height * 3, 1);
    if (des_buffer == NULL)
    {
        printf("Destination Buffer allocation failed of size = %d\n", width * height * 3);
    }

    fp = fopen((const char*)(ui->file_path->text().toStdString().c_str()), "r+");
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
    memset(tmp_bufer, 0x00, sizeof(tmp_bufer));
    sprintf(tmp_bufer, "%d", file_length);
    ui->file_size->setText(tmp_bufer);
    if (frame_stride && file_length)
    {
        in_img_frame_count = file_length/frame_stride;
    }else if (file_length)
    {
        in_img_frame_count = ((file_length-rm_header)/ (int32_t)(height*width*(bpp_container/8.0f)) +
                (((file_length-rm_header)%(int32_t)(height*width*(bpp_container/8.0f)))?1:0));
    }
    sprintf(tmp_bufer, "%d", in_img_frame_count);

    ui->Source_img_integrity->setText(
            ((file_length-rm_header)%
             (int32_t)(height*width*(bpp_container/8.0f)))?"Fail":"Pass");
    ui->src_img_count->setMaximum(in_img_frame_count-1);
    ui->num_frames->setText(tmp_bufer);

    if (frame_stride)
    {
        fseek(fp, frame_stride * ui->src_img_count->value(), SEEK_CUR);
    }else
    {
        fseek(fp, (height*width*(bpp_container/8.0f) * ui->src_img_count->value()), SEEK_CUR);
    }
    size = fread(src_buffer, 1, width * height * (bpp_container/8.0f), fp);
    switch(pix_fmt)
    {
        case Y8:
            {
                convert_y8_rgb888(src_buffer, des_buffer, width, height);
            }break;

        case Y16:
            {
                convert_y16_rgb888((uint16_t *)src_buffer, des_buffer, width, height);
            }break;

        case UYVY:
            {
                printf("Width %d Height %d \n", width, height);
                convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 2);
                save_asyuv(des_buffer,width, height);
            }break;

        case YVYU:
            {
                convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 1);
                save_buffer(des_buffer, width*height*3);
            }break;

        case YUYV:
            {
                convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 0);
            }break;

        case VYUY:
            {
                convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 3);
            }break;

        case YUV422P_YUV:
            {
                convert_yuy422p_rgb888(src_buffer, des_buffer, width, height);
            }break;

        case YUV420:
            {
                convert_yuy420p_rgb888(src_buffer, des_buffer, width, height);
            }break;

        case NV12:
            {
                convert_nv12_rgb888(src_buffer, des_buffer, width, height);
            }break;

        case RGB444_BGGR:
            {
            }break;

        case RGB555_RGGB:
            {
                convert_rgb555_888(src_buffer, des_buffer, width, height, 0);
            }break;

        case RGB565_RGGB:
            {
                convert_rgb565_888(src_buffer, des_buffer, width, height, 0);
            }break;

        case RGB565_GBRG:
            {
                convert_rgb565_888(src_buffer, des_buffer, width, height, 1);
            }break;

        case RGB565_BGGR:
            {
                convert_rgb565_888(src_buffer, des_buffer, width, height, 2);
            }break;

        case RGB565_GRBG:
            {
                convert_rgb565_888(src_buffer, des_buffer, width, height, 3);
            }break;

        case BAYER_BGGR:
            {
                convert_bayer_rgb24(src_buffer, des_buffer, width, height, 0, bpp_container, bpp_shift);
            }break;

        case BAYER_GBRG:
            {
                convert_bayer_rgb24(src_buffer, des_buffer, width, height, 1, bpp_container, bpp_shift);
            }break;

        case BAYER_RGGB:
            {
                convert_bayer_rgb24(src_buffer, des_buffer, width, height, 2, bpp_container, bpp_shift);
            }break;

        case BAYER_GRBG:
            {
                convert_bayer_rgb24(src_buffer, des_buffer, width, height, 3, bpp_container, bpp_shift);
            }break;

        case RCCG:
            {
                convert_rccg_rgb24(src_buffer, des_buffer, width, height, 0, bpp_container, bpp_shift);
            }break;

        case BMP24_BGR:
            {
                convert_bgr888_rgb888(src_buffer, des_buffer, width, height);
            }break;

        case BMP24_RGB:
            {
                memcpy(des_buffer, src_buffer, width * height * 3);
            }break;

        case ABMP32_RGB:
            {
                convert_argb32_rgb(src_buffer, des_buffer, width, height);
            }break;

        case BAYER10_PACKED:
            {
                uint8_t *src_buffer1 = (uint8_t *)calloc(width * height, 1);
                convert_bayer10_packed_rgbir(src_buffer, src_buffer1, width, height);
                convert_bayer_rgb24(src_buffer1, des_buffer, width, height, 3, 8, 0);
                //dc1394_bayer_decoding_8bit(src_buffer1, des_buffer,  width, height,
                //   DC1394_COLOR_FILTER_BGGR, DC1394_BAYER_METHOD_SIMPLE);
                save_asyuv(des_buffer,width, height);
                free(src_buffer1);

                uint8_t *src_ir = (uint8_t *)calloc((width * height)/4, 1);
                extract_bayer10_packed_ir(src_buffer, src_ir, width, height);
                save_ir_asyuv(src_ir,width/2, height/2);
                free(src_ir);
                break;
            }

        case RGBIR16:
            {
                uint8_t *src_buffer1 = (uint8_t *)calloc(width * height, 1);
                convert_RGBIR16_bayer8(src_buffer, src_buffer1, width, height);
                convert_bayer_rgb24(src_buffer1, des_buffer, width, height, 1, 8, 0);
                perform_equalize_rgb24 (des_buffer, width, height);
                save_asyuv(des_buffer,width, height);
                uint8_t *src_ir = (uint8_t *)calloc((width * height)/4, 1);
                extract_RGBIR16_IR8(src_buffer, src_ir, width, height);
                save_ir_asyuv(src_ir,width/2, height/2);
                free(src_buffer1);
                break;
            }
        case BGGR16:
            {
                uint8_t *src_buffer1 = (uint8_t *)calloc(width * height, 1);
                convert_bit16_bit8((uint16_t *)src_buffer, src_buffer1, width, height);
                convert_bayer_rgb24(src_buffer1, des_buffer, width, height, 2, 8, 0);
                save_asyuv(des_buffer,width, height);
                uint8_t *src_ir = (uint8_t *)calloc((width * height)/4, 1);
                extract_RGBIR16_IR8(src_buffer, src_ir, width, height);
                save_ir_asyuv(src_ir,width/2, height/2);
                free(src_buffer1);
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
    pthread_mutex_unlock(&lock); 
}

void convert::on_pixel_fmt_currentIndexChanged(int32_t index)
{
    char tmp_buffer[50];

    pix_fmt = (enum pix_fmt)ui->pixel_fmt->currentIndex();
    memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
    sprintf(tmp_buffer, "%d", img_pix_fmt[pix_fmt].bpp_container);
    ui->bpp_container->setText(QString::fromStdString(tmp_buffer));

    memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
    sprintf(tmp_buffer, "%d", img_pix_fmt[pix_fmt].bpp);
    ui->bpp->setText(QString::fromStdString(tmp_buffer));

    memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
    sprintf(tmp_buffer, "%d", img_pix_fmt[pix_fmt].bpp_shift);
    ui->bpp_shift->setText(QString::fromStdString(tmp_buffer));

    width = atoi(ui->width->text().toLocal8Bit().data());
    height = atoi(ui->height->text().toLocal8Bit().data());
    frame_stride = atoi(ui->frame_stride->text().toLocal8Bit().data());
    bpp = atoi(ui->bpp->text().toLocal8Bit().data());
    bpp_container = atoi(ui->bpp_container->text().toLocal8Bit().data());
    bpp_shift = atoi(ui->bpp_shift->text().toLocal8Bit().data());
    paintimage();
}

void convert::on_src_img_count_valueChanged(int32_t)
{
    paintimage();
}

void convert::on_height_lostFocus()
{
    paintimage();
}

void convert::on_equalize_clicked()
{
    int32_t width = atoi(ui->width->text().toLocal8Bit().data());
    int32_t height = atoi(ui->height->text().toLocal8Bit().data());
    QImage *imageObject = new QImage(width, height, QImage::Format_RGB888);

    uint8_t* modifed_buffer = (uint8_t*)calloc(width * height * 3, 1);
    memcpy(modifed_buffer, des_buffer, width * height *3);
    if (modifed_buffer)
    {
        perform_equalize_rgb24 (modifed_buffer, width, height);
        memcpy(imageObject->bits(), modifed_buffer, width * height *3);
    }
    ui->modified_draw_window->setPixmap(QPixmap::fromImage(*imageObject));
    ui->modified_draw_window->update();

    free(modifed_buffer);
    delete imageObject;
}


void convert::on_bpp_editingFinished()
{
    bpp = atoi(ui->bpp->text().toLocal8Bit().data());
    bpp_container = atoi(ui->bpp_container->text().toLocal8Bit().data());
    bpp_shift = atoi(ui->bpp_shift->text().toLocal8Bit().data());
    paintimage();
}

void convert::on_bpp_container_editingFinished()
{
    bpp = atoi(ui->bpp->text().toLocal8Bit().data());
    bpp_container = atoi(ui->bpp_container->text().toLocal8Bit().data());
    bpp_shift = atoi(ui->bpp_shift->text().toLocal8Bit().data());
    paintimage();
}

void convert::on_Crop_clicked()
{
    int32_t x = atoi(ui->crop_x->text().toLocal8Bit().data());
    int32_t y = atoi(ui->crop_y->text().toLocal8Bit().data());
    int32_t width = atoi(ui->crop_width->text().toLocal8Bit().data());
    int32_t render_width = width%4?(width-(width%4)):width;
    int32_t height = atoi(ui->crop_height->text().toLocal8Bit().data());
    int32_t srcwidth = atoi(ui->width->text().toLocal8Bit().data());
    int32_t srcheight = atoi(ui->height->text().toLocal8Bit().data());
    int32_t bpp = 3;
    uint8_t*crop_buffer = (uint8_t*)malloc(width*height*bpp);
    int32_t ret = -1;
    
    QImage *srcimageObject = new QImage(srcwidth, srcheight, QImage::Format_RGB888);
    memcpy(srcimageObject->bits(), des_buffer, srcwidth * srcheight *3);
    QPainter paint32_ter(srcimageObject);
    QPen pen;
    pen.setWidth(1);
    pen.setColor(Qt::red);
    QRect rect;

    paint32_ter.setPen(pen);
    rect.setTopLeft(QPoint(x-(width/2),y-(height/2)));
    rect.setWidth(width);
    rect.setHeight(height);
    paint32_ter.drawRect(rect);
    ui->draw_window->setPixmap(QPixmap::fromImage(*srcimageObject));
    ui->draw_window->update();

    QImage *cropimageObject = new QImage(render_width, height, QImage::Format_RGB888);
    ret = perform_crop(crop_buffer, des_buffer, x, y, width, height, bpp, srcwidth, srcheight);
    if (ret < 0)
    {
        QMessageBox msgBox;
        msgBox.setText("Crop limit specified is not feasbile to crop");
        msgBox.exec();
    }
    ret = perform_stride_correction(cropimageObject->bits(), crop_buffer,
           render_width, height, width, height, bpp);
    
    perform_equalize_rgb24 (cropimageObject->bits(), render_width, height);

    ui->modified_draw_window->setPixmap(QPixmap::fromImage(*cropimageObject));
    ui->modified_draw_window->update();
    //delete cropimageObject;
    //delete srcimageObject;
}

void convert::on_bpp_shift_editingFinished()
{
    bpp = atoi(ui->bpp->text().toLocal8Bit().data());
    bpp_container = atoi(ui->bpp_container->text().toLocal8Bit().data());
    bpp_shift = atoi(ui->bpp_shift->text().toLocal8Bit().data());
    paintimage();
}


void convert::on_Save_clicked()
{
    struct __attribute__((packed)) __bmp_header bmp;
    int32_t width = atoi(ui->width->text().toLocal8Bit().data());
    int32_t height = atoi(ui->height->text().toLocal8Bit().data());
    uint8_t *bmp_fmt_buffer_bgr = (uint8_t *)malloc(width * height * 3);

    bmp.signature[0]='B';
    bmp.signature[1]='M';
    bmp.filesize = width * height * 3 + sizeof(bmp);
    bmp.dataOffset = 54;
    bmp.sizeofinfoheader = 40;
    bmp.width = width;
    bmp.height = height;
    bmp.planes = 1;
    bmp.bitsperpixel = 24;
    bmp.compression = 0;
    bmp.imagesize=0;
    bmp.xpixpermeter = width;
    bmp.ypixelpermeter = height;
    bmp.colorused = 256;
    bmp.importantcolors = 0;

    FILE *fp;
    fp = fopen("Sample.bmp", "w");
    if (fp == NULL)
    {
        printf("Fail creation failed \n");
    }
    fwrite (&bmp, 1, sizeof(bmp), fp);
    convert_bgr888_rgb888(des_buffer, bmp_fmt_buffer_bgr, width, height);
    fwrite (bmp_fmt_buffer_bgr, 1, width * height * 3, fp);
    free(bmp_fmt_buffer_bgr);
    fclose(fp);
}

