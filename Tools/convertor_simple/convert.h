#ifndef CONVERT_H
#define CONVERT_H

#include <QMainWindow>
#include <stdint.h>
namespace Ui {
    class convert;
}

enum g_pix_fmt
{
    YUV = 4,
    BMP24 = 2,
    BAYER = 4,
};

enum pix_fmt
{
    Y8,
    Y16,

    UYVY,
    YVYU,
    YUYV,
    VYUY,

    YUV422P_YUV,
    YUV420,
    NV12,

    RGB444_BGGR,

    RGB555_RGGB,

    RGB565_GBRG,
    RGB565_RGGB,
    RGB565_BGGR,
    RGB565_GRBG,

    BAYER_BGGR,
    BAYER_GBRG,
    BAYER_GRBG,
    BAYER_RGGB,

    RCCG,

    BMP24_BGR,
    BMP24_RGB,

    ABMP32_RGB,

    BAYER10_PACKED,
    RGBIR16,
    BGGR16
};

struct __attribute__((packed)) __bmp_header
{
    uint8_t  signature[2];
    uint32_t filesize;
    uint32_t reserved;
    uint32_t dataOffset;

    uint32_t sizeofinfoheader;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bitsperpixel;
    uint32_t compression;
    uint32_t imagesize;
    uint32_t xpixpermeter;
    uint32_t ypixelpermeter;
    uint32_t colorused;
    uint32_t importantcolors;
};

class convert : public QMainWindow {
    Q_OBJECT
public:
    convert(QWidget *parent = 0);
    ~convert();
    QImage *imageObject;

protected:
    void changeEvent(QEvent *e);

private:
    Ui::convert *ui;
    void paintimage();
    void resize_base(int width,int height);
    unsigned char *src_buffer, *des_buffer;
    unsigned int width, height, bpp, bpp_container, bpp_shift;
    int frame_stride, size;  
    enum pix_fmt pix_fmt;

private slots:
    void on_pixel_fmt_currentIndexChanged(int index);
    void on_src_img_count_valueChanged(int );
    void on_height_lostFocus();
    void on_equalize_clicked();
    void on_bpp_editingFinished();
    void on_Crop_clicked();
    void on_bpp_container_editingFinished();
    void on_bpp_shift_editingFinished();
    void on_Save_clicked();
};

#define SUCCESS 1
#define FAIL    -1
#define MAX_LIMIT_BYTE 5 // T A 'R G B'


#endif // CONVERT_H
