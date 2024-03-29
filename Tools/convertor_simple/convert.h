#ifndef CONVERT_H
#define CONVERT_H

#include <QMainWindow>

namespace Ui {
    class convert;
}

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

private slots:
    void on_pixel_fmt_currentIndexChanged(int index);
    void on_src_img_count_valueChanged(int );
    void on_height_lostFocus();
    void on_bpp_pad_editingFinished();
    void on_bpp_pad_textChanged(const QString &arg1);
    void on_bpp_pad_returnPressed();
};

#define SUCCESS 1
#define FAIL    -1
#define MAX_LIMIT_BYTE 5 // T A 'R G B'

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

    BAYER8_BGGR,
    BAYER8_GBRG,
    BAYER8_GRBG,
    BAYER8_RGGB,

    BAYER10_BGGR,
    BAYER10_GBRG,
    BAYER10_GRBG,
    BAYER10_RGGB,

    BAYER12_BGGR,
    BAYER12_GBRG,
    BAYER12_GRBG,
    BAYER12_RGGB,

    BMP24_BGR,
    BMP24_RGB,

    ABMP32_RGB,

    BAYER10_PACKED,
    RGBIR16,
    BGGR16
};

#endif // CONVERT_H
