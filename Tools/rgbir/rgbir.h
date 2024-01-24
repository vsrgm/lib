#ifndef RGBIR_H
#define RGBIR_H

#include <QMainWindow>
#include <QThread>
namespace Ui {
class rgbir;
}

class rgbir : public QMainWindow
{
    Q_OBJECT

public:
    explicit rgbir(QWidget *parent = 0);
    ~rgbir();
    void update_rgb_buffer(void);
    void update_ir_buffer(void);
private:
    Ui::rgbir *ui;
    static void* render_rgb_data(void *args);
    static void* render_ir_data(void *args);
    QImage *rgb_img_data;
    QImage *ir_img_data;
    struct camera_common_info *info;
    unsigned char start_stream;

private slots:
    void on_start_stream_clicked();
    void on_stop_stream_clicked();
    void on_saveImage_clicked();

};

class RenderRgbThread:  public QThread
{
Q_OBJECT
  // as needed
signals:
  void buffer_available(void);
public:
  void run();
};

class RenderIrThread:  public QThread
{
Q_OBJECT
  // as needed
signals:
  void buffer_available(void);
public:
  void run();
};


#endif // RGBIR_H
