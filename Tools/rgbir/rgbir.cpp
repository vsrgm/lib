#include "rgbir.h"
#include "ui_rgbir.h"

#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <string>
#include <vector>
#include <pthread.h>
#include <sys/mman.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "camcommon.h"


pthread_mutex_t ir_child_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t ir_child_cond = PTHREAD_COND_INITIALIZER;

pthread_mutex_t rgb_child_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t rgb_child_cond = PTHREAD_COND_INITIALIZER;
void sig_frmapp_irdata(int sigval)
{
    signal(SIGUSR1, sig_frmapp_irdata); /* reset signal */
    pthread_mutex_lock(&ir_child_lock);
    pthread_cond_signal(&ir_child_cond);
    pthread_mutex_unlock(&ir_child_lock);
}

void sig_frmapp_rgbdata(int sigval)
{
    signal(SIGUSR2, sig_frmapp_rgbdata);     /* reset signal */
    pthread_mutex_lock(&rgb_child_lock);
    pthread_cond_signal(&rgb_child_cond);
    pthread_mutex_unlock(&rgb_child_lock);
}

void file_check_available(char *fname)
{
    FILE *fp = fopen(fname, "r");
    if (fp == NULL)
    {
        fp = fopen(fname, "w");
        fclose(fp);
    }
    else
    {
        fclose(fp);
    }
    return;
}

struct fn_args
{
    struct camera_common_info *info;
    Ui::rgbir *ui;
    QImage *rgb_img_data;
    QImage *ir_img_data;
};


void RenderRgbThread::run()
{
    while (1)
    {
        pthread_mutex_lock(&rgb_child_lock);
        pthread_cond_wait(&rgb_child_cond, &rgb_child_lock);
        emit buffer_available();
        pthread_mutex_unlock(&rgb_child_lock);
    }
}

void RenderIrThread::run()
{
    while (1)
    {
        pthread_mutex_lock(&ir_child_lock);
        pthread_cond_wait(&ir_child_cond, &ir_child_lock);
        emit buffer_available();
        pthread_mutex_unlock(&ir_child_lock);
    }
}

void rgbir::on_saveImage_clicked(void)
{
    char tmp_buffer[100];
    QPalette palette = ui->rgbir_saved_image_status->palette();
    palette.setColor(ui->rgbir_saved_image_status->foregroundRole(), Qt::red);

    memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
    sprintf(tmp_buffer, "Raw RGBIR Image Saving in Progress");
    ui->rgbir_saved_image_status->setText(tmp_buffer);
    ui->rgbir_saved_image_status->setPalette(palette);

    memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
    sprintf(tmp_buffer, "Bayer Image Saving in Progress");
    ui->bayer_saved_image_status_status->setText(tmp_buffer);
    ui->bayer_saved_image_status_status->setPalette(palette);

    memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
    sprintf(tmp_buffer, "Extracted IR Saving in Progress");
    ui->ir_saved_image_status->setText(tmp_buffer);
    ui->ir_saved_image_status->setPalette(palette);
    info->save_image = 1;
}
void rgbir::on_stop_stream_clicked(void)
{
    start_stream = 0U;
}

void rgbir::on_start_stream_clicked(void)
{
    start_stream = 1U;
}

void rgbir::update_rgb_buffer(void)
{
    if (start_stream)
    {
        memcpy(rgb_img_data->bits(), info->rgb_data, g_display_width * g_display_height *4);
        ui->draw_window_rgb->setPixmap(QPixmap::fromImage(*rgb_img_data));
        ui->draw_window_rgb->update();
    }
}

void rgbir::update_ir_buffer(void)
{
    if (start_stream)
    {
        memcpy(ir_img_data->bits(), info->ir_data, g_display_width * g_display_height *4);
        ui->draw_window_ir->setPixmap(QPixmap::fromImage(*ir_img_data));
        ui->draw_window_ir->update();

        static unsigned int frame_counter = 0;
        unsigned int current_frame_counter = (info->emb_data.b.frame_counter[0]<<24) |
            (info->emb_data.b.frame_counter[1]<< 16) |
            (info->emb_data.b.frame_counter[2]<<8) |
            (info->emb_data.b.frame_counter[3]);
        if (frame_counter != current_frame_counter)
        {
            char tmp_buffer[100];
            QPalette palette = ui->sensor_uid->palette();
            palette.setColor(ui->sensor_uid->foregroundRole(), Qt::red);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%02x-%02x-%02x", info->emb_data.b.reg300a,
                    info->emb_data.b.reg300b,
                    info->emb_data.b.reg300c);
            ui->sensor_uid->setText(tmp_buffer);
            ui->sensor_uid->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%d",(info->emb_data.b.exposure[0] << 8) |
                    (info->emb_data.b.exposure[1]));
            ui->sensor_exposure->setText(tmp_buffer);
            ui->sensor_exposure->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%d",(info->emb_data.b.again[0] << 8) |
                    (info->emb_data.b.again[1]));
            ui->sensor_gain->setText(tmp_buffer);
            ui->sensor_gain->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%d",info->emb_data.b.temp_int);
            ui->sensor_temperature->setText(tmp_buffer);
            ui->sensor_temperature->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%s",(info->emb_data.b.test_pattern & 0x80)?
                    "Enabled":"Disabled");
            ui->sensor_testpattern->setText(tmp_buffer);
            ui->sensor_testpattern->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%s",(info->emb_data.b.flip & 0x04)?
                    "Enabled":"Disabled");
            ui->sensor_flip->setText(tmp_buffer);
            ui->sensor_flip->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%s",(info->emb_data.b.mirror & 0x04)?
                    "Enabled":"Disabled");
            ui->sensor_mirror->setText(tmp_buffer);
            ui->sensor_mirror->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%dx%d",
                    (info->emb_data.b.width[0] << 8) | (info->emb_data.b.width[1]),
                    (info->emb_data.b.height[0] << 8) | (info->emb_data.b.height[1]));
            ui->sensor_output_resolution->setText(tmp_buffer);
            ui->sensor_output_resolution->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%d", current_frame_counter);
            ui->sensor_frame_counter->setText(tmp_buffer);
            ui->sensor_frame_counter->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%s",info->emb_data.b.ir_led_status?
                    "Enabled":"Disabled");
            ui->sensor_ir_status->setText(tmp_buffer);
            ui->sensor_ir_status->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "x:%d y:%d cw:%d ch:%d",
                    (info->emb_data.b.crop_x[0] << 8) | (info->emb_data.b.crop_x[1]),
                    (info->emb_data.b.crop_y[0] << 8) | (info->emb_data.b.crop_y[1]),
                    (info->emb_data.b.crop_windowx[0] << 8) | (info->emb_data.b.crop_windowx[1]),
                    (info->emb_data.b.crop_windowy[0] << 8) | (info->emb_data.b.crop_windowy[1]));
            ui->sensor_crop_info->setText(tmp_buffer);
            ui->sensor_crop_info->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%d fps", info->ir_fps);
            ui->ir_fps->setText(tmp_buffer);
            ui->ir_fps->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%d fps", info->rgb_fps);
            ui->rgb_fps->setText(tmp_buffer);
            ui->rgb_fps->setPalette(palette);


            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%.06f ms", info->ir_latency * 1000);
            ui->ir_latency->setText(tmp_buffer);
            ui->ir_latency->setPalette(palette);

            memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
            sprintf(tmp_buffer, "%.06f ms", info->rgb_latency * 1000);
            ui->rgb_latency->setText(tmp_buffer);
            ui->rgb_latency->setPalette(palette);

            ui->sensor_crop_info->setPalette(palette);
            info->rotation = (ui->rotation->isChecked() == true) ?1:0;
            info->gamma_enable = (ui->gamma->isChecked() == true) ?1:0;
            info->ir_led_enable = (ui->irled->isChecked() == true) ?1:0;
            info->testpattern_enable = (ui->testpattern->isChecked() == true) ?1:0;

            if ((info->save_image_status & 0x02) == 0)
            {
                memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
                sprintf(tmp_buffer, "Raw RGBIR Image Saved Successfully");
                ui->rgbir_saved_image_status->setText(tmp_buffer);
                ui->rgbir_saved_image_status->setPalette(palette);
            }

            if ((info->save_image_status & 0x04) == 0)
            {
                memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
                sprintf(tmp_buffer, "Bayer Image Saved Successfully");
                ui->bayer_saved_image_status_status->setText(tmp_buffer);
                ui->bayer_saved_image_status_status->setPalette(palette);
            }
            if ((info->save_image_status & 0x01) == 0)
            {
                memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
                sprintf(tmp_buffer, "Extracted IR Saved Successfully");
                ui->ir_saved_image_status->setText(tmp_buffer);
                ui->ir_saved_image_status->setPalette(palette);
            }
        }
    }
}

rgbir::rgbir(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::rgbir),
    start_stream(1U)
{
    ui->setupUi(this);
    signal(SIGUSR1, sig_frmapp_irdata);
    signal(SIGUSR2, sig_frmapp_rgbdata);

    file_check_available("shmfile");
    char * shm_common_info = (char*)shmat(shmget(ftok("shmfile", 66),
                sizeof(struct camera_common_info), 0666 | IPC_CREAT), (void*)0, 0);
    if (shm_common_info == NULL)
    {
        printf("Shm memory mapping failed shm %p\n", shm_common_info);
        exit(0);
    }

    rgb_img_data = new QImage(g_display_width, g_display_height, QImage::Format_ARGB32);
    ir_img_data = new QImage(g_display_width, g_display_height, QImage::Format_ARGB32);
    info = (struct camera_common_info *)shm_common_info;
 
    RenderRgbThread *_thread = new RenderRgbThread;
    connect(_thread, &RenderRgbThread::buffer_available, this, &rgbir::update_rgb_buffer);
    _thread->start();

    RenderIrThread *_thread1 = new RenderIrThread;
    connect(_thread1, &RenderIrThread::buffer_available, this, &rgbir::update_ir_buffer);
    _thread1->start();

}

rgbir::~rgbir()
{
    delete ui;
}
