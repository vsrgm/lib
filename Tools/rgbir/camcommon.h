#ifndef __CAMCOMMON_H
#define __CAMCOMMON_H

static const unsigned int g_cam_width               = 2592;
static const unsigned int g_cam_height              = 1944;
static const unsigned int g_embedded_date_height    = 2;
static const unsigned int g_scaled_width            = 800;
static const unsigned int g_scaled_height           = 600;
static const unsigned int g_display_width           = 800;
static const unsigned int g_display_height          = 600;

union frame_data
{
    unsigned char data[100];
    struct __attribute__ ((__packed__))
    {
       unsigned char reg300a;
       unsigned char reg300b;
       unsigned char reg300c;
       unsigned char frame_counter[4];
       unsigned char ir_led_status;
       unsigned char crop_x[2];
       unsigned char crop_y[2];
       unsigned char crop_windowx[2];
       unsigned char crop_windowy[2];
       unsigned char width[2];
       unsigned char height[2];
       unsigned char test_pattern;
       unsigned char flip;
       unsigned char mirror;
       unsigned char exposure[2];
       unsigned char again[2];
       unsigned char temp_int;
       unsigned char temp_dec;
    }b;
};

struct __attribute__ ((__packed__)) camera_common_info 
{
    unsigned int magic_number;
    union frame_data emb_data;
    unsigned char bggr_data[g_cam_width * g_cam_height * 2];
    unsigned char rgb_data[g_display_width * g_display_height * 4];
    unsigned char ir_data[g_display_width * g_display_height * 4];
    unsigned int frame_number;
    unsigned int print_info;
    unsigned int signal_given;
    struct timeval start_time;
    struct timeval signal_given_time;

    unsigned char rotation;
    unsigned char gamma_enable;
    unsigned char ir_led_enable;
    unsigned char testpattern_enable;
    unsigned char save_image;
    unsigned char save_image_status;
    unsigned char ir_fps;
    unsigned char rgb_fps;
    double ir_latency;
    double rgb_latency;
};

#endif
