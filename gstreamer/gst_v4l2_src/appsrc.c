#include <gst/gst.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <gst/video/video.h>
#include <gst/app/gstappsrc.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int width = 1920, height = 1080;
enum _fmt_mode {
	UYVY = 0,
	YUY2,
	RGB565,
	BAYER,
	JPEG,
	H264,
};

enum _mach_type {
	X86,
	TEGRA,
};

int fmt = UYVY;
int mach = X86;

int v4l2_mipi1_get_buffer(unsigned char **buf);
int v4l2_mipi1_src_init (void);
void v4l2_mipi1_src_exit(void);

static GMainLoop *loop;

static void
cb_need_data (GstElement *appsrc, 
			guint	unused_size, 
			gpointer user_data)
{
#if defined(GSTREAMER_1__0)
	static GstClockTime timestamp = 0;
#endif
	static GstBuffer *appsrc_buffer;
	GstFlowReturn ret;

	unsigned char *v4l2_buffer1;

	static struct timeval cur_tv, prev_tv;
	static int old_stream_count, stream_count, fps;
	static struct timeval t0 = {0}, t1 = {0};
	static struct timeval t0_start = {0}, t1_end = {0};

	static long int cap_buf1_time, total_time;
	static long int cap_buf1_max, total_max;
	static long int total_hit;
	static long int temp_time;

#if defined(GSTREAMER_1__0)
	static GstMemory *wrapped_mem = NULL;
#elif defined(GSTREAMER_0__1_0)
//	GstBuffer *buffer;
//	printf("%s %s %d \n", __FILE__, __func__, __LINE__);
#endif
	guint size;
	gettimeofday(&t0_start, NULL);

//	gettimeofday(&t1, NULL);
	gettimeofday(&t0, NULL);
	v4l2_mipi1_get_buffer(&v4l2_buffer1);
	gettimeofday(&t1, NULL);
	temp_time = ((long int)(t1.tv_sec - t0.tv_sec)*1000000 + t1.tv_usec - t0.tv_usec);
	cap_buf1_max = (cap_buf1_max < temp_time)?temp_time:cap_buf1_max;
	cap_buf1_time += temp_time;
//	printf("Get Buf 1 %06lu\n", (long int)(t1.tv_sec - t0.tv_sec)*1000000 + t1.tv_usec - t0.tv_usec);

	size = width * height * 2;

	appsrc_buffer = gst_buffer_new();
#if defined(GSTREAMER_1__0)
	gst_buffer_remove_all_memory(appsrc_buffer);
	wrapped_mem = gst_memory_new_wrapped(0, 
				(gpointer) v4l2_buffer1, size, 0, 
				size, NULL, NULL);

	gst_buffer_append_memory(appsrc_buffer, wrapped_mem);
	GST_BUFFER_PTS (appsrc_buffer) = timestamp;
	GST_BUFFER_DURATION (appsrc_buffer) = gst_util_uint64_scale_int (1, GST_SECOND, 30);
	timestamp += GST_BUFFER_DURATION (appsrc_buffer);

#elif defined(GSTREAMER_0__1_0)
	GST_BUFFER_DATA(appsrc_buffer) = (guint8 *)v4l2_buffer1;
	GST_BUFFER_SIZE(appsrc_buffer) = size;
#endif

	g_signal_emit_by_name (appsrc, "push-buffer", appsrc_buffer, &ret);
	
	if (ret != GST_FLOW_OK) {
		/* something wrong, stop pushing */
		g_main_loop_quit (loop);
	}
	gst_buffer_unref(appsrc_buffer);

	gettimeofday(&cur_tv, NULL);
	if(cur_tv.tv_sec > prev_tv.tv_sec) {
		prev_tv	= cur_tv;
		fps	= stream_count - old_stream_count;
		old_stream_count = stream_count;
		printf("fps %06d count %06d ", fps, stream_count);
		if (fps) {
			printf("Buf1 %06ld:%06ld total_time %06ld:%06ld:%06ld \n", 
			(long int)(cap_buf1_time/fps), (long int)(cap_buf1_max),
			(long int)(total_time/fps), (long int)(total_max),total_hit);
			cap_buf1_time = total_time = 0; 
			cap_buf1_max = total_max = total_hit = 0; 
		}else {
			printf("\n");
		}
	}
//	printf("cpy Buf 1 %06lu\n", (long int)(t1.tv_sec - t0.tv_sec)*1000000 + t1.tv_usec - t0.tv_usec);
//	gettimeofday(&t0, NULL);

	stream_count++;
	gettimeofday(&t1_end, NULL);
	temp_time = ((long int)(t1_end.tv_sec - t0_start.tv_sec)*1000000 + t1_end.tv_usec - t0_start.tv_usec);
	total_time += temp_time;
	total_max = (total_max < temp_time)?temp_time:total_max;
	total_hit = (temp_time > 16666)?(total_hit+1):total_hit;
	v4l2_mipi1_put_buffer(&v4l2_buffer1);

	return;
}
int print_usage(int argc, char **argv)
{
	printf ("%s Application usage\n", argv[0]);
	printf ("-iw = input width\n");
	printf ("-ih = input height\n");
	printf ("-ifmt = input format [UYVY, YUY2, JPEG, H264] \n");
	printf ("-imachine = Running machine type [x86, tegra]\n");
	return 0;
}

int parse_args (int argc, char** argv)
{
	int i;
	/* init GStreamer */
        for (i = 1; i < argc; i++) {
                if (strcmp(argv[i], "-iw") == 0) {
                        width = atoi(argv[++i]);
		}else if (strcmp(argv[i], "-ih") == 0) {
                        height = atoi(argv[++i]);
		}else if (strcmp(argv[i], "-ifmt") == 0) {
			i++;
			if (strcmp(argv[i], "UYVY") == 0) {
				fmt = UYVY;
			}else if (strcmp(argv[i], "YUY2") == 0) {
				fmt = YUY2;
			}else if (strcmp(argv[i], "JPEG") == 0) {
				fmt = JPEG;
			}else if (strcmp(argv[i], "H264") == 0) {
				fmt = H264;
			}else if (strcmp(argv[i], "BAYER") == 0) {
				fmt = BAYER;
			}else if (strcmp(argv[i], "RGB565") == 0) {
				fmt = RGB565;
			}
		}else if (strcmp(argv[i], "-imachine") == 0) {
			i++;
			if (strcmp(argv[i], "x86") == 0) {
				mach = X86;
			}else if (strcmp(argv[i], "tegra") == 0) {
				mach = TEGRA;
			}
		}else if (strcmp(argv[i], "--help") == 0) {
			print_usage(argc, argv);
		}
	}

	printf("Application selected information \n");
	printf("========================================================================= \n");
	printf("Width 	= %d \n", width);
	printf("Height	= %d \n", height);
	printf("fmt 	= %s \n", (fmt == UYVY)?"UYVY":
				(fmt == YUY2)?"YUY2":
				(fmt == JPEG)?"JPEG":
				(fmt == H264)?"H264":
				(fmt == BAYER)?"BAYER":
				(fmt == RGB565)?"RGB565":
				"UNKNOWN SELECTED");
	printf("Machine = %s \n", (mach == X86)?"x86":
				(mach == TEGRA)?"tegra":
				"UNKNOWN SELECTED");	
	printf("========================================================================= \n");
	return 0;
}

gint
main (gint argc, gchar *argv[])
{
	GstElement *pipeline, *appsrc;
	gchar *descr;

	GError *error = NULL;
	char gst_pipeline[1024] = {0};

	parse_args(argc, argv);
	argc =1;

	gst_init (&argc, &argv);
	loop = g_main_loop_new (NULL, FALSE);

	
	printf("Camera Requested dimention is %dx%d\n", width, height);

	/* create a new pipeline */
#if defined(GSTREAMER_1__0)
	if (mach == TEGRA) {
		descr = g_strdup_printf ("appsrc name=source ! nvvidconv ! \
			video/x-raw(memory:NVMM),width=%d,height=%d,format=I420,framerate=(fraction)22/1 ! \
			tee name=camconv ! nvhdmioverlaysink overlay=2 name=videosink async=false ", width, height);
	}else if (mach == X86) {
		sprintf(gst_pipeline, "appsrc name=source ! %s fpsdisplaysink name=videosink sync=false fps-update-interval=1000",
					fmt == JPEG?"jpegdec ! ":
					fmt == H264?"h264parse ! avdec_h264 ! ":"");
		descr = g_strdup_printf (gst_pipeline);
	}

//	camconv. omxh264enc iframeinterval=30 ! tee name=encode ! queue ! matroskamux ! filesink async=false location=sample.h264 \
//	encode. ! rtph264pay mtu=60000 ! udpsink clients=192.168.6.161:8000 async=false", width, height);

#elif defined(GSTREAMER_0__1_0)
	descr = g_strdup_printf ("appsrc name=source ! nvvidconv ! \
			video/x-nv-yuv, format=(fourcc)I420, width=(int)%d, height=(int)%d ! 			\
			tee name=camconv ! nv_omx_h264enc bitrate=10000000 rc-mode=cbr ! 			\
			rtph264pay mtu=60000 ! tee name=rtp264 ! udpsink clients=192.168.6.161:8000 async=false \
			camconv. ! nv_omx_hdmi_videosink overlay=2 name=videosink async=false rtp264. ! 	\
			rtph264depay ! filesink async=false location=sample.h264 ",width, height);
//	descr = g_strdup_printf ("appsrc name=source ! nvvidconv ! 
//	video/x-nv-yuv,width=%d,height=%d,format=(fourcc)I420 ! fakesink", width, height);

//	descr = g_strdup_printf ("appsrc name=source ! fakesink");
#endif
	pipeline = gst_parse_launch (descr, &error);
	if (error != NULL) {
		g_print ("could not construct pipeline: %s\n", error->message);
		g_clear_error (&error);
		exit (-1);
	}

	appsrc = gst_bin_get_by_name (GST_BIN (pipeline), "source");

#if defined(GSTREAMER_1__0)
	if (fmt == UYVY) {
		g_object_set (G_OBJECT (appsrc), "caps", 
			gst_caps_new_simple ("video/x-raw", 
						 "format", G_TYPE_STRING, "UYVY", 
						 "width", G_TYPE_INT, width, 
						 "height", G_TYPE_INT, height, 
						 "framerate", GST_TYPE_FRACTION, 0, 1, 
						 NULL), NULL);
	}else if (fmt == RGB565) {
		g_object_set (G_OBJECT (appsrc), "caps", 
			gst_caps_new_simple ("video/x-raw", 
						 "format", G_TYPE_STRING, "RGB16", 
						 "width", G_TYPE_INT, width, 
						 "height", G_TYPE_INT, height, 
						 "framerate", GST_TYPE_FRACTION, 0, 1, 
						 NULL), NULL);
	}else if (fmt == JPEG) {
		g_object_set (G_OBJECT (appsrc), "caps", 
			gst_caps_new_simple ("image/jpeg",
						 "width", G_TYPE_INT, width, 
						 "height", G_TYPE_INT, height, 
						 "framerate", GST_TYPE_FRACTION, 0, 1, 
						 NULL), NULL);
	}else if (fmt == H264) {
		g_object_set (G_OBJECT (appsrc), "caps", 
			gst_caps_new_simple ("video/x-h264",
						 "stream-format", G_TYPE_STRING, "byte-stream",
						 "width", G_TYPE_INT, width, 
						 "height", G_TYPE_INT, height, 
						 "framerate", GST_TYPE_FRACTION, 0, 1, 
						 NULL), NULL);
	}else if (fmt == YUY2) {
		g_object_set (G_OBJECT (appsrc), "caps", 
			gst_caps_new_simple ("video/x-raw", 
						 "format", G_TYPE_STRING, "YUY2", 
						 "width", G_TYPE_INT, width, 
						 "height", G_TYPE_INT, height, 
						 "framerate", GST_TYPE_FRACTION, 22, 1, 
						 NULL), NULL);
	}
#elif defined(GSTREAMER_0__1_0)
	gst_app_src_set_caps(GST_APP_SRC(appsrc), 
			gst_video_format_new_caps(GST_VIDEO_FORMAT_YUY2, width, height, 60, 1, 16, 9));
#endif

	/* setup v4l2 input src */
	set_cam1_width_height(width, height, fmt);
	v4l2_mipi1_src_init ();


	/* setup appsrc */
	g_object_set (G_OBJECT (appsrc), 
		"stream-type", 0, 
		"format", GST_FORMAT_TIME, NULL);
	g_signal_connect (appsrc, "need-data", G_CALLBACK (cb_need_data), NULL);

	/* play */
	gst_element_set_state (pipeline, GST_STATE_PLAYING);

	g_main_loop_run (loop);

	/* clean up */
	gst_element_set_state (pipeline, GST_STATE_NULL);
	gst_object_unref (GST_OBJECT (pipeline));
	g_main_loop_unref (loop);

	v4l2_mipi1_src_exit();

	return 0;
}
