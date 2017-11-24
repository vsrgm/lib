#include <gst/gst.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#if defined (USE_CUDA_BASE)
	#include <cuda.h>
	#include <cuda_runtime.h>

	// helper functions and utilities to work with CUDA
	#include <helper_functions.h>
	#include <helper_cuda.h>
#elif defined (USE_ARM_NEON)
	#include <arm_neon.h>
#endif

#include <pthread.h>
#include <gst/video/video.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include "cam1_mipi.h"

#if defined(GSTREAMER_1__0)
GstSample *sample_mipi1;
GstMapInfo map_mipi1;
#endif

GstBuffer *buffer_mipi1;
int map_return_mipi1 = 0;
GstElement *pipeline_mipi1, *sink_mipi1, *appsrc_mipi1;
int ping_ping_mipi1 = 0;
int camera_fd_mipi1;
struct v4l2_buffer capture_buff_mipi1 = {0};
struct v4l2_buffer capture_buff_last_mipi1 = {0};

enum _fmt_mode {
	UYVY = 0,
	YUY2,
	RGB565,
	BAYER,
	JPEG,
	H264,
};

int cam1_width = 1280, cam1_height = 720, cam1_fmt = JPEG;

void 
set_cam1_width_height(int width, int height, int fmt)
{
	cam1_width = width;
	cam1_height = height;
	cam1_fmt = fmt;
	return;
}
void
v4l2_mipi1_src_exit(void) 
{
	/* cleanup and exit */
	gst_element_set_state (pipeline_mipi1, GST_STATE_NULL);
	gst_object_unref (pipeline_mipi1);
}

int
v4l2_mipi1_put_buffer(void) {
#if defined(GSTREAMER_1__0)
	if (sample_mipi1) {
		if (map_return_mipi1) {
			gst_buffer_unmap (buffer_mipi1, &map_mipi1);
			map_return_mipi1 = 0;
		}
		gst_sample_unref (sample_mipi1);
		sample_mipi1 = NULL;
	}

#elif defined(GSTREAMER_0__1_0)
	gst_buffer_unref (buffer_mipi1);
	buffer_mipi1 = NULL;
#endif

	return 0;
}
	
int 
v4l2_mipi1_get_buffer(unsigned char **buf) {
#if defined(GSTREAMER_1__0)
	gboolean res;
	gint width, height;
#endif

#if defined(GSTREAMER_1__0)
	/* get the preroll buffer from appsink, this block untils appsink really
	 * prerolls */
	g_signal_emit_by_name (sink_mipi1, "pull-sample", &sample_mipi1, NULL);

	/* if we have a buffer now, convert it to a pixbuf. It's possible that we
	 * don't have a buffer because we went EOS right away or had an error. */
	if (sample_mipi1) {
		GstCaps *caps;
		GstStructure *s;

		/* get the snapshot buffer format now. We set the caps on the appsink so
		 * that it can only be an rgb buffer. The only thing we have not specified
		 * on the caps is the height, which is dependant on the pixel-aspect-ratio
		 * of the source material */
		caps = gst_sample_get_caps (sample_mipi1);
		if (!caps) {
			g_print ("could not get snapshot format\n");
			exit (-1);
		}
		s = gst_caps_get_structure (caps, 0);

		/* we need to get the final caps on the buffer to get the size */
		res = gst_structure_get_int (s, "width", &width);
		res |= gst_structure_get_int (s, "height", &height);
		if (!res) {
			g_print ("could not get snapshot dimension\n");
			exit (-1);
		}
		buffer_mipi1 = gst_sample_get_buffer (sample_mipi1);
		/* Mapping a buffer can fail (non-readable) */
		map_return_mipi1 = gst_buffer_map (buffer_mipi1, &map_mipi1, GST_MAP_READ);

	}

#elif defined(GSTREAMER_0__1_0)
	g_signal_emit_by_name (sink_mipi1, "pull-buffer", &buffer_mipi1);
#endif

#if defined(GSTREAMER_1__0)
	*buf = map_mipi1.data;
#elif defined(GSTREAMER_0__1_0)
	*buf = GST_BUFFER_DATA(buffer_mipi1);
#endif
	return 0;
}

static void
cb_need_data_mipi1 (GstElement *appsrc,
				guint unused_size,
				gpointer user_data)
{
	static GstClockTime timestamp = 0;
	static GstBuffer *cam_buffer;
	GstFlowReturn ret;
	guint size;
	gpointer mem;
	static int queue_last_frame = -1;

	size = cam1_width * cam1_height * 2;

	capture_buff_mipi1.type	 = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	capture_buff_mipi1.memory = V4L2_MEMORY_MMAP;
	if(dequeue_buffer_mipi1(camera_fd_mipi1, &capture_buff_mipi1) == -1) {
		printf("cannot deqeue\n");
	}
	mem = get_frame_virt_mipi1(&capture_buff_mipi1);

	cam_buffer = gst_buffer_new();

 
#if defined(GSTREAMER_1__0)
	gst_buffer_remove_all_memory(cam_buffer);
	gst_buffer_append_memory(cam_buffer,
		gst_memory_new_wrapped(0,
			mem, size, 0,
			size, NULL, NULL));
	GST_BUFFER_PTS (cam_buffer) = timestamp;

#elif defined(GSTREAMER_0__1_0)
	GST_BUFFER_DATA(cam_buffer) = (guint8 *)mem;
	GST_BUFFER_SIZE(cam_buffer) = size;
#endif

//	GST_BUFFER_DURATION (cam_buffer) = gst_util_uint64_scale_int (1, GST_SECOND, 2);

	timestamp += GST_BUFFER_DURATION (cam_buffer);

	g_signal_emit_by_name (appsrc, "push-buffer", cam_buffer, &ret);

	if (ret != GST_FLOW_OK) {
		/* something wrong, stop pushing */
//		g_main_loop_quit (loop);
	}

	gst_buffer_unref(cam_buffer);

	if (queue_last_frame != -1) {
		if(enqueue_buffer_mipi1(camera_fd_mipi1, &capture_buff_last_mipi1) == -1) {
			printf("cannot enqueue\n");
		}
		capture_buff_last_mipi1 = capture_buff_mipi1;
	}else {
		queue_last_frame = capture_buff_mipi1.index;
		capture_buff_last_mipi1 = capture_buff_mipi1;
	}

}

int
v4l2_mipi1_src_init (void)
{
	gchar *descr;
	GError *error = NULL;
#if defined(GSTREAMER_1__0)
	gint64 duration, position;
#endif
	GstStateChangeReturn ret;

	/* create a new pipeline */
#if defined(GSTREAMER_1__0)
	descr =	g_strdup_printf ("appsrc name=mipi1_src ! " \
					" appsink name=mipi1_sink drop=true async=false");
#elif defined(GSTREAMER_0__1_0)
	descr =	g_strdup_printf ("appsrc name=mipi1_src ! " \
					" appsink name=mipi1_sink drop=true async=false");
#endif
	pipeline_mipi1 = gst_parse_launch (descr, &error);

	if (error != NULL) {
		g_print ("could not construct pipeline: %s\n", error->message);
		g_clear_error (&error);
		exit (-1);
	}

	/* Init Camera */
	camera_fd_mipi1 = init_camera_mipi1((char *)"/dev/video0", cam1_width, cam1_height,
	 					(cam1_fmt == YUY2)?V4L2_PIX_FMT_YUYV:
						(cam1_fmt == UYVY)?V4L2_PIX_FMT_UYVY:
						(cam1_fmt == RGB565)?V4L2_PIX_FMT_RGB565:
						(cam1_fmt == JPEG)?V4L2_PIX_FMT_MJPEG:
						(cam1_fmt == H264)?V4L2_PIX_FMT_H264:
						V4L2_PIX_FMT_MJPEG);							

	/* get sink */
	sink_mipi1 = gst_bin_get_by_name (GST_BIN (pipeline_mipi1), "mipi1_sink");
	appsrc_mipi1 = gst_bin_get_by_name (GST_BIN (pipeline_mipi1), "mipi1_src");

#if defined(GSTREAMER_1__0)
	if (cam1_fmt == UYVY) {
		g_object_set (G_OBJECT (appsrc_mipi1), "caps", 
			gst_caps_new_simple ("video/x-raw", 
						 "format", G_TYPE_STRING, "UYVY", 
						 "width", G_TYPE_INT, cam1_width, 
						 "height", G_TYPE_INT, cam1_height, 
						 "framerate", GST_TYPE_FRACTION, 0, 1, 
						 NULL), NULL);
	}else if (cam1_fmt == RGB565) {
		g_object_set (G_OBJECT (appsrc_mipi1), "caps", 
			gst_caps_new_simple ("video/x-raw", 
						 "format", G_TYPE_STRING, "RGB16", 
						 "width", G_TYPE_INT, cam1_width, 
						 "height", G_TYPE_INT, cam1_height, 
						 "framerate", GST_TYPE_FRACTION, 0, 1, 
						 NULL), NULL);
	}else if (cam1_fmt == JPEG) {
		g_object_set (G_OBJECT (appsrc_mipi1), "caps", 
			gst_caps_new_simple ("image/jpeg",
						 "width", G_TYPE_INT, cam1_width, 
						 "height", G_TYPE_INT, cam1_height, 
						 "framerate", GST_TYPE_FRACTION, 0, 1, 
						 NULL), NULL);
	}else if (cam1_fmt == H264) {
		g_object_set (G_OBJECT (appsrc_mipi1), "caps", 
			gst_caps_new_simple ("video/x-h264",
						 "stream-format", G_TYPE_STRING, "byte-stream",
						 "width", G_TYPE_INT, cam1_width, 
						 "height", G_TYPE_INT, cam1_height, 
						 "framerate", GST_TYPE_FRACTION, 0, 1, 
						 NULL), NULL);
	}else if (cam1_fmt == YUY2) {
		g_object_set (G_OBJECT (appsrc_mipi1), "caps", 
			gst_caps_new_simple ("video/x-raw", 
						 "format", G_TYPE_STRING, "YUY2", 
						 "width", G_TYPE_INT, cam1_width, 
						 "height", G_TYPE_INT, cam1_height, 
						 "framerate", GST_TYPE_FRACTION, 22, 1, 
						 NULL), NULL);
	}
	/* setup appsrc */
	g_object_set (G_OBJECT (appsrc_mipi1),
		"stream-type", 0,
		"format", GST_FORMAT_TIME, NULL);
#elif defined(GSTREAMER_0__1_0)
	gst_app_src_set_caps(GST_APP_SRC(appsrc_mipi1), 
				gst_video_format_new_caps(GST_VIDEO_FORMAT_YUY2, cam1_width, cam1_height, 60, 1, 16, 9));
#endif
	g_signal_connect (appsrc_mipi1, "need-data", G_CALLBACK (cb_need_data_mipi1), NULL);


	/* set to PAUSED to make the first frame arrive in the sink */
	ret = gst_element_set_state (pipeline_mipi1, GST_STATE_PLAYING);
	switch (ret) {
		case GST_STATE_CHANGE_FAILURE:
			g_print ("failed to play the file\n");
			exit (-1);
		case GST_STATE_CHANGE_NO_PREROLL:
			/* for live sources, we need to set the pipeline to PLAYING before we can
			 * receive a buffer. We don't do that yet */
			g_print ("live sources not supported yet\n");
			exit (-1);
		default:
			break;
	}
	/* This can block for up to 5 seconds. If your machine is really overloaded,
	 * it might time out before the pipeline prerolled and we generate an error. A
	 * better way is to run a mainloop and catch errors there. */
	ret = gst_element_get_state (pipeline_mipi1, NULL, NULL, 5 * GST_SECOND);
	if (ret == GST_STATE_CHANGE_FAILURE) {
		g_print ("failed to play the file \n");
		exit (-1);
	}
#if defined(GSTREAMER_1__0)
	/* get the duration */
	gst_element_query_duration (pipeline_mipi1, GST_FORMAT_TIME, &duration);

	if (duration != -1)
		/* we have a duration, seek to 5% */
		position = duration * 5 / 100;
	else
		/* no duration, seek to 1 second, this could EOS */
		position = 1 * GST_SECOND;

	/* seek to the a position in the file. Most files have a black first frame so
	 * by seeking to somewhere else we have a bigger chance of getting something
	 * more interesting. An optimisation would be to detect black images and then
	 * seek a little more */
	gst_element_seek_simple (pipeline_mipi1, GST_FORMAT_TIME,
			GST_SEEK_FLAG_KEY_UNIT | GST_SEEK_FLAG_FLUSH, position);
#endif
	return 0;	
}
