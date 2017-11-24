#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "v4l2.h"
#include "fn_prototype.h"
#include "gst_meta_buf.h"

#define MAX_BUF_SIZE 2048


static GMainLoop *loop;
pthread_mutex_t buf_lock = PTHREAD_MUTEX_INITIALIZER;

static void cb_need_data (GstElement *appsrc,
		guint unused_size,
		gpointer user_data)
{
	static GstClockTime timestamp = 0;
	GstBuffer *buffer;
	GstImxV4l2Meta *meta;
	GstImxPhysMemMeta *phys_mem_meta;

	guint size;
	GstFlowReturn ret;
	struct cam_info *cam = user_data;
	char* mem, *pmem;
	static struct timeval cur_tv = {0};
	static struct timeval prev_tv = {0};
	int fps = 0;
	static int old_stream_count = 0, stream_count = 0;
	static struct v4l2_buffer cambuffer[6];
	int i;

	if (!cam) {
		printf("%s %d : User data is NULL\n", __func__, __LINE__);
		return;
	}
	size = cam->width * cam->height * 2;

	buffer = gst_buffer_new();
	meta = GST_IMX_V4L2_META_ADD(buffer);
	meta->vbuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	meta->vbuffer.memory = V4L2_MEMORY_MMAP;

	if(dequeue_buffer(cam->fd, &meta->vbuffer) == -1) {
		printf("cannot deqeue\n");
	}
	mem = get_frame_virt(cam, &meta->vbuffer);

	/*buffer ready to be processed */
	phys_mem_meta = GST_IMX_PHYS_MEM_META_ADD(buffer);
	phys_mem_meta->phys_addr = get_frame_phy(cam, &meta->vbuffer);
	meta->vbuffer.length = size;//get_frame_lenght(&meta->vbuffer);

	gst_buffer_remove_all_memory(buffer);
	gst_buffer_append_memory(buffer,
			gst_memory_new_wrapped(0,
				mem, size, 0,
				size, NULL, NULL));
	GST_BUFFER_PTS (buffer) = timestamp;
        //GST_BUFFER_TIMESTAMP(buffer) = GST_TIMEVAL_TO_TIME(meta->vbuffer.timestamp);

	//GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale_int (1, GST_SECOND, 30);
	timestamp += GST_BUFFER_DURATION (buffer);

	g_signal_emit_by_name (appsrc, "push-buffer", buffer, &ret);

	if (ret != GST_FLOW_OK) {
		/* something wrong, stop pushing */
		g_main_loop_quit (loop);
	}
	
	if(enqueue_buffer(cam->fd, &meta->vbuffer) == -1) {
		printf("cannot enqueue\n");
	}

	gst_buffer_unref(buffer);

	gettimeofday(&cur_tv, NULL);
	if(cur_tv.tv_sec > prev_tv.tv_sec)
	{
		prev_tv	= cur_tv;
		fps = stream_count - old_stream_count;
		old_stream_count = stream_count;
		printf("Rendering Frame rate %d \n",fps);
	}
	stream_count++;
	return ;
}

gint main (gint argc, gchar *argv[])
{
	GstElement *pipeline;
	gchar pipeline_str[MAX_BUF_SIZE] = {0};
	gchar *descr;
	GError *error = NULL;
	GstElement *camsrc[MAX_CAM], *videosink;
	int cam_id;
	char id_name[MAX_CAM_LEN];
	struct cam_info cam[MAX_CAM] = {0};
	GstStateChangeReturn ret;
	gint64 duration, position;

	/* init GStreamer */
	gst_init (&argc, &argv);
	loop = g_main_loop_new (NULL, FALSE);
	/* setup pipeline */

	sprintf(pipeline_str, "appsrc name=cam0 ! imxpxpvideosink name=sink use-vsync=true");

	descr = g_strdup_printf (pipeline_str);
	pipeline = gst_parse_launch (descr, &error);
	if (error != NULL) {
		g_print ("could not construct pipeline: %s\n", error->message);
		g_clear_error (&error);
	}

	/* setup */
	cam_id = 0;
	memset(id_name, 0x00, sizeof(id_name));
	sprintf(id_name, "cam%d", cam_id);

	camsrc[cam_id] = gst_bin_get_by_name (GST_BIN (pipeline), id_name);
	g_object_set (G_OBJECT (camsrc[cam_id]), "caps",
			gst_caps_new_simple ("video/x-raw",
				"format", G_TYPE_STRING, "YUY2",
				"width", G_TYPE_INT, CAM_WIDTH,
				"height", G_TYPE_INT, CAM_HEIGHT,
				"framerate", GST_TYPE_FRACTION, 30, 1,
				NULL), NULL);

	cam[cam_id].width = CAM_WIDTH;
	cam[cam_id].height = CAM_HEIGHT;

	/* setup appsrc */
	g_object_set (G_OBJECT (camsrc[cam_id]),
			"stream-type", 0,
			"format", GST_FORMAT_TIME,
			"is-live", TRUE,
			NULL);

	memset(id_name, 0x00, sizeof(id_name));
	sprintf(id_name, "/dev/video1");

	cam[cam_id].fd = init_cam(id_name, cam[cam_id].buf, cam[cam_id].width, cam[cam_id].height);
	if (cam[cam_id].fd < 0) {
		printf("Failed to start camera \n");
	}

	g_signal_connect (camsrc[cam_id], "need-data", G_CALLBACK (cb_need_data), &cam[cam_id]);

	/* play */
	gst_element_set_state (pipeline, GST_STATE_PLAYING);
	ret = gst_element_get_state (pipeline, NULL, NULL, 1 * GST_SECOND);
	if (ret == GST_STATE_CHANGE_FAILURE) {
		g_print ("failed to play the file \n");
	}

	g_main_loop_run (loop);

	/* clean up */
	gst_element_set_state (pipeline, GST_STATE_NULL);
	gst_object_unref (GST_OBJECT (pipeline));
	g_main_loop_unref (loop);
	return 0;
}
