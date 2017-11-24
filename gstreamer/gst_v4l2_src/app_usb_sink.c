#include <gst/gst.h>
#include <stdio.h>
#include <stdlib.h>

GstSample *sample_mipi1;
GstBuffer *buffer_mipi1;
GstMapInfo map_mipi1;
int map_return_mipi1 = 0;
GstElement *pipeline_mipi1, *sink_mipi1;

int
v4l2_mipi_src1_exit(void) 
{
  /* cleanup and exit */
  gst_element_set_state (pipeline_mipi1, GST_STATE_NULL);
  gst_object_unref (pipeline_mipi1);
}

int 
v4l2_mipi_get1_buffer(unsigned char **buf) {
  gboolean res;
  gint width, height;

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
  }

   buffer_mipi1 = gst_sample_get_buffer (sample_mipi1);
    /* Mapping a buffer can fail (non-readable) */
    if (map_return_mipi1 = gst_buffer_map (buffer_mipi1, &map_mipi1, GST_MAP_READ)) {
      *buf = map_mipi1.data;
    }
  return 0;
}

int
v4l2_mipi_put1_buffer(void) {
	if (sample_mipi1) {
		if (map_return_mipi1) {
		    gst_buffer_unmap (buffer_mipi1, &map_mipi1);
		    map_return_mipi1 = 0;
		}
	    gst_sample_unref (sample_mipi1);
	    sample_mipi1 = NULL;
	}
}

int
v4l2_mipi_src1_init ()
{
  gchar *descr;
  GError *error = NULL;
  gint64 duration, position;
  GstStateChangeReturn ret;
  gboolean res;

  /* create a new pipeline */
  descr =
      g_strdup_printf ("v4l2src device=/dev/video1 ! "
      " image/jpeg, width=(int)1280, height=(int)720, interlaced=(boolean)false, pixel-aspect-ratio=(fraction)1/1 ! "
      "jpegdec !  video/x-raw,format=I420,width=1280,height=720 ! videoscale add-borders=false ! video/x-raw,format=I420,width=960,height=1080 !"
      " appsink name=sink drop=true max-buffers=1 ");
  pipeline_mipi1 = gst_parse_launch (descr, &error);

  if (error != NULL) {
    g_print ("could not construct pipeline: %s\n", error->message);
    g_clear_error (&error);
    exit (-1);
  }

  /* get sink */
  sink_mipi1 = gst_bin_get_by_name (GST_BIN (pipeline_mipi1), "sink");

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
    g_print ("failed to play the file\n");
    exit (-1);
  }

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

  return 0;
}
