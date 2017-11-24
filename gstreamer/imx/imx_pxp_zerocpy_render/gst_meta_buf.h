#ifndef __GST_META_BUF_H__
#define __GST_META_BUF_H__

#include <linux/videodev2.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <gst/video/gstvideometa.h>

typedef unsigned long gst_imx_phys_addr_t;
typedef struct _GstImxV4l2Meta GstImxV4l2Meta;
typedef struct _GstImxPhysMemMeta GstImxPhysMemMeta;

GType gst_imx_v4l2_meta_api_get_type (void);
const GstMetaInfo * gst_imx_v4l2_meta_get_info (void);
GType gst_imx_phys_mem_meta_api_get_type(void);
GstMetaInfo const * gst_imx_phys_mem_meta_get_info(void);
#define GST_IMX_V4L2_META_GET(buf) ((GstImxV4l2Meta *)gst_buffer_get_meta(buf,gst_imx_v4l2_meta_api_get_type()))
#define GST_IMX_V4L2_META_ADD(buf) ((GstImxV4l2Meta *)gst_buffer_add_meta(buf,gst_imx_v4l2_meta_get_info(),NULL))
#define GST_IMX_PHYS_MEM_META_GET(buffer)      ((GstImxPhysMemMeta *)gst_buffer_get_meta((buffer), gst_imx_phys_mem_meta_api_get_type()))
#define GST_IMX_PHYS_MEM_META_ADD(buffer)      ((GstImxPhysMemMeta *)gst_buffer_add_meta((buffer), gst_imx_phys_mem_meta_get_info(), NULL))


struct _GstImxV4l2Meta {
	GstMeta meta;
	gpointer mem;
	struct v4l2_buffer vbuffer;
};

struct _GstImxPhysMemMeta
{
	GstMeta meta;
	gst_imx_phys_addr_t phys_addr;
	gsize x_padding, y_padding;
	GstBuffer *parent;
};

#endif
