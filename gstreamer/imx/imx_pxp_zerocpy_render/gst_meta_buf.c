#include "gst_meta_buf.h"

GType gst_imx_v4l2_meta_api_get_type(void)
{
	static volatile GType type;
	static const gchar *tags[] = {
		GST_META_TAG_VIDEO_STR, GST_META_TAG_MEMORY_STR, NULL
	};

	if (g_once_init_enter(&type))
	{
		GType _type = gst_meta_api_type_register("GstImxV4l2MetaAPI", tags);
		g_once_init_leave(&type, _type);
	}
	return type;
}


const GstMetaInfo *gst_imx_v4l2_meta_get_info(void)
{
	static const GstMetaInfo *meta_info = NULL;

	if (g_once_init_enter(&meta_info))
	{
		const GstMetaInfo *meta = gst_meta_register(
				gst_imx_v4l2_meta_api_get_type(),
				"GstImxV4l2Meta",
				sizeof(GstImxV4l2Meta),
				(GstMetaInitFunction)NULL,
				(GstMetaFreeFunction)NULL,
				(GstMetaTransformFunction)NULL);
		g_once_init_leave(&meta_info, meta);
	}
	return meta_info;
}
