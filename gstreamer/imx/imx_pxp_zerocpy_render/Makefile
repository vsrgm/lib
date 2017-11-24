# -mfloat-abi=hard -mfpu=neon
SYSROOT:=/
hexcam.elf: main.c  v4l2.c gst_meta_buf.c
	${CC}  -g --sysroot=${SYSROOT} -DGSTREAMER_1__0 -o $@ $+ ${FLAGS_1_0} ${CFLAGS} `pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0` -lpthread -lgstimxcommon
