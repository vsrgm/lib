SYSROOT:=/
CC:=${CROSS_COMPILE}gcc
PKG_CONFIG_PATH=${SYSROOT}/usr/lib/pkgconfig/
FLAGS:=`pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0` 

CFLAGS:=

appsrc: appsrc.c
	@echo "building streamer in push mode . "
	${CC}  -g --sysroot=${SYSROOT} -mfloat-abi=hard -mfpu=neon -DPUSH_MODE -o streamer1.0 $+ ${FLAGS} ${CFLAGS}
	@echo "Done."

clean:
	@echo "cleaning .."
	@rm -f streamer
	@echo "removed streamer"
