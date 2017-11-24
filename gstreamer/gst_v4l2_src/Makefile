SYSROOT:=/
CC:=${CROSS_COMPILE}gcc
PKG_CONFIG_PATH=${SYSROOT}/usr/lib/pkgconfig/
FLAGS_0_1_0:=`pkg-config --cflags --libs gstreamer-0.10 gstreamer-app-0.10 gstreamer-video-0.10` 
FLAGS_1_0:=`pkg-config --cflags --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0` 

SRC=$(wildcard *.cu)
OBJS=$(join $(addsuffix , $(dir $(SRC))), $(notdir $(SRC:.cu=.o)))

CFLAGS:=

all : gst_v4l2_src_0.10.elf gst_v4l2_src_1.0.elf
	@echo "Done"

#For Neon code -DUSE_ARM_NEON -mfloat-abi=hard -mfpu=neon
gst_v4l2_src_0.10.elf: appsrc.c app_mipi1_sink.c  cam1_mipi.c 
	${CC}  -g --sysroot=${SYSROOT} -DGSTREAMER_0__1_0 -I/usr/lib/gcc/arm-linux-gnueabihf/4.8/include/ -I/usr/include/SDL/ -o $@ $+ ${FLAGS_0_1_0} ${CFLAGS} -lpthread
	@echo "Done."

gst_v4l2_src_1.0.elf: appsrc.c app_mipi1_sink.c  cam1_mipi.c
	${CC}  -g --sysroot=${SYSROOT} -DGSTREAMER_1__0 -I/usr/lib/gcc/arm-linux-gnueabihf/4.8/include/  -I/usr/include/SDL/ -o $@ $+ ${FLAGS_1_0} ${CFLAGS} -lpthread
	@echo "Done."

clean:
	@echo "cleaning .."
	@rm -f *.elf
