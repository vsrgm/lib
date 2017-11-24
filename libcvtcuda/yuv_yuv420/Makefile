SYSROOT:=/
CROSS_COMPILE:= 
CC:=${CROSS_COMPILE}gcc
PKG_CONFIG_PATH=${SYSROOT}/usr/lib/pkgconfig/
FLAGS:=`pkg-config --cflags --libs gstreamer-0.10 gstreamer-video-0.10 gstreamer-app-0.10` 

convert_yuv_yuv420.elf: convert_yuv_yuv420.o
	/usr/local/cuda-6.5/bin/nvcc -rdc=true -ccbin g++  -Xcompiler -mfloat-abi=hard  -Xlinker --dynamic-linker=/lib/ld-linux-armhf.so.3  -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 $< ../libcvtcuda.a -o $@  
#-I/usr/include/gstreamer-0.10 -I/usr/include/glib-2.0 -I/usr/lib/arm-linux-gnueabihf/glib-2.0/include -I/usr/include/libxml2  -lgstreamer-0.10 -lgobject-2.0 -lgmodule-2.0 -lgthread-2.0 -lxml2 -lglib-2.0  -lgstapp-0.10 -lgstvideo-0.10 -lpthread
	@echo "Done."

%.o : %.cu
	/usr/local/cuda-6.5/bin/nvcc -rdc=true -ccbin g++ -I/home/ubuntu/NVIDIA_CUDA-6.5_Samples/common/inc  -m32   -Xcompiler -mfloat-abi=hard  -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -c $< -o $@

clean:
	@echo "cleaning .."
	@rm -f *.elf
