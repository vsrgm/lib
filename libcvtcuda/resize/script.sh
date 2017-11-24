#!/bin/sh
while [ 1 ]
do
	gst-launch-0.10 filesrc location=file_sample.raw blocksize=3110400  ! "video/x-raw-yuv, format=(fourcc)I420, width=(int)1920, height=(int)1080, framerate=(fraction)60/1" ! nv_omx_hdmi_videosink overlay-x=860 overlay-y=100 overlay-w=960 overlay-h=1080 overlay=2  filesrc location=file_sample1.raw blocksize=3110400 ! "video/x-raw-yuv, format=(fourcc)I420, width=(int)1920, height=(int)1080, framerate=(fraction)60/1" ! nv_omx_hdmi_videosink overlay-w=860 overlay-h=1080 overlay-y=100;	date +%H:%M:%S.%N
done

