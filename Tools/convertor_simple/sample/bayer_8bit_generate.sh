gst-launch-1.0 -v videotestsrc num-buffers=1 ! video/x-raw,format=BGRA,width=800,height=600,framerate=1/1 ! videoconvert ! video/x-raw,format=ARGB,framerate=1/1 ! rgb2bayer ! filesink location=out.bggr
gst-launch-1.0 -v videotestsrc num-buffers=1 ! video/x-raw,format=BGRA,width=800,height=600,framerate=1/1 ! videoconvert ! jpegenc ! filesink location=out.jpg
gst-launch-1.0 -v filesrc location=out.bggr ! bayer2rgb ! filesink location=out1.bggr
gst-launch-1.0 -v filesrc location=out.bggr ! bayer2rgb ! video/x-raw,format=BGRA,width=800,height=600,framerate=1/1 ! videoconvert ! jpegenc ! filesink location=out.jpg
