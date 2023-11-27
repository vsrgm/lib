#include <stdio.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

#define NBUF 3

void query_capabilites(int fd)
{
    struct v4l2_capability cap;

    if (-1 == ioctl(fd, VIDIOC_QUERYCAP, &cap)) {
        perror("Query capabilites");
        exit(EXIT_FAILURE);
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "Device is no video capture device\\n");
        exit(EXIT_FAILURE);
    }

    if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
        fprintf(stderr, "Device does not support read i/o\\n");
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "Devices does not support streaming i/o\\n");
        exit(EXIT_FAILURE);
    }
}

int queue_buffer(int fd, int index) {
    struct v4l2_buffer bufd = {0};
    bufd.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufd.memory = V4L2_MEMORY_MMAP;
    bufd.index = index;
    if(-1 == ioctl(fd, VIDIOC_QBUF, &bufd))
    {
        perror("Queue Buffer");
        return 1;
    }
    return bufd.bytesused;
}
int dequeue_buffer(int fd) {
    struct v4l2_buffer bufd = {0};
    bufd.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bufd.memory = V4L2_MEMORY_MMAP;
    bufd.index = 0;
    if(-1 == ioctl(fd, VIDIOC_DQBUF, &bufd))
    {
        perror("DeQueue Buffer");
        return 1;
    }
    return bufd.index;
}


int start_streaming(int fd) {
    unsigned int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if(ioctl(fd, VIDIOC_STREAMON, &type) == -1){
        perror("VIDIOC_STREAMON");
        exit(EXIT_FAILURE);
    }
}

int stop_streaming(int fd) {
    unsigned int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if(ioctl(fd, VIDIOC_STREAMOFF, &type) == -1){
        perror("VIDIOC_STREAMON");
        exit(EXIT_FAILURE);
    }
}

int query_buffer(int fd, int index, unsigned char **buffer) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = index;
    int res = ioctl(fd, VIDIOC_QUERYBUF, &buf);
    if(res == -1) {
        perror("Could not query buffer");
        return 2;
    }


    *buffer = (u_int8_t*)mmap (NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    return buf.length;
}

int request_buffer(int fd, int count) {
    struct v4l2_requestbuffers req = {0};
    req.count = count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (-1 == ioctl(fd, VIDIOC_REQBUFS, &req))
    {
        perror("Requesting Buffer");
        exit(EXIT_FAILURE);
    }
    return req.count;
}

int set_format(int fd) {
    struct v4l2_format format = {0};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = 320;
    format.fmt.pix.height = 240;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    int res = ioctl(fd, VIDIOC_S_FMT, &format);
    if(res == -1) {
        perror("Could not set format");
        exit(EXIT_FAILURE);
    }
    return res;
}

int main() {
    unsigned char *buffer[NBUF];
    int fd = open("/dev/video0", O_RDWR);
    int size;
    int index;
    int nbufs;

    query_capabilites(fd);
    set_format(fd);
    nbufs = request_buffer(fd, NBUF);
    if ( nbufs > NBUF) {
        fprintf(stderr, "Increase NBUF to at least %i\n", nbufs);
        exit(1);

    }

    for (int i = 0; i < NBUF; i++) {

        /* Assume all sizes is equal.. */
        size = query_buffer(fd, 0, &buffer[0]);

        queue_buffer(fd, i);
    }

    start_streaming(fd);
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    struct timeval tv = {0};
    tv.tv_sec = 2;
    int r = select(fd+1, &fds, NULL, NULL, &tv);
    if(-1 == r){
        perror("Waiting for Frame");
        exit(1);
    }

    index = dequeue_buffer(fd);
    int file = open("output.raw", O_RDWR | O_CREAT, 0666);
    fprintf(stderr, "file == %i\n", file);
    write(file, buffer[index], size);

    stop_streaming(fd);

    close(file);
    close(fd);

    return 0;
}
