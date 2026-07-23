#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#define SW_VERSION "1.0.304"

#define PORT 5001
#define VIDEO_CHUNK_SIZE 1400
#define LOG_FILE "/tmp/server.log"
#define DUMP_FILE "/tmp/stream_data.h264"
#define VIDEO_DEVICE "/dev/video0"
#define BUFFER_COUNT 4

int stop_requested = 0;
pid_t stream_pid = -1;

struct buffer {
    void   *start;
    size_t  length;
};

void log_message(const char *tag, const char *message) {
    FILE *fp = fopen(LOG_FILE, "a");
    if (!fp) return;
    time_t now; time(&now);
    char *date = ctime(&now);
    date[strlen(date) - 1] = '\0';
    fprintf(fp, "[%s] [%s] %s\n", date, tag, message);
    printf("[%s] %s\n", tag, message);
    fclose(fp);
}

static int xioctl(int fh, int request, void *arg) {
    int r;
    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}

void cleanup_processes() {
    log_message("SERVER", "Cleaning up camera hardware...");
    if (stream_pid > 0) {
        log_message("SERVER", "Killing previous stream process...");
        kill(stream_pid, SIGTERM);
        usleep(500000);
        kill(stream_pid, SIGKILL);
        stream_pid = -1;
    }
    system("pkill -9 raspivid 2>/dev/null");
    system("pkill -9 libcamera-vid 2>/dev/null");
    sleep(1);
}

void start_manual_data_pump(const char *target_ip, int target_port, int use_mjpeg, int fps, int width, int height) {
    int sock, stats_sock;
    struct sockaddr_in server_addr, stats_addr;
    long total_bytes_sent = 0;
    int video_fd = -1;
    struct buffer *buffers = NULL;
    time_t last_stats_sent = 0;

    log_message("PUMP", use_mjpeg ? "Starting MJPEG Data Pump..." : "Starting H264 Data Pump...");

    // ... existing socket setup ...
    if ((sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        log_message("ERROR", "Video socket creation failed");
        return;
    }
    if ((stats_sock = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        log_message("ERROR", "Stats socket creation failed");
        close(sock);
        return;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(target_port);
    server_addr.sin_addr.s_addr = inet_addr(target_ip);

    memset(&stats_addr, 0, sizeof(stats_addr));
    stats_addr.sin_family = AF_INET;
    stats_addr.sin_port = htons(target_port + 1);
    stats_addr.sin_addr.s_addr = inet_addr(target_ip);

    // 2. Open Video Device
    video_fd = open(VIDEO_DEVICE, O_RDWR | O_NONBLOCK, 0);
    if (video_fd < 0) {
        log_message("ERROR", "Cannot open /dev/video0.");
        close(sock); close(stats_sock);
        return;
    }

    // 3. Set Format
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width > 0 ? width : 640;
    fmt.fmt.pix.height = height > 0 ? height : 480;
    fmt.fmt.pix.pixelformat = use_mjpeg ? V4L2_PIX_FMT_MJPEG : V4L2_PIX_FMT_H264;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(video_fd, VIDIOC_S_FMT, &fmt) < 0) {
        log_message("ERROR", "Failed to set pixel format.");
        goto cleanup;
    }

    // 3.5 Set Frame Rate
    struct v4l2_streamparm parm = {0};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps > 0 ? fps : 20;
    xioctl(video_fd, VIDIOC_S_PARM, &parm);

    // 4. Set Camera Controls
    if (use_mjpeg) {
        struct v4l2_control ctrl = {V4L2_CID_JPEG_COMPRESSION_QUALITY, 50};
        xioctl(video_fd, VIDIOC_S_CTRL, &ctrl);
    } else {
        struct v4l2_ext_controls ctrls = {0};
        struct v4l2_ext_control ctrl[4] = {0};
        ctrl[0].id = V4L2_CID_MPEG_VIDEO_BITRATE;
        ctrl[0].value = 2000000;
        ctrl[1].id = V4L2_CID_MPEG_VIDEO_GOP_SIZE;
        ctrl[1].value = 30;
        ctrl[2].id = V4L2_CID_MPEG_VIDEO_REPEAT_SEQ_HEADER;
        ctrl[2].value = 1;
        ctrl[3].id = V4L2_CID_MPEG_VIDEO_H264_PROFILE;
        ctrl[3].value = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
        ctrls.count = 4;
        ctrls.ctrl_class = V4L2_CTRL_CLASS_MPEG;
        ctrls.controls = ctrl;
        xioctl(video_fd, VIDIOC_S_EXT_CTRLS, &ctrls);
    }

    struct v4l2_control rot_ctrl = {V4L2_CID_ROTATE, 270};
    xioctl(video_fd, VIDIOC_S_CTRL, &rot_ctrl);

    // ... rest of setup ...

    // 5. Request and Map Buffers
    struct v4l2_requestbuffers req = {0};
    req.count = BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(video_fd, VIDIOC_REQBUFS, &req) < 0) {
        log_message("ERROR", "Buffer request failed");
        goto cleanup;
    }

    buffers = calloc(req.count, sizeof(*buffers));
    for (int i = 0; i < req.count; ++i) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (xioctl(video_fd, VIDIOC_QUERYBUF, &buf) < 0) {
            log_message("ERROR", "Query buffer failed");
            goto cleanup;
        }
        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, video_fd, buf.m.offset);
        if (xioctl(video_fd, VIDIOC_QBUF, &buf) < 0) {
            log_message("ERROR", "Queue buffer failed");
            goto cleanup;
        }
    }

    // 6. Start Streaming
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(video_fd, VIDIOC_STREAMON, &type) < 0) {
        log_message("ERROR", "Stream on failed");
        goto cleanup;
    }

    // 7. Data Loop
    // FILE *dump_fp = fopen(DUMP_FILE, "wb");
    log_message("PUMP", "V4L2 Stream Active. Sending Video data...");

    unsigned int frame_counter = 0;
    while (!stop_requested) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (xioctl(video_fd, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) {
                usleep(10000);
                continue;
            }
            log_message("ERROR", "Dequeue buffer failed");
            break;
        }

        unsigned char *data = (unsigned char *)buffers[buf.index].start;
        size_t size = buf.bytesused;

        // Send to phone in chunks with sequence headers
        size_t sent = 0;
        unsigned short packet_idx = 0;
        unsigned short frame_id = (unsigned short)(frame_counter & 0xFFFF);

        while (sent < size) {
            size_t to_send = (size - sent > VIDEO_CHUNK_SIZE) ? VIDEO_CHUNK_SIZE : (size - sent);

            // 6-byte header: [0x55 0xAA] [FrameID(2)] [PacketIdx(2)]
            unsigned char packet_buf[VIDEO_CHUNK_SIZE + 6];
            packet_buf[0] = 0x55;
            packet_buf[1] = 0xAA;
            packet_buf[2] = (frame_id >> 8) & 0xFF;
            packet_buf[3] = frame_id & 0xFF;
            packet_buf[4] = (packet_idx >> 8) & 0xFF;
            packet_buf[5] = packet_idx & 0xFF;

            memcpy(packet_buf + 6, data + sent, to_send);
            sendto(sock, packet_buf, to_send + 6, 0, (struct sockaddr *)&server_addr, sizeof(server_addr));

            sent += to_send;
            packet_idx++;
        }

        frame_counter++;
        total_bytes_sent += size;

        // Send stats to phone every second
        time_t now = time(NULL);
        if (now > last_stats_sent) {
            char stats_pkt[64];
            snprintf(stats_pkt, sizeof(stats_pkt), "SENT:%ld", total_bytes_sent);
            sendto(stats_sock, stats_pkt, strlen(stats_pkt), 0, (struct sockaddr *)&stats_addr, sizeof(stats_addr));
            last_stats_sent = now;

            char stats_log[64];
            snprintf(stats_log, sizeof(stats_log), "Total Sent: %ld bytes", total_bytes_sent);
            log_message("STATS", stats_log);
        }

        if (xioctl(video_fd, VIDIOC_QBUF, &buf) < 0) {
            log_message("ERROR", "Re-queue buffer failed");
            break;
        }
    }

    xioctl(video_fd, VIDIOC_STREAMOFF, &type);
    // if (dump_fp) fclose(dump_fp);

cleanup:
    if (buffers) {
        for (int i = 0; i < BUFFER_COUNT; ++i) {
            if (buffers[i].start) munmap(buffers[i].start, buffers[i].length);
        }
        free(buffers);
    }
    if (video_fd >= 0) close(video_fd);
    close(sock);
    close(stats_sock);
    log_message("PUMP", "V4L2 Stream stopped and resources released.");
}

void handle_request(int client_socket) {
    char buffer[4096] = {0};
    read(client_socket, buffer, sizeof(buffer)-1);
    log_message("REQUEST", buffer);

    if (strncmp(buffer, "GET /start_stream", 17) == 0) {
        char target_ip[64] = {0};
        char port_str[16] = "5000";
        char fmt_str[16] = "h264";
        char fps_str[16] = "20";
        char w_str[16] = "640";
        char h_str[16] = "480";
        char *ip_ptr = strstr(buffer, "ip=");
        char *port_ptr = strstr(buffer, "port=");
        char *fmt_ptr = strstr(buffer, "fmt=");
        char *fps_ptr = strstr(buffer, "fps=");
        char *w_ptr = strstr(buffer, "w=");
        char *h_ptr = strstr(buffer, "h=");

        if (ip_ptr) sscanf(ip_ptr, "ip=%[^& \t\n\r]", target_ip);
        if (port_ptr) sscanf(port_ptr, "port=%[^& \t\n\r]", port_str);
        if (fmt_ptr) sscanf(fmt_ptr, "fmt=%[^& \t\n\r]", fmt_str);
        if (fps_ptr) sscanf(fps_ptr, "fps=%[^& \t\n\r]", fps_str);
        if (w_ptr) sscanf(w_ptr, "w=%[^& \t\n\r]", w_str);
        if (h_ptr) sscanf(h_ptr, "h=%[^& \t\n\r]", h_str);

        int target_port = atoi(port_str);
        int target_fps = atoi(fps_str);
        int target_width = atoi(w_str);
        int target_height = atoi(h_str);
        int use_mjpeg = (strcmp(fmt_str, "mjpeg") == 0);

        if (strlen(target_ip) == 0) {
            char resp[] = "HTTP/1.1 400 Bad Request\r\n\r\n{\"error\":\"No IP\"}";
            write(client_socket, resp, strlen(resp));
        } else {
            stop_requested = 0;
            cleanup_processes();

            stream_pid = fork();
            if (stream_pid == 0) {
                start_manual_data_pump(target_ip, target_port, use_mjpeg, target_fps, target_width, target_height);
                exit(0);
            }

            char body[256];
            snprintf(body, sizeof(body), "{\"status\":\"success\",\"message\":\"%s Pump started to %s:%d\"}",
                     use_mjpeg ? "MJPEG" : "H264", target_ip, target_port);
            char resp[1024];
            snprintf(resp, sizeof(resp),
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: %zu\r\n"
                "Connection: close\r\n\r\n%s",
                strlen(body), body);

            write(client_socket, resp, strlen(resp));
        }
    } else if (strncmp(buffer, "GET /stop_stream", 16) == 0) {
        stop_requested = 1;
        cleanup_processes();
        char resp[] = "HTTP/1.1 200 OK\r\n\r\n{\"status\":\"stopped\"}";
        write(client_socket, resp, strlen(resp));
    }

    usleep(200000);
    close(client_socket);
}

int main() {
    int server_fd;
    struct sockaddr_in address;
    socklen_t addrlen = sizeof(address);
    signal(SIGCHLD, SIG_IGN);
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);
    bind(server_fd, (struct sockaddr *)&address, sizeof(address));
    listen(server_fd, 5);

    log_message("START", "Internal V4L2 Server Active.");

    while (1) {
        addrlen = sizeof(address);
        int client_socket = accept(server_fd, (struct sockaddr *)&address, &addrlen);
        if (client_socket >= 0) handle_request(client_socket);
    }
    return 0;
}
