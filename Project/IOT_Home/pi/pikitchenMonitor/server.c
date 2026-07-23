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
#include <sys/statvfs.h>
#include <pthread.h>
#include <sys/stat.h>
#include <dirent.h>
#include <wait.h>
#include "credentials.h"

#define SW_VERSION "1.0.304"
#define PORT 5001
#define VIDEO_CHUNK_SIZE 1400
#define LOG_FILE "/tmp/server.log"
#define VIDEO_DEVICE "/dev/video0"
#define BUFFER_COUNT 4
#define CAPTURE_DIR "/home/pi/camera"
#define CONFIG_FILE "config.json"

char mqtt_broker[128] = "localhost";
char fb_url[256] = "https://gapsmarthome-default-rtdb.asia-southeast1.firebasedatabase.app";
char fb_node[128] = "pi_kitchen";
char fb_secret[128] = "";
int fb_width = 640;
int fb_height = 480;
int active_video_fd = -1;
pthread_mutex_t v_fd_mutex = PTHREAD_MUTEX_INITIALIZER;

static int xioctl(int fh, int request, void *arg);

void update_firebase_controls() {
    int v_fd = open(VIDEO_DEVICE, O_RDWR);
    if (v_fd < 0) return;

    char *controls_list = malloc(32768);
    char *ctrl_node = malloc(32768);
    if (!controls_list || !ctrl_node) { if(controls_list) free(controls_list); if(ctrl_node) free(ctrl_node); close(v_fd); return; }

    strcpy(controls_list, "[");
    strcpy(ctrl_node, "{");

    struct v4l2_queryctrl qctrl = {0};
    qctrl.id = V4L2_CTRL_FLAG_NEXT_CTRL;
    int first = 1;
    while (xioctl(v_fd, VIDIOC_QUERYCTRL, &qctrl) == 0) {
        if (!(qctrl.flags & V4L2_CTRL_FLAG_DISABLED)) {
            struct v4l2_control ctrl = {qctrl.id, 0};
            xioctl(v_fd, VIDIOC_G_CTRL, &ctrl);
            if (!first) { strcat(controls_list, ","); strcat(ctrl_node, ","); }
            char item[512];
            snprintf(item, sizeof(item), "{\"id\":%u,\"name\":\"%s\",\"min\":%d,\"max\":%d,\"val\":%d,\"def\":%d,\"type\":%u}",
                     qctrl.id, qctrl.name, qctrl.minimum, qctrl.maximum, ctrl.value, qctrl.default_value, qctrl.type);
            strcat(controls_list, item);
            char node_item[256];
            snprintf(node_item, sizeof(node_item), "\"%s\":%d", qctrl.name, ctrl.value);
            strcat(ctrl_node, node_item);
            first = 0;
        }
        qctrl.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
    }
    close(v_fd);
    strcat(controls_list, "]");
    strcat(ctrl_node, "}");

    FILE *tf = fopen("/tmp/controls.json", "w");
    if (tf) {
        fprintf(tf, "{\"controls\":%s, \"ctrl\": %s}", controls_list, ctrl_node);
        fclose(tf);
        char post_url[2048];
        if (strlen(fb_secret) > 0) snprintf(post_url, sizeof(post_url), "curl -s -X PATCH -T /tmp/controls.json %s/FrmPi/%s.json?auth=%s &", fb_url, fb_node, fb_secret);
        else snprintf(post_url, sizeof(post_url), "curl -s -X PATCH -T /tmp/controls.json %s/FrmPi/%s.json &", fb_url, fb_node);
        system(post_url);
    }
    free(controls_list); free(ctrl_node);
}

int stop_requested = 0;
pid_t stream_pid = -1;
int capture_requested = 0;
int fb_stream_active = 0;
long total_bytes_sent = 0;

struct pump_args {
    char target_ip[64];
    int target_port;
    int use_mjpeg;
    int fps;
    int width;
    int height;
};

unsigned char latest_frame[256 * 1024];
size_t latest_frame_len = 0;
pthread_mutex_t frame_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_t camera_thread_id = 0;
int camera_thread_running = 0;

struct buffer {
    void   *start;
    size_t  length;
};

void normalize_fb_url() {
    int len = strlen(fb_url);
    if (len > 0 && fb_url[len - 1] == '/') fb_url[len - 1] = '\0';
}

static const char base64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
void base64_encode(const unsigned char *data, size_t input_length, char *encoded_data) {
    int i = 0, j = 0;
    unsigned char char_array_3[3], char_array_4[4];
    while (input_length--) {
        char_array_3[i++] = *(data++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for (i = 0; i < 4; i++) *encoded_data++ = base64_chars[char_array_4[i]];
            i = 0;
        }
    }
    if (i) {
        for (j = i; j < 3; j++) char_array_3[j] = '\0';
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        for (j = 0; j < i + 1; j++) *encoded_data++ = base64_chars[char_array_4[j]];
        while (i++ < 3) *encoded_data++ = '=';
    }
    *encoded_data = '\0';
}

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

void parse_json_field(const char *buffer, const char *field, char *output, int out_len) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", field);
    char *ptr = strstr(buffer, search);
    if (ptr) {
        ptr = strstr(ptr, ":");
        if (ptr) {
            ptr++; // Skip :
            while (*ptr == ' ') ptr++; // Skip space
            if (*ptr == '\"') {
                ptr++; // Skip opening quote
                int i = 0;
                while (*ptr != '\"' && *ptr != '\0' && i < out_len - 1) {
                    output[i++] = *ptr++;
                }
                output[i] = '\0';
            } else {
                // Not a string (number or bool)
                int i = 0;
                while (*ptr != ',' && *ptr != '}' && *ptr != ' ' && *ptr != '\0' && i < out_len - 1) {
                    output[i++] = *ptr++;
                }
                output[i] = '\0';
            }
        }
    }
}

void load_config() {
    FILE *fp = fopen(CONFIG_FILE, "r");
    if (fp) {
        char buffer[2048];
        int n = fread(buffer, 1, sizeof(buffer) - 1, fp);
        buffer[n] = '\0';
        fclose(fp);
        parse_json_field(buffer, "mqtt_broker", mqtt_broker, sizeof(mqtt_broker));
        parse_json_field(buffer, "fb_url", fb_url, sizeof(fb_url));
        parse_json_field(buffer, "fb_node", fb_node, sizeof(fb_node));
        parse_json_field(buffer, "fb_secret", fb_secret, sizeof(fb_secret));
        char tmp[16];
        tmp[0] = '\0'; parse_json_field(buffer, "fb_width", tmp, sizeof(tmp));
        if (tmp[0]) fb_width = atoi(tmp);
        tmp[0] = '\0'; parse_json_field(buffer, "fb_height", tmp, sizeof(tmp));
        if (tmp[0]) fb_height = atoi(tmp);
        normalize_fb_url();
        log_message("CONFIG", "Loaded and Normalized");
    }
}

void save_config(const char *json) {
    FILE *fp = fopen(CONFIG_FILE, "w");
    if (fp) {
        fputs(json, fp);
        fclose(fp);
        log_message("CONFIG", "Saved to file");
    }
}

static int xioctl(int fh, int request, void *arg) {
    int r;
    do { r = ioctl(fh, request, arg); } while (-1 == r && EINTR == errno);
    return r;
}

void cleanup_processes() {
    log_message("SERVER", "Cleaning up camera hardware...");
    stop_requested = 1;
    if (camera_thread_running) {
        pthread_join(camera_thread_id, NULL);
        camera_thread_id = 0;
        camera_thread_running = 0;
    }
    stream_pid = -1;
    system("pkill -9 raspivid 2>/dev/null");
    system("pkill -9 libcamera-vid 2>/dev/null");
}

void check_and_cleanup_storage() {
    struct statvfs vfs;
    if (statvfs(CAPTURE_DIR, &vfs) == 0) {
        double free_pct = (double)vfs.f_bavail / vfs.f_blocks * 100.0;
        if (free_pct < 10.0) {
            log_message("CLEANUP", "Storage low, deleting oldest images...");
            struct dirent **namelist;
            int n = scandir(CAPTURE_DIR, &namelist, NULL, alphasort);
            if (n >= 0) {
                int deleted = 0;
                for (int i = 0; i < n && deleted < 50; i++) {
                    if (namelist[i]->d_type == DT_REG && strstr(namelist[i]->d_name, "IMG_")) {
                        char path[512];
                        snprintf(path, sizeof(path), "%s/%s", CAPTURE_DIR, namelist[i]->d_name);
                        remove(path);
                        deleted++;
                    }
                    free(namelist[i]);
                }
                free(namelist);
                log_message("CLEANUP", "Storage cleanup complete");
            }
        }
    }
}

void save_frame_as_jpeg(unsigned char *data, size_t size) {
    char filename[256];
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    static int counter = 0;
    struct stat st = {0};
    if (stat(CAPTURE_DIR, &st) == -1) {
        mkdir("/home/pi", 0755);
        mkdir(CAPTURE_DIR, 0755);
    }
    check_and_cleanup_storage();
    strftime(filename, sizeof(filename), CAPTURE_DIR "/IMG_%Y%m%d_%H%M%S", t);
    sprintf(filename + strlen(filename), "_%03d.jpg", counter++ % 1000);
    FILE *fp = fopen(filename, "wb");
    if (fp) {
        fwrite(data, 1, size, fp);
        fclose(fp);
        log_message("CAPTURE", filename);
    }
}

void handle_sigusr1(int sig) { capture_requested = 1; }

void *start_manual_data_pump(void *arg) {
    struct pump_args *a = (struct pump_args *)arg;
    int target_port = a->target_port;
    int use_mjpeg = a->use_mjpeg;
    int fps = a->fps;
    int width = a->width;
    int height = a->height;

    int sock = -1, stats_sock = -1;
    struct sockaddr_in server_addr, stats_addr;
    int video_fd = -1;
    struct buffer *buffers = NULL;
    time_t last_stats_sent = 0;

    signal(SIGUSR1, handle_sigusr1);
    log_message("PUMP", "Starting V4L2 Data Pump...");

    if (strlen(a->target_ip) > 0) {
        sock = socket(AF_INET, SOCK_DGRAM, 0);
        stats_sock = socket(AF_INET, SOCK_DGRAM, 0);
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(target_port);
        server_addr.sin_addr.s_addr = inet_addr(a->target_ip);
        memset(&stats_addr, 0, sizeof(stats_addr));
        stats_addr.sin_family = AF_INET;
        stats_addr.sin_port = htons(target_port + 1);
        stats_addr.sin_addr.s_addr = inet_addr(a->target_ip);
    }

    video_fd = open(VIDEO_DEVICE, O_RDWR | O_NONBLOCK, 0);
    if (video_fd < 0) { log_message("PUMP", "Error: Could not open video device"); free(a); camera_thread_running = 0; return NULL; }

    pthread_mutex_lock(&v_fd_mutex);
    active_video_fd = video_fd;
    pthread_mutex_unlock(&v_fd_mutex);

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width > 0 ? width : 640;
    fmt.fmt.pix.height = height > 0 ? height : 480;
    fmt.fmt.pix.pixelformat = use_mjpeg ? V4L2_PIX_FMT_MJPEG : V4L2_PIX_FMT_H264;
    xioctl(video_fd, VIDIOC_S_FMT, &fmt);

    struct v4l2_streamparm parm = {0};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps > 0 ? fps : 20;
    xioctl(video_fd, VIDIOC_S_PARM, &parm);

    struct v4l2_control rot_ctrl = {V4L2_CID_ROTATE, 270};
    xioctl(video_fd, VIDIOC_S_CTRL, &rot_ctrl);

    struct v4l2_requestbuffers req = {0};
    req.count = BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(video_fd, VIDIOC_REQBUFS, &req);
    buffers = calloc(req.count, sizeof(*buffers));
    for (int i = 0; i < req.count; ++i) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        xioctl(video_fd, VIDIOC_QUERYBUF, &buf);
        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, video_fd, buf.m.offset);
        xioctl(video_fd, VIDIOC_QBUF, &buf);
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(video_fd, VIDIOC_STREAMON, &type);

    unsigned int frame_counter = 0;
    while (!stop_requested) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(video_fd, VIDIOC_DQBUF, &buf) < 0) { usleep(10000); continue; }

        unsigned char *data = (unsigned char *)buffers[buf.index].start;
        size_t size = buf.bytesused;

        if (use_mjpeg) {
            pthread_mutex_lock(&frame_mutex);
            if (size < sizeof(latest_frame)) {
                memcpy(latest_frame, data, size);
                latest_frame_len = size;
            }
            pthread_mutex_unlock(&frame_mutex);
        }

        static size_t last_size = 0;
        static time_t last_cap = 0;
        if (last_size > 0 && abs((int)size - (int)last_size) > (last_size / 8)) {
            if (time(NULL) - last_cap > 5) {
                capture_requested = 1;
                last_cap = time(NULL);
                log_message("MOTION", "Scene change detected");

                if (strlen(mqtt_broker) > 0) {
                    char m_cmd[512];
                    snprintf(m_cmd, sizeof(m_cmd), "mosquitto_pub -h %s -t 'FrmPi/%s/motion' -m 'detected' 2>/dev/null &", mqtt_broker, fb_node);
                    system(m_cmd);
                }

                if (strlen(fb_url) > 0) {
                    char f_cmd[1024];
                    if (strlen(fb_secret) > 0) snprintf(f_cmd, sizeof(f_cmd), "curl -s -X PATCH -d '{\"last_motion\":%ld}' %s/FrmPi/%s.json?auth=%s &", time(NULL), fb_url, fb_node, fb_secret);
                    else snprintf(f_cmd, sizeof(f_cmd), "curl -s -X PATCH -d '{\"last_motion\":%ld}' %s/FrmPi/%s.json &", time(NULL), fb_url, fb_node);
                    system(f_cmd);
                }
            }
        }
        last_size = size;

        if (capture_requested) { save_frame_as_jpeg(data, size); capture_requested = 0; }

        if (sock >= 0) {
            size_t sent = 0;
            unsigned short p_idx = 0;
            unsigned short f_id = (unsigned short)(frame_counter & 0xFFFF);
            while (sent < size) {
                size_t to_send = (size - sent > VIDEO_CHUNK_SIZE) ? VIDEO_CHUNK_SIZE : (size - sent);
                unsigned char pkt[VIDEO_CHUNK_SIZE + 8];
                pkt[0] = 0x55; pkt[1] = 0xAA;
                pkt[2] = (f_id >> 8) & 0xFF; pkt[3] = f_id & 0xFF;
                pkt[4] = (p_idx >> 8) & 0xFF; pkt[5] = p_idx & 0xFF;
                pkt[6] = (sent + to_send >= size) ? 1 : 0; pkt[7] = 0;
                memcpy(pkt + 8, data + sent, to_send);
                sendto(sock, pkt, to_send + 8, 0, (struct sockaddr *)&server_addr, sizeof(server_addr));
                sent += to_send; p_idx++;
            }
            frame_counter++; total_bytes_sent += size;
            if (time(NULL) > last_stats_sent) {
                char s_pkt[64]; snprintf(s_pkt, sizeof(s_pkt), "SENT:%ld", total_bytes_sent);
                sendto(stats_sock, s_pkt, strlen(s_pkt), 0, (struct sockaddr *)&stats_addr, sizeof(stats_addr));
                last_stats_sent = time(NULL);
            }
        }
        xioctl(video_fd, VIDIOC_QBUF, &buf);
    }

    if (buffers) {
        for (int i = 0; i < BUFFER_COUNT; ++i) if (buffers[i].start) munmap(buffers[i].start, buffers[i].length);
        free(buffers);
    }
    if (video_fd >= 0) close(video_fd);

    pthread_mutex_lock(&v_fd_mutex);
    if (active_video_fd == video_fd) active_video_fd = -1;
    pthread_mutex_unlock(&v_fd_mutex);

    if (sock >= 0) close(sock);
    if (stats_sock >= 0) close(stats_sock);
    camera_thread_running = 0;
    free(a);
    return NULL;
}

pid_t current_shell_pid = -1;
int shell_stdin_pipe[2];

void handle_shell_command(int client_socket, char *cmd) {
    char body[8192] = {0};
    int stdout_pipe[2];
    if (pipe(stdout_pipe) == -1 || pipe(shell_stdin_pipe) == -1) {
        write(client_socket, "HTTP/1.1 500 Error\r\n\r\n", 22);
        return;
    }
    current_shell_pid = fork();
    if (current_shell_pid == 0) {
        close(stdout_pipe[0]); close(shell_stdin_pipe[1]);
        dup2(stdout_pipe[1], STDOUT_FILENO); dup2(stdout_pipe[1], STDERR_FILENO);
        dup2(shell_stdin_pipe[0], STDIN_FILENO);
        execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
        exit(1);
    } else {
        close(stdout_pipe[1]); close(shell_stdin_pipe[0]);
        char path[1024];
        strcat(body, "{\"output\":\"");
        int flags = fcntl(stdout_pipe[0], F_GETFL, 0);
        fcntl(stdout_pipe[0], F_SETFL, flags | O_NONBLOCK);
        usleep(500000);
        FILE *fp = fdopen(stdout_pipe[0], "r");
        if (fp) {
            while (fgets(path, sizeof(path), fp) != NULL) {
                for(int i=0; path[i]; i++) {
                    if(path[i] == '\n') strcat(body, "\\n");
                    else if(path[i] == '\"') strcat(body, "\\\"");
                    else if(path[i] == '\\') strcat(body, "\\\\");
                    else { char temp[2] = {path[i], 0}; strcat(body, temp); }
                }
            }
            fclose(fp);
        }
        strcat(body, "\"}");
        if (waitpid(current_shell_pid, NULL, WNOHANG) != 0) {
            current_shell_pid = -1; close(shell_stdin_pipe[1]);
        }
    }
    char resp[9000];
    snprintf(resp, sizeof(resp), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n\r\n%s", strlen(body), body);
    write(client_socket, resp, strlen(resp));
}

void *firebase_shell_poller(void *arg) {
    char last_cmd_id[128] = "";
    char last_in_id[128] = "";
    char last_config[4096] = "";
    log_message("FB_POLLER", "Worker started");
    while (1) {
        if (strlen(fb_url) == 0) { sleep(5); continue; }

        char url[1024];
        // 1. Poll for configuration from FrmMobile
        if (strlen(fb_secret) > 0) snprintf(url, sizeof(url), "curl -s %s/FrmMobile/%s/config.json?auth=%s", fb_url, fb_node, fb_secret);
        else snprintf(url, sizeof(url), "curl -s %s/FrmMobile/%s/config.json", fb_url, fb_node);

        FILE *cfp = popen(url, "r");
        if (cfp) {
            char config_buffer[4096] = {0};
            int n = fread(config_buffer, 1, sizeof(config_buffer)-1, cfp);
            config_buffer[n] = '\0';
            pclose(cfp);
            if (n > 2 && strcmp(config_buffer, last_config) != 0 && strstr(config_buffer, "{")) {
                strcpy(last_config, config_buffer);
                save_config(config_buffer);
                load_config();
                update_firebase_controls();

                // Confirm connection by posting heartbeat
                char h_cmd[1024];
                if (strlen(fb_secret) > 0) snprintf(h_cmd, sizeof(h_cmd), "curl -s -X PATCH -d '{\"last_poll\":%ld,\"v\":\"%s\"}' %s/FrmPi/%s.json?auth=%s &", time(NULL), SW_VERSION, fb_url, fb_node, fb_secret);
                else snprintf(h_cmd, sizeof(h_cmd), "curl -s -X PATCH -d '{\"last_poll\":%ld,\"v\":\"%s\"}' %s/FrmPi/%s.json &", time(NULL), SW_VERSION, fb_url, fb_node);
                system(h_cmd);
                log_message("CONFIG", "Updated and Heartbeat sent");
            }
        }

        // 2. Poll for shell commands from FrmMobile
        if (strlen(fb_secret) > 0) snprintf(url, sizeof(url), "curl -s %s/FrmMobile/%s/shell.json?auth=%s", fb_url, fb_node, fb_secret);
        else snprintf(url, sizeof(url), "curl -s %s/FrmMobile/%s/shell.json", fb_url, fb_node);

        FILE *fp = popen(url, "r");
        if (fp) {
            char buffer[16384] = {0};
            int n = fread(buffer, 1, sizeof(buffer)-1, fp);
            buffer[n] = '\0';
            pclose(fp);

            char id[128] = {0}, cmd[512] = {0};
            parse_json_field(buffer, "id", id, sizeof(id));
            parse_json_field(buffer, "cmd", cmd, sizeof(cmd));

            if (id[0] != '\0' && cmd[0] != '\0' && strcmp(id, last_cmd_id) != 0) {
                strcpy(last_cmd_id, id);
                char response_data[16384] = "Command Executed";

                if (strcmp(cmd, "refresh_images") == 0) {
                    DIR *d = opendir(CAPTURE_DIR);
                    char *p = response_data;
                    strcpy(p, "IMG_LIST:["); p += 10;
                    if (d) {
                        struct dirent *dir; int first = 1;
                        while ((dir = readdir(d)) != NULL) {
                            if (dir->d_type == DT_REG && strstr(dir->d_name, ".jpg")) {
                                if (!first) { *p++ = ','; }
                                size_t n_len = strlen(dir->d_name);
                                if ((p - response_data) + n_len + 10 > sizeof(response_data)) break;
                                strcpy(p, "\\\""); p += 2;
                                strcpy(p, dir->d_name); p += n_len;
                                strcpy(p, "\\\""); p += 2;
                                first = 0;
                            }
                        }
                        closedir(d);
                    }
                    strcpy(p, "]");
                } else if (strcmp(cmd, "query_controls") == 0) {
                    update_firebase_controls();
                    strcpy(response_data, "Controls Updated in Firebase");
                } else if (strncmp(cmd, "set_control:", 12) == 0) {
                    unsigned int id_val; int val;
                    sscanf(cmd + 12, "%u:%d", &id_val, &val);
                    struct v4l2_control ctrl = {id_val, val};
                    pthread_mutex_lock(&v_fd_mutex);
                    int res = -1;
                    if (active_video_fd >= 0) res = xioctl(active_video_fd, VIDIOC_S_CTRL, &ctrl);
                    else {
                        int v_fd = open(VIDEO_DEVICE, O_RDWR);
                        if (v_fd >= 0) { res = xioctl(v_fd, VIDIOC_S_CTRL, &ctrl); close(v_fd); }
                    }
                    pthread_mutex_unlock(&v_fd_mutex);
                    if (res == 0) {
                        update_firebase_controls();
                        strcpy(response_data, "Control Set Successfully");
                    } else {
                        strcpy(response_data, "Error Setting Control");
                    }
                } else if (strcmp(cmd, "reset_controls") == 0) {
                    int v_fd = open(VIDEO_DEVICE, O_RDWR);
                    if (v_fd >= 0) {
                        struct v4l2_queryctrl qctrl = {0};
                        qctrl.id = V4L2_CTRL_FLAG_NEXT_CTRL;
                        while (xioctl(v_fd, VIDIOC_QUERYCTRL, &qctrl) == 0) {
                            if (!(qctrl.flags & V4L2_CTRL_FLAG_DISABLED)) {
                                struct v4l2_control ctrl = {qctrl.id, qctrl.default_value};
                                pthread_mutex_lock(&v_fd_mutex);
                                if (active_video_fd >= 0) xioctl(active_video_fd, VIDIOC_S_CTRL, &ctrl);
                                else xioctl(v_fd, VIDIOC_S_CTRL, &ctrl);
                                pthread_mutex_unlock(&v_fd_mutex);
                            }
                            qctrl.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
                        }
                        close(v_fd);
                        update_firebase_controls();
                    }
                    strcpy(response_data, "Controls Reset Successfully");
                } else if (strcmp(cmd, "start_fb_stream") == 0) {
                    fb_stream_active = 1;
                    strcpy(response_data, "Firebase Stream Started");
                } else if (strcmp(cmd, "stop_fb_stream") == 0) {
                    fb_stream_active = 0;
                    strcpy(response_data, "Firebase Stream Stopped");
                } else if (strncmp(cmd, "get_image:", 10) == 0) {
                    char name[128]; strcpy(name, cmd + 10);
                    char path[256]; snprintf(path, sizeof(path), CAPTURE_DIR "/%s", name);
                    char b64_cmd[512];
                    snprintf(b64_cmd, sizeof(b64_cmd), "cat %s | openssl base64 -A", path);
                    FILE *bfp = popen(b64_cmd, "r");
                    if (bfp) {
                        strcpy(response_data, "IMG_DATA:");
                        fread(response_data + 9, 1, sizeof(response_data) - 10, bfp);
                        pclose(bfp);
                    } else strcpy(response_data, "ERROR: Could not read image");
                } else {
                    FILE *efp = popen(cmd, "r");
                    if (efp) {
                        char line[256];
                        char *p = response_data;
                        *p = '\0';
                        size_t max_len = sizeof(response_data) - 100;
                        while (fgets(line, sizeof(line), efp)) {
                            for(int i=0; line[i] && (p - response_data) < max_len; i++) {
                                if(line[i] == '\n') { strcpy(p, "\\n"); p += 2; }
                                else if(line[i] == '\r') continue;
                                else if(line[i] == '\"') { strcpy(p, "\\\""); p += 2; }
                                else if(line[i] == '\\') { strcpy(p, "\\\\"); p += 2; }
                                else { *p++ = line[i]; }
                            }
                        }
                        *p = '\0';
                        pclose(efp);
                    }
                }

                char post_url[2048];
                FILE *tf = fopen("/tmp/fb_resp.json", "w");
                if (tf) {
                    fprintf(tf, "{\"response\":\"%s\",\"res_id\":\"%s\"}", response_data, id);
                    fclose(tf);
                    if (strlen(fb_secret) > 0) snprintf(post_url, sizeof(post_url), "curl -s -X PATCH -T /tmp/fb_resp.json %s/FrmPi/%s/shell.json?auth=%s &", fb_url, fb_node, fb_secret);
                    else snprintf(post_url, sizeof(post_url), "curl -s -X PATCH -T /tmp/fb_resp.json %s/FrmPi/%s/shell.json &", fb_url, fb_node);
                    system(post_url);
                }
            }
            char *in_id_ptr = strstr(buffer, "\"in_id\":\"");
            char *stdin_ptr = strstr(buffer, "\"stdin\":\"");
            if (in_id_ptr && stdin_ptr && current_shell_pid > 0) {
                char in_id[128], input[512];
                sscanf(in_id_ptr, "\"in_id\":\"%[^\"]", in_id);
                sscanf(stdin_ptr, "\"stdin\":\"%[^\"]", input);
                if (strcmp(in_id, last_in_id) != 0) {
                    strcpy(last_in_id, in_id);
                    write(shell_stdin_pipe[1], input, strlen(input));
                    write(shell_stdin_pipe[1], "\n", 1);
                }
            }
        }
        sleep(2);
    }
    return NULL;
}

void *firebase_live_stream_thread(void *arg) {
    log_message("FB_STREAM", "Worker started");
    while (1) {
        if (fb_stream_active && strlen(fb_url) > 0) {
            // Check if user is still active
            char url[1024];
            if (strlen(fb_secret) > 0) snprintf(url, sizeof(url), "curl -s %s/FrmMobile/%s/active.json?auth=%s", fb_url, fb_node, fb_secret);
            else snprintf(url, sizeof(url), "curl -s %s/FrmMobile/%s/active.json", fb_url, fb_node);

            FILE *fp = popen(url, "r");
            if (fp) {
                char res[32] = {0};
                fread(res, 1, sizeof(res)-1, fp);
                pclose(fp);
                if (strstr(res, "false")) {
                    log_message("FB_STREAM", "User inactive, stopping stream");
                    fb_stream_active = 0;
                    continue;
                }
            }

            int capture_success = 0;
            char *b64_data = malloc(512 * 1024);

            if (!camera_thread_running) {
                struct pump_args *args = calloc(1, sizeof(struct pump_args));
                strcpy(args->target_ip, "");
                args->use_mjpeg = 1; args->fps = 5; args->width = fb_width; args->height = fb_height;
                stop_requested = 0; camera_thread_running = 1;
                pthread_create(&camera_thread_id, NULL, start_manual_data_pump, args);
                stream_pid = 1;
                usleep(1000000); // Give it time to start
            }

            pthread_mutex_lock(&frame_mutex);
            if (latest_frame_len > 0) {
                base64_encode(latest_frame, latest_frame_len, b64_data);
                capture_success = 1;
            }
            pthread_mutex_unlock(&frame_mutex);

            if (capture_success) {
                FILE *tf = fopen("/tmp/fb_stream.json", "w");
                if (tf) {
                    fprintf(tf, "{\"frame\":\"%s\",\"ts\":%ld}", b64_data, time(NULL));
                    fclose(tf);
                    char post_url[1024];
                    if (strlen(fb_secret) > 0) snprintf(post_url, sizeof(post_url), "curl -s -X PATCH -d @/tmp/fb_stream.json %s/FrmPi/%s/live.json?auth=%s &", fb_url, fb_node, fb_secret);
                    else snprintf(post_url, sizeof(post_url), "curl -s -X PATCH -d @/tmp/fb_stream.json %s/FrmPi/%s/live.json &", fb_url, fb_node);
                    system(post_url);
                    log_message("FB_STREAM", "Frame pushed to Firebase");
                }
            }
            free(b64_data);
            sleep(2);
        } else {
            sleep(2);
        }
    }
    return NULL;
}

void handle_request(int client_socket) {
    char buffer[4096] = {0};
    int n = read(client_socket, buffer, sizeof(buffer)-1);
    if (n <= 0) { close(client_socket); return; }

    char log_buf[512];
    char first_line[256] = {0};
    sscanf(buffer, "%255[^\r\n]", first_line);
    snprintf(log_buf, sizeof(log_buf), "REQ: %s", first_line);
    log_message("SERVER", log_buf);

    if (strncmp(buffer, "GET /version", 12) == 0) {
        char body[128];
        snprintf(body, sizeof(body), "{\"version\":\"%s\"}", SW_VERSION);
        char resp[256];
        snprintf(resp, sizeof(resp), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s", strlen(body), body);
        write(client_socket, resp, strlen(resp));
    } else if (strncmp(buffer, "GET /start_stream", 17) == 0) {
        char target_ip[64] = {0}; char port_str[16] = "5000"; char fmt_str[16] = "h264";
        char fps_str[16] = "20"; char w_str[16]; char h_str[16];
        snprintf(w_str, sizeof(w_str), "%d", fb_width);
        snprintf(h_str, sizeof(h_str), "%d", fb_height);
        char *ip_ptr = strstr(buffer, "ip="); char *port_ptr = strstr(buffer, "port=");
        char *fmt_ptr = strstr(buffer, "fmt="); char *fps_ptr = strstr(buffer, "fps=");
        char *w_ptr = strstr(buffer, "w="); char *h_ptr = strstr(buffer, "h=");
        if (ip_ptr) sscanf(ip_ptr, "ip=%[^& ]", target_ip);
        if (port_ptr) sscanf(port_ptr, "port=%[^& ]", port_str);
        if (fmt_ptr) sscanf(fmt_ptr, "fmt=%[^& ]", fmt_str);
        if (fps_ptr) sscanf(fps_ptr, "fps=%[^& ]", fps_str);
        if (w_ptr) sscanf(w_ptr, "w=%[^& ]", w_str);
        if (h_ptr) sscanf(h_ptr, "h=%[^& ]", h_str);

        struct pump_args *args = malloc(sizeof(struct pump_args));
        strcpy(args->target_ip, target_ip);
        args->target_port = atoi(port_str);
        args->use_mjpeg = (strcmp(fmt_str, "mjpeg") == 0);
        args->fps = atoi(fps_str);
        args->width = atoi(w_str);
        args->height = atoi(h_str);

        cleanup_processes();
        stop_requested = 0;
        camera_thread_running = 1;
        pthread_create(&camera_thread_id, NULL, start_manual_data_pump, args);
        stream_pid = 1;

        const char *resp = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
        write(client_socket, resp, strlen(resp));
    } else if (strncmp(buffer, "GET /stop_stream", 16) == 0) {
        stop_requested = 1; cleanup_processes();
        const char *resp = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
        write(client_socket, resp, strlen(resp));
    } else if (strncmp(buffer, "GET /capture", 12) == 0) {
        capture_requested = 1;
        const char *resp = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
        write(client_socket, resp, strlen(resp));
    } else if (strncmp(buffer, "GET /query_controls", 19) == 0) {
        int v_fd = open(VIDEO_DEVICE, O_RDWR);
        char *body = malloc(32768);
        if (!body) { close(v_fd); return; }
        strcpy(body, "{\"controls\":[");
        if (v_fd >= 0) {
            struct v4l2_queryctrl qctrl = {0};
            qctrl.id = V4L2_CTRL_FLAG_NEXT_CTRL;
            int first = 1;
            while (xioctl(v_fd, VIDIOC_QUERYCTRL, &qctrl) == 0) {
                if (!(qctrl.flags & V4L2_CTRL_FLAG_DISABLED)) {
                    struct v4l2_control ctrl = {qctrl.id, 0};
                    xioctl(v_fd, VIDIOC_G_CTRL, &ctrl);
                    if (!first) strcat(body, ",");
                    char item[512];
                    snprintf(item, sizeof(item), "{\"id\":%u,\"name\":\"%s\",\"min\":%d,\"max\":%d,\"val\":%d,\"def\":%d,\"type\":%u}",
                             qctrl.id, qctrl.name, qctrl.minimum, qctrl.maximum, ctrl.value, qctrl.default_value, qctrl.type);
                    strcat(body, item);
                    first = 0;
                }
                qctrl.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
            }
            close(v_fd);
        }
        strcat(body, "]}");
        char resp_header[256];
        snprintf(resp_header, sizeof(resp_header), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", strlen(body));
        write(client_socket, resp_header, strlen(resp_header));
        write(client_socket, body, strlen(body));
        free(body);
        return;
    } else if (strncmp(buffer, "GET /set_control", 16) == 0) {
        char *id_ptr = strstr(buffer, "id=");
        char *val_ptr = strstr(buffer, "val=");
        if (id_ptr && val_ptr) {
            unsigned int id = (unsigned int)strtoul(id_ptr + 3, NULL, 10);
            int val = atoi(val_ptr + 4);
            struct v4l2_control ctrl = {id, val};

            pthread_mutex_lock(&v_fd_mutex);
            if (active_video_fd >= 0) xioctl(active_video_fd, VIDIOC_S_CTRL, &ctrl);
            else {
                int v_fd = open(VIDEO_DEVICE, O_RDWR);
                if (v_fd >= 0) { xioctl(v_fd, VIDIOC_S_CTRL, &ctrl); close(v_fd); }
            }
            pthread_mutex_unlock(&v_fd_mutex);
        }
        const char *resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
        write(client_socket, resp, strlen(resp));
    } else if (strncmp(buffer, "GET /reset_controls", 19) == 0) {
        int v_fd = open(VIDEO_DEVICE, O_RDWR);
        if (v_fd >= 0) {
            struct v4l2_queryctrl qctrl = {0};
            qctrl.id = V4L2_CTRL_FLAG_NEXT_CTRL;
            while (xioctl(v_fd, VIDIOC_QUERYCTRL, &qctrl) == 0) {
                if (!(qctrl.flags & V4L2_CTRL_FLAG_DISABLED)) {
                    struct v4l2_control ctrl = {qctrl.id, qctrl.default_value};
                    pthread_mutex_lock(&v_fd_mutex);
                    if (active_video_fd >= 0) xioctl(active_video_fd, VIDIOC_S_CTRL, &ctrl);
                    else xioctl(v_fd, VIDIOC_S_CTRL, &ctrl);
                    pthread_mutex_unlock(&v_fd_mutex);
                }
                qctrl.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
            }
            close(v_fd);
        }
        const char *resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
        write(client_socket, resp, strlen(resp));
    } else if (strncmp(buffer, "GET /list_images", 16) == 0) {
        DIR *d = opendir(CAPTURE_DIR);
        char *body = malloc(128 * 1024);
        strcpy(body, "{\"images\":[");
        if (d) {
            struct dirent *dir; int first = 1;
            while ((dir = readdir(d)) != NULL) {
                if (dir->d_type == DT_REG && strstr(dir->d_name, ".jpg")) {
                    if (strlen(body) > 120000) break;
                    if (!first) strcat(body, ",");
                    strcat(body, "\""); strcat(body, dir->d_name); strcat(body, "\"");
                    first = 0;
                }
            }
            closedir(d);
        }
        strcat(body, "]}");
        char resp_header[256];
        snprintf(resp_header, sizeof(resp_header), "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", strlen(body));
        write(client_socket, resp_header, strlen(resp_header));
        write(client_socket, body, strlen(body));
        free(body);
        return;
    } else if (strncmp(buffer, "GET /get_image", 14) == 0) {
        char name[128] = {0}; char *name_ptr = strstr(buffer, "name=");
        if (name_ptr) sscanf(name_ptr, "name=%[^& ]", name);
        char path[256]; snprintf(path, sizeof(path), CAPTURE_DIR "/%s", name);
        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
            char h[128]; snprintf(h, sizeof(h), "HTTP/1.1 200 OK\r\nContent-Length: %ld\r\nConnection: close\r\n\r\n", s);
            write(client_socket, h, strlen(h));
            unsigned char *b = malloc(s); fread(b, 1, s, f);
            write(client_socket, b, s); free(b); fclose(f);
        } else {
            const char *not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            write(client_socket, not_found, strlen(not_found));
        }
    } else if (strncmp(buffer, "GET /shell", 10) == 0) {
        char cmd[512] = {0}; char *cmd_ptr = strstr(buffer, "cmd=");
        if (!cmd_ptr) cmd_ptr = strstr(buffer, "data=");
        if (cmd_ptr) {
            sscanf(cmd_ptr, "%*[^=]=%[^& ]", cmd);
            for(int i=0; cmd[i]; i++) if(cmd[i] == '+') cmd[i] = ' ';
            handle_shell_command(client_socket, cmd);
            return; // handle_shell_command sends its own response
        }
    } else if (strncmp(buffer, "GET /shell_in", 13) == 0) {
        char input[512] = {0}; char *data_ptr = strstr(buffer, "data=");
        if (data_ptr) {
            sscanf(data_ptr, "data=%[^& ]", input);
            for(int i=0; input[i]; i++) if(input[i] == '+') input[i] = ' ';
            if (current_shell_pid > 0) {
                write(shell_stdin_pipe[1], input, strlen(input));
                write(shell_stdin_pipe[1], "\n", 1);
            }
        }
        const char *resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
        write(client_socket, resp, strlen(resp));
        return;
    } else if (strncmp(buffer, "POST /write_file", 16) == 0) {
        char name[128] = "temp.txt"; char *name_ptr = strstr(buffer, "name=");
        if (name_ptr) sscanf(name_ptr, "name=%[^& ]", name);
        char *body = strstr(buffer, "\r\n\r\n");
        if (body) {
            body += 4; FILE *f = fopen(name, "w");
            if (f) { fputs(body, f); fclose(f); write(client_socket, "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nOK", 39); }
        }
    } else if (strncmp(buffer, "POST /config", 12) == 0) {
        char *body = strstr(buffer, "\r\n\r\n");
        if (body) { body += 4; save_config(body); load_config(); write(client_socket, "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nOK", 39); }
    } else {
        const char *unknown = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        write(client_socket, unknown, strlen(unknown));
    }
    fsync(client_socket);
    usleep(50000);
    close(client_socket);
}

int main() {
    load_config();
    int s_fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {AF_INET, htons(PORT), INADDR_ANY};
    int opt = 1; setsockopt(s_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    bind(s_fd, (struct sockaddr *)&addr, sizeof(addr));
    listen(s_fd, 5);
    int flags = fcntl(s_fd, F_GETFL, 0); fcntl(s_fd, F_SETFL, flags | O_NONBLOCK);
    pthread_t shell_thread, stream_thread;
    pthread_create(&shell_thread, NULL, firebase_shell_poller, NULL);
    pthread_create(&stream_thread, NULL, firebase_live_stream_thread, NULL);
    log_message("START", "Server Active");
    while (1) {
        int c_s = accept(s_fd, NULL, NULL);
        if (c_s >= 0) handle_request(c_s);
        else usleep(100000);
    }
    return 0;
}
