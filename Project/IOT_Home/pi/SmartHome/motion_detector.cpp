/*
 * Raspberry Pi Motion Detector & Face Recognizer
 *
 * --- INSTALLATION COMMANDS ---
 * 1. sudo apt-get update
 * 2. sudo apt-get install -y libopencv-dev libpaho-mqtt-dev g++
 *
 * --- FACE RECOGNITION SETUP ---
 * 1. You need a "haarcascade_frontalface_default.xml" file in this directory.
 * 2. To identify specific people, you need a "trainer.yml" file.
 *    (If trainer.yml is missing, it will just detect "Unknown Person")
 *
 * --- COMPILATION ---
 * g++ motion_detector.cpp -o motion_detector `pkg-config --cflags --libs opencv4` -lpaho-mqtt3c
 */

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>
#include <MQTTClient.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <ctime>

using namespace cv;
using namespace cv::face;
using namespace std;

// --- MQTT Configuration ---
#define ADDRESS     "tcp://broker.hivemq.com:1883"
#define CLIENTID    "PiMotionDetector"
#define BASE_TOPIC  "smart_home/"
#define TOPIC_IMG   BASE_TOPIC "camera/motion"
#define TOPIC_NAME  BASE_TOPIC "camera/name"
#define QOS         1
#define TIMEOUT     10000L

// --- Detection Parameters ---
#define SENSITIVITY 25
#define MIN_AREA    1000
#define COOLDOWN    5

// Simple Label Mapping (Example)
string getName(int id) {
    if (id == 1) return "User";
    if (id == 2) return "Friend";
    return "Unknown Person";
}

void logEvent(string name) {
    ofstream logFile;
    logFile.open("motion_log.csv", ios_base::app);

    time_t now = time(0);
    char* dt = ctime(&now);
    string ts(dt);
    ts.erase(ts.find('\n')); // Remove newline

    logFile << ts << "," << name << endl;
    logFile.close();
    printf("Logged: %s at %s\n", name.c_str(), ts.c_str());
}

int main() {
    // 1. MQTT Setup
    MQTTClient client;
    MQTTClient_connectOptions conn_opts = MQTTClient_connectOptions_initializer;
    MQTTClient_create(&client, ADDRESS, CLIENTID, MQTTCLIENT_PERSISTENCE_NONE, NULL);
    conn_opts.keepAliveInterval = 20;
    conn_opts.cleansession = 1;

    if (MQTTClient_connect(client, &conn_opts) != MQTTCLIENT_SUCCESS) {
        printf("Failed to connect to MQTT\n");
        return -1;
    }

    // 2. Load Face Models
    CascadeClassifier face_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        printf("Error: Could not load haarcascade_frontalface_default.xml\n");
        // We can continue with motion only, but recognition won't work
    }

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
    bool modelLoaded = false;
    try {
        model->read("trainer.yml");
        modelLoaded = true;
        printf("Face recognition model loaded.\n");
    } catch (...) {
        printf("Warning: trainer.yml not found. Recognition will default to Unknown.\n");
    }

    // 3. Camera Setup
    VideoCapture cap(0);
    if (!cap.isOpened()) return -1;
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    Mat frame, gray, prev_gray, diff, thresh;
    time_t last_send_time = 0;

    cap.read(frame);
    cvtColor(frame, prev_gray, COLOR_BGR2GRAY);
    GaussianBlur(prev_gray, prev_gray, Size(21, 21), 0);

    printf("Monitoring started...\n");

    while (true) {
        cap.read(frame);
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(21, 21), 0);

        absdiff(prev_gray, gray, diff);
        threshold(diff, thresh, SENSITIVITY, 255, THRESH_BINARY);
        dilate(thresh, thresh, Mat(), Point(-1, -1), 2);

        if (countNonZero(thresh) > MIN_AREA) {
            time_t now = time(NULL);
            if (difftime(now, last_send_time) > COOLDOWN) {

                string identifiedName = "Motion Detected";

                // --- Face Recognition Logic ---
                vector<Rect> faces;
                face_cascade.detectMultiScale(gray, faces, 1.1, 4);

                if (faces.size() > 0) {
                    identifiedName = "Unknown Person";
                    if (modelLoaded) {
                        int label = -1;
                        double confidence = 0.0;
                        // Just check the first face for simplicity
                        Mat faceROI = gray(faces[0]);
                        model->predict(faceROI, label, confidence);

                        if (confidence < 80) { // Confidence threshold (lower is better for LBPH)
                            identifiedName = getName(label);
                        }
                    }
                }

                printf("Alert: %s\n", identifiedName.c_str());
                logEvent(identifiedName);

                // --- Send Data via MQTT ---
                // A. Send Image
                vector<uchar> buf;
                imencode(".jpg", frame, buf, {IMWRITE_JPEG_QUALITY, 50});

                MQTTClient_message pubmsg = MQTTClient_message_initializer;
                pubmsg.payload = buf.data();
                pubmsg.payloadlen = (int)buf.size();
                pubmsg.qos = QOS;
                MQTTClient_deliveryToken token;
                MQTTClient_publishMessage(client, TOPIC_IMG, &pubmsg, &token);

                // B. Send Name
                MQTTClient_message nameMsg = MQTTClient_message_initializer;
                nameMsg.payload = (void*)identifiedName.c_str();
                nameMsg.payloadlen = identifiedName.length();
                nameMsg.qos = QOS;
                MQTTClient_publishMessage(client, TOPIC_NAME, &nameMsg, &token);

                MQTTClient_waitForCompletion(client, token, TIMEOUT);
                last_send_time = now;
            }
        }
        prev_gray = gray.clone();
    }

    MQTTClient_disconnect(client, 10000);
    MQTTClient_destroy(&client);
    return 0;
}
