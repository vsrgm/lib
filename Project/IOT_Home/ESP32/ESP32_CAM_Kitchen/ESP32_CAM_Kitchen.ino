#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <Adafruit_ADS1X15.h>
#include "PCF8574.h"
#include "FS.h"
#include "SD_MMC.h"
#include <ArduinoJson.h>
#include <Firebase_ESP_Client.h>
#include "credentials.h"
#include <time.h>
#include <WebServer.h>

// ESP32-CAM Pin definitions
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define I2C_SDA 1
#define I2C_SCL 3

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

PCF8574 pcf(0x20, I2C_SDA, I2C_SCL);
Adafruit_BMP280 bmp(&Wire);
Adafruit_ADS1115 ads;

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
WebServer server(80);

// Firebase Data objects
FirebaseData fbdo;
FirebaseAuth fbAuth;
FirebaseConfig fbConfig;

const char* ssid = HOME_NETWORK_SSID;
const char* password = HOME_NETWORK_PASSWORD;
String mqttBroker = MQTT_BROKER;
int mqttPort = MQTT_PORT;
int currentMqttPortIndex = 0;
const int mqttPorts[] = {1883, 8000, 8883, 8884};
const int numMqttPorts = 4;

String clientId = "ESP32_Kitchen_" + String((uint32_t)ESP.getEfuseMac(), HEX);
String baseTopic = "smart_home/kitchen/";
String statusTopic = baseTopic + "status";
String cmdTopic = baseTopic + "commands";
String imageTopic = baseTopic + "image";

bool lastPIR = false;
bool lastLDR = false;
bool lastPower = false;
bool relayState = false;
bool buzzerState = false;
int gasValue = 0;

unsigned long lastStatusMillis = 0;
const long statusInterval = 30000;

void reconnectMqtt() {
  static unsigned long lastReconnectAttempt = 0;
  if (millis() - lastReconnectAttempt > 5000) {
    lastReconnectAttempt = millis();

    mqttPort = mqttPorts[currentMqttPortIndex];
    currentMqttPortIndex = (currentMqttPortIndex + 1) % numMqttPorts;

    Serial.print("Attempting MQTT connection on port ");
    Serial.print(mqttPort);
    Serial.print("... ");

    mqttClient.setServer(mqttBroker.c_str(), mqttPort);

    if (mqttClient.connect(clientId.c_str())) {
      Serial.println("connected");
      mqttClient.subscribe(cmdTopic.c_str());
      publishStatus();
    } else {
      Serial.print("failed, rc=");
      Serial.println(mqttClient.state());
    }
  }
}

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  esp_camera_init(&config);
}

void logToSD(String event) {
  File file = SD_MMC.open("/log.csv", FILE_APPEND);
  if(!file) return;
  time_t now = time(nullptr);
  file.printf("%ld,%s,%d,%d,%d,%d,%d\n", now, event.c_str(), lastPIR, lastLDR, relayState, gasValue, buzzerState);
  file.close();
}

void captureImage() {
  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb) return;
  String path = "/img_" + String(time(nullptr)) + ".jpg";
  File file = SD_MMC.open(path.c_str(), FILE_WRITE);
  if(file) {
    file.write(fb->buf, fb->len);
    file.close();
  }
  esp_camera_fb_return(fb);
}

void publishImage() {
  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb) return;
  mqttClient.beginPublish(imageTopic.c_str(), fb->len, false);
  mqttClient.write(fb->buf, fb->len);
  mqttClient.endPublish();
  esp_camera_fb_return(fb);
}

void publishStatus() {
  StaticJsonDocument<512> doc;
  doc["pir"] = lastPIR;
  doc["ldr"] = lastLDR;
  doc["pwr"] = lastPower;
  doc["relay"] = relayState;
  doc["buzzer"] = buzzerState;
  doc["gas"] = gasValue;
  doc["temp"] = bmp.readTemperature();
  doc["pres"] = bmp.readPressure() / 100.0F;

  char buffer[512];
  serializeJson(doc, buffer);

  if (mqttClient.connected()) {
    mqttClient.publish(statusTopic.c_str(), buffer);
  }

  // Parallel Push to Firebase
  if (Firebase.ready()) {
    Firebase.RTDB.setString(&fbdo, FIREBASE_NODE "/status", buffer);
  }
}

void handleStatusReq() {
  StaticJsonDocument<512> doc;
  doc["pir"] = lastPIR;
  doc["ldr"] = lastLDR;
  doc["pwr"] = lastPower;
  doc["relay"] = relayState;
  doc["buzzer"] = buzzerState;
  doc["gas"] = gasValue;
  doc["temp"] = bmp.readTemperature();
  doc["pres"] = bmp.readPressure() / 100.0F;
  String json;
  serializeJson(doc, json);
  server.send(200, "application/json", json);
}

void handleStream() {
  WiFiClient client = server.client();
  String response = "HTTP/1.1 200 OK\r\n";
  response += "Content-Type: " + String(_STREAM_CONTENT_TYPE) + "\r\n";
  response += "\r\n";
  server.sendContent(response);

  while (true) {
    if (!client.connected()) break;
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) continue;

    size_t hlen = snprintf(NULL, 0, _STREAM_PART, fb->len);
    char * hbuf = (char *)malloc(hlen + 1);
    snprintf(hbuf, hlen + 1, _STREAM_PART, fb->len);

    server.sendContent(_STREAM_BOUNDARY);
    server.sendContent(hbuf, hlen);
    server.sendContent((char *)fb->buf, fb->len);

    free(hbuf);
    esp_camera_fb_return(fb);
    delay(1);
  }
}

void handleCommand(char* topic, byte* payload, unsigned int length) {
  String msg = "";
  for (int i = 0; i < length; i++) msg += (char)payload[i];
  if (msg == "RELAY_ON") { pcf.write(3, HIGH); relayState = true; }
  else if (msg == "RELAY_OFF") { pcf.write(3, LOW); relayState = false; }
  else if (msg == "BUZZER_ON") { pcf.write(4, HIGH); buzzerState = true; }
  else if (msg == "BUZZER_OFF") { pcf.write(4, LOW); buzzerState = false; }
  publishStatus();
}

void handleFirebaseStream(FirebaseStream data) {
  if (data.dataPath() == "/command") {
    String msg = data.stringData();
    Serial.println("Firebase Command Received: " + msg);

    // Convert String to byte* for handleCommand
    handleCommand((char*)cmdTopic.c_str(), (byte*)msg.c_str(), msg.length());
  }
}

void setupFirebase() {
  fbConfig.host = FIREBASE_HOST;
  fbConfig.api_key = FIREBASE_API_KEY;
  fbAuth.user.email = FIREBASE_USER_EMAIL;
  fbAuth.user.password = FIREBASE_USER_PASSWORD;

  Firebase.begin(&fbConfig, &fbAuth);
  Firebase.reconnectWiFi(true);

  if (!Firebase.RTDB.beginStream(&fbdo, FIREBASE_NODE)) {
    Serial.printf("Firebase Stream begin error, %s\n\n", fbdo.errorReason().c_str());
  }
  Firebase.RTDB.setStreamCallback(&fbdo, handleFirebaseStream, [](bool timeout) {
    if (timeout) Serial.println("Firebase Stream timeout, resuming...");
  });
}

void setup() {
  Serial.begin(115200);
  Wire.begin(I2C_SDA, I2C_SCL);
  pcf.begin();
  for(int i=0; i<3; i++) pcf.pinMode(i, INPUT);
  pcf.pinMode(3, OUTPUT);
  pcf.pinMode(4, OUTPUT);

  bmp.begin(0x76);
  ads.begin();
  setupCamera();
  SD_MMC.begin();

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  configTime(0, 0, "pool.ntp.org");

  mqttClient.setServer(mqttBroker.c_str(), mqttPort);
  mqttClient.setCallback(handleCommand);

  setupFirebase();

  server.on("/stream", handleStream);
  server.on("/status", handleStatusReq);
  server.on("/control", []() {
    String cmd = server.arg("cmd");
    if (cmd == "RELAY_ON") { pcf.write(3, HIGH); relayState = true; }
    else if (cmd == "RELAY_OFF") { pcf.write(3, LOW); relayState = false; }
    else if (cmd == "BUZZER_ON") { pcf.write(4, HIGH); buzzerState = true; }
    else if (cmd == "BUZZER_OFF") { pcf.write(4, LOW); buzzerState = false; }
    server.send(200, "text/plain", "OK");
    publishStatus();
  });
  server.on("/config", HTTP_POST, []() {
    if (server.hasArg("plain")) {
      StaticJsonDocument<256> doc;
      deserializeJson(doc, server.arg("plain"));
      if (doc.containsKey("mqtt_broker")) mqttBroker = doc["mqtt_broker"].as<String>();
      if (doc.containsKey("mqtt_port")) mqttPort = doc["mqtt_port"].as<int>();
      mqttClient.setServer(mqttBroker.c_str(), mqttPort);
      server.send(200, "text/plain", "Config Updated");
    }
  });
  server.begin();
}

void loop() {
  server.handleClient();
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttClient.connected()) {
      reconnectMqtt();
    }
  }
  mqttClient.loop();

  bool currentPIR = pcf.read(0);
  bool currentLDR = pcf.read(1);
  bool currentPower = pcf.read(2);
  gasValue = ads.readADC_SingleEnded(0);

  bool changed = false;
  if (currentPIR != lastPIR) { lastPIR = currentPIR; changed = true; if(lastPIR) captureImage(); }
  if (currentLDR != lastLDR) { lastLDR = currentLDR; changed = true; }
  if (currentPower != lastPower) { lastPower = currentPower; changed = true; }

  if (lastPIR && lastLDR) {
    pcf.write(3, HIGH); relayState = true;
  } else if (!lastPIR || !lastLDR) {
    pcf.write(3, LOW); relayState = false;
  }

  if (gasValue > 1500) {
    pcf.write(4, HIGH); buzzerState = true;
  } else {
    pcf.write(4, LOW); buzzerState = false;
  }

  if (changed || millis() - lastStatusMillis > statusInterval) {
    publishStatus();
    if (changed) publishImage();
    lastStatusMillis = millis();
  }
  delay(10);
}
