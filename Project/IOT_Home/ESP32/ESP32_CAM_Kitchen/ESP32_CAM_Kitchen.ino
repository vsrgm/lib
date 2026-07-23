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

const String SW_VERSION = "1.0.304";

// State Variables
bool lastPIR = false;
bool lastPower = false;
bool lastGasDigital = false;
bool relayState = false;
bool buzzerState = false;
bool manualOverride = false;
int gasAnalog = 0;
int ldrAnalog = 0;
float lm358Temp = 0.0;
float bmpTemp = 0.0;
float bmpPres = 0.0;

int configWidth = 640;
int configHeight = 480;

unsigned long lastStatusMillis = 0;
const long statusInterval = 30000;

void reconnectMqtt() {
  static unsigned long lastReconnectAttempt = 0;
  if (millis() - lastReconnectAttempt > 5000) {
    lastReconnectAttempt = millis();
    mqttPort = mqttPorts[currentMqttPortIndex];
    currentMqttPortIndex = (currentMqttPortIndex + 1) % numMqttPorts;
    mqttClient.setServer(mqttBroker.c_str(), mqttPort);
    if (mqttClient.connect(clientId.c_str())) {
      mqttClient.subscribe(cmdTopic.c_str());
      publishStatus();
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
  file.printf("%ld,%s,PIR:%d,PWR:%d,REL:%d,GAS_D:%d,GAS_A:%d,LM358:%.1f,BMP_T:%.1f,LDR:%d,MAN:%d\n",
              now, event.c_str(), lastPIR, lastPower, relayState, lastGasDigital, gasAnalog, lm358Temp, bmpTemp, ldrAnalog, manualOverride);
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
  StaticJsonDocument<1024> doc;
  doc["pir"] = lastPIR;
  doc["ldr"] = ldrAnalog;
  doc["pwr"] = lastPower;
  doc["relay"] = relayState;
  doc["buzzer"] = buzzerState;
  doc["gas_d"] = lastGasDigital;
  doc["gas_a"] = gasAnalog;
  doc["temp_lm358"] = lm358Temp;
  doc["temp_bmp"] = bmpTemp;
  doc["pres"] = bmpPres;
  doc["manual_override"] = manualOverride;
  doc["v_w"] = configWidth;
  doc["v_h"] = configHeight;
  doc["ver"] = SW_VERSION;
  doc["ip"] = WiFi.localIP().toString();
  doc["id"] = clientId;

  char buffer[1024];
  serializeJson(doc, buffer);
  if (mqttClient.connected()) mqttClient.publish(statusTopic.c_str(), buffer);
  if (Firebase.ready()) Firebase.RTDB.setString(&fbdo, FIREBASE_NODE "/status", buffer);
}

void handleStatusReq() {
  StaticJsonDocument<1024> doc;
  doc["pir"] = lastPIR;
  doc["ldr"] = ldrAnalog;
  doc["pwr"] = lastPower;
  doc["relay"] = relayState;
  doc["buzzer"] = buzzerState;
  doc["gas_d"] = lastGasDigital;
  doc["gas_a"] = gasAnalog;
  doc["temp_lm358"] = lm358Temp;
  doc["temp_bmp"] = bmpTemp;
  doc["pres"] = bmpPres;
  doc["manual_override"] = manualOverride;
  doc["ver"] = SW_VERSION;
  doc["ip"] = WiFi.localIP().toString();
  doc["id"] = clientId;
  String json;
  serializeJson(doc, json);
  server.send(200, "application/json", json);
}

void handleStream() {
  WiFiClient client = server.client();
  String response = "HTTP/1.1 200 OK\r\nContent-Type: " + String(_STREAM_CONTENT_TYPE) + "\r\n\r\n";
  server.sendContent(response);
  while (true) {
    if (!client.connected()) break;
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) continue;
    server.sendContent(_STREAM_BOUNDARY);
    String header = "Content-Type: image/jpeg\r\nContent-Length: " + String(fb->len) + "\r\n\r\n";
    server.sendContent(header);
    server.sendContent((char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    delay(1);
  }
}

void setRelay(bool on) { relayState = on; pcf.write(2, on ? HIGH : LOW); }
void setBuzzer(bool on) { buzzerState = on; pcf.write(3, on ? HIGH : LOW); }

void applyResolution() {
  sensor_t * s = esp_camera_sensor_get();
  if (!s) return;

  framesize_t fs = FRAMESIZE_VGA; // Default
  if (configWidth <= 160) fs = FRAMESIZE_QQVGA;
  else if (configWidth <= 320) fs = FRAMESIZE_QVGA;
  else if (configWidth <= 640) fs = FRAMESIZE_VGA;
  else if (configWidth <= 800) fs = FRAMESIZE_SVGA;
  else if (configWidth <= 1024) fs = FRAMESIZE_XGA;
  else if (configWidth <= 1280) fs = FRAMESIZE_SXGA;
  else fs = FRAMESIZE_UXGA;

  s->set_framesize(s, fs);
  Serial.printf("Resolution updated to %d x %d (FS:%d)\n", configWidth, configHeight, fs);
}

void handleCommand(char* topic, byte* payload, unsigned int length) {
  String msg = "";
  for (int i = 0; i < length; i++) msg += (char)payload[i];
  if (msg == "RELAY_ON") setRelay(true);
  else if (msg == "RELAY_OFF") setRelay(false);
  else if (msg == "BUZZER_ON") setBuzzer(true);
  else if (msg == "BUZZER_OFF") setBuzzer(false);
  else if (msg == "OVERRIDE_ON") manualOverride = true;
  else if (msg == "OVERRIDE_OFF") manualOverride = false;
  else if (msg == "CAPTURE") { captureImage(); publishImage(); }
  else if (msg.startsWith("RES:")) {
      int comma = msg.indexOf(',');
      if (comma != -1) {
          configWidth = msg.substring(4, comma).toInt();
          configHeight = msg.substring(comma + 1).toInt();
          applyResolution();
      }
  }
  publishStatus();
}

void handleFirebaseStream(FirebaseStream data) {
  if (data.dataPath() == "/command") {
    handleCommand((char*)cmdTopic.c_str(), (byte*)data.stringData().c_str(), data.stringData().length());
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(I2C_SDA, I2C_SCL);
  pcf.begin();
  pcf.pinMode(0, INPUT); // PIR
  pcf.pinMode(1, INPUT); // PWR
  pcf.pinMode(2, OUTPUT); // RELAY
  pcf.pinMode(3, OUTPUT); // BUZZER
  pcf.pinMode(4, INPUT); // GAS DIGITAL

  bmp.begin(0x76);
  ads.begin();
  setupCamera();
  SD_MMC.begin();
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  configTime(5.5 * 3600, 0, "pool.ntp.org");
  mqttClient.setServer(mqttBroker.c_str(), mqttPort);
  mqttClient.setCallback(handleCommand);

  fbConfig.host = FIREBASE_HOST;
  fbConfig.api_key = FIREBASE_API_KEY;
  fbAuth.user.email = FIREBASE_USER_EMAIL;
  fbAuth.user.password = FIREBASE_USER_PASSWORD;
  Firebase.begin(&fbConfig, &fbAuth);
  Firebase.reconnectWiFi(true);
  Firebase.RTDB.beginStream(&fbdo, FIREBASE_NODE);
  Firebase.RTDB.setStreamCallback(&fbdo, handleFirebaseStream, [](bool timeout){});

  server.on("/stream", handleStream);
  server.on("/status", handleStatusReq);
  server.on("/control", []() {
    handleCommand((char*)"", (byte*)server.arg("cmd").c_str(), server.arg("cmd").length());
    server.send(200, "text/plain", "OK");
  });
  server.begin();
}

void loop() {
  server.handleClient();
  if (WiFi.status() == WL_CONNECTED && !mqttClient.connected()) reconnectMqtt();
  mqttClient.loop();

  bool currentPIR = pcf.read(0);
  bool currentPower = pcf.read(1);
  bool currentGasDigital = pcf.read(4);

  ldrAnalog = ads.readADC_SingleEnded(0);
  lm358Temp = ads.readADC_SingleEnded(1) * 0.125; // Simple calibration for LM358
  gasAnalog = ads.readADC_SingleEnded(2);
  bmpTemp = bmp.readTemperature();
  bmpPres = bmp.readPressure() / 100.0F;

  bool changed = (currentPIR != lastPIR || currentPower != lastPower || currentGasDigital != lastGasDigital);
  if (currentPIR && !lastPIR) { captureImage(); logToSD("Motion Detected"); }

  lastPIR = currentPIR;
  lastPower = currentPower;
  lastGasDigital = currentGasDigital;

  if (!manualOverride) {
    // Logic: Low light (LDR high value) and PIR Motion -> Kitchen lamp ON
    if (lastPIR && ldrAnalog > 15000) { if(!relayState) { setRelay(true); logToSD("Auto Lamp ON"); changed = true; } }
    else if (ldrAnalog < 10000) { if(relayState) { setRelay(false); logToSD("Auto Lamp OFF"); changed = true; } }

    // Gas leak logic
    if (lastGasDigital || gasAnalog > 20000) { if(!buzzerState) { setBuzzer(true); logToSD("Gas Alarm ON"); changed = true; } }
    else { if(buzzerState) { setBuzzer(false); logToSD("Gas Alarm OFF"); changed = true; } }
  }

  if (changed || millis() - lastStatusMillis > statusInterval) {
    publishStatus();
    // Requirements: Capture image at every system state change for MQTT/Firebase update
    if (changed) {
        captureImage();
        publishImage();
    }
    lastStatusMillis = millis();
  }
  delay(10);
}
