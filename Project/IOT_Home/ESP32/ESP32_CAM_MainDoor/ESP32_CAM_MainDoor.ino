#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>
#include <ArduinoOTA.h>
#include <PubSubClient.h>
#include <Wire.h>

// Resolve sensor_t naming conflict between esp_camera and Adafruit Unified Sensor
#define sensor_t adafruit_sensor_t
#include <Adafruit_BMP280.h>
#undef sensor_t

#include "PCF8574.h"
#include "FS.h"
#include "SD_MMC.h"
#include <ArduinoJson.h>
#include <Firebase_ESP_Client.h>
#include "credentials.h"
#include <time.h>
#include <WebServer.h>
#include <ElegantOTA.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

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

// I2C on GPIO 13/15 (SD Data 3 and CMD pins freed)
#define I2C_SDA 13
#define I2C_SCL 15
#define PCF_INT 2 // Safe pin avoiding PSRAM conflict
#define FLASH_LED_GPIO 4

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

PCF8574 pcf(0x20);
Adafruit_BMP280 bmp(&Wire);

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
WebServer server(80);

// Firebase Data objects
FirebaseData fbdo;
FirebaseData fbdoPush;
FirebaseAuth fbAuth;
FirebaseConfig fbConfig;

const char* ssid = HOME_NETWORK_SSID;
const char* password = HOME_NETWORK_PASSWORD;
String mqttBroker = MQTT_BROKER;
int mqttPort = MQTT_PORT;
int currentMqttPortIndex = 0;
const int mqttPorts[] = {1883, 8000, 8883, 8884};
const int numMqttPorts = 4;

String clientId = "ESP32_MainDoor_" + String((uint32_t)ESP.getEfuseMac(), HEX);
String statusTopic = "FrmEsp32/Securitymaindoor/status";
String cmdTopic = "FrmMobile/esp32cam/Securitymaindoor/command";
String imageTopic = "FrmEsp32/Securitymaindoor/image";

const String SW_VERSION = "1.0.328";

bool lastPIR = false;
bool lastDoor = false;
bool lastBell = false;
bool lastLDR = false;
bool lastPower = false;
bool relayState = false;
bool buzzerState = false;
bool flashState = false;
bool cameraOK = false;

unsigned long lastStatusMillis = 0;
const long statusInterval = 30000; // 30 seconds

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
  // Manual power cycle for OV3660
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, HIGH);
  delay(100);
  digitalWrite(PWDN_GPIO_NUM, LOW);
  delay(200);

  // Software pull-ups for SCCB to help weak hardware pull-ups
  pinMode(SIOD_GPIO_NUM, INPUT_PULLUP);
  pinMode(SIOC_GPIO_NUM, INPUT_PULLUP);

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
  config.xclk_freq_hz = 10000000; // Lowered to 10MHz for probe stability
  config.pixel_format = PIXFORMAT_JPEG;

  if(psramFound()){
    config.frame_size = FRAMESIZE_SVGA; // Balanced resolution for stream stability
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    cameraOK = false;
    return;
  }
  cameraOK = true;

  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL && s->id.PID == 0x3660) {
    s->set_vflip(s, 1);
    s->set_hmirror(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
    Serial.println("OV3660 Camera detected and optimized");
  }
}

void logToSD(String event) {
  File file = SD_MMC.open("/log.csv", FILE_APPEND);
  if(!file) {
    Serial.println("Failed to open log file");
    return;
  }

  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char timeStr[25];
  strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &timeinfo);

  String data = String(timeStr) + "," + event + "," +
                String(lastPIR) + "," + String(lastDoor) + "," +
                String(lastBell) + "," + String(lastLDR) + "," +
                String(relayState) + "\n";
  file.print(data);
  file.close();
}

void captureImage() {
  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  time_t now = time(nullptr);
  String path = "/img_" + String(now) + ".jpg";
  File file = SD_MMC.open(path.c_str(), FILE_WRITE);
  if(file) {
    file.write(fb->buf, fb->len);
    file.close();
    Serial.println("Image saved: " + path);
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
  doc["door"] = lastDoor;
  doc["bell"] = lastBell;
  doc["ldr"] = lastLDR;
  doc["pwr"] = lastPower;
    doc["relay"] = relayState;
    doc["buzzer"] = buzzerState;
    doc["flash"] = flashState;
    doc["ver"] = SW_VERSION;
    doc["ip"] = WiFi.localIP().toString();
    doc["id"] = clientId;

  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char timeStr[25];
  strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &timeinfo);
  doc["ts"] = timeStr;

  float temp = bmp.readTemperature();
  float pres = bmp.readPressure() / 100.0F;
  doc["temp"] = temp;
  doc["pres"] = pres;
  doc["heap"] = ESP.getFreeHeap();

  // For MJPEG over MQTT, we can send a low-res image if requested
  // However, Requirement 5.2 says MJPEG over MQTT. We'll send the latest path or a small base64 if needed.
  // For now, let's keep it simple and focus on status.

  char buffer[1024];
  serializeJson(doc, buffer);

  if (mqttClient.connected()) {
    mqttClient.publish(statusTopic.c_str(), buffer);
  }

  // Parallel Push to Firebase
  if (Firebase.ready()) {
    FirebaseJson json;
    json.setJsonData(buffer);
    if (Firebase.RTDB.setJSON(&fbdoPush, "FrmEsp32/Securitymaindoor/status", &json)) {
      Serial.println("Firebase status updated successfully");
    } else {
      Serial.print("Firebase status update failed: ");
      Serial.println(fbdoPush.errorReason());
    }
  } else {
    Serial.println("Firebase not ready");
  }
}

void handleStatusReq() {
  StaticJsonDocument<512> doc;
  doc["pir"] = lastPIR;
  doc["door"] = lastDoor;
  doc["bell"] = lastBell;
  doc["ldr"] = lastLDR;
  doc["pwr"] = lastPower;
    doc["relay"] = relayState;
    doc["buzzer"] = buzzerState;
    doc["flash"] = flashState;
    doc["ver"] = SW_VERSION;
    doc["ip"] = WiFi.localIP().toString();
    doc["id"] = clientId;

  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char timeStr[25];
  strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &timeinfo);
  doc["ts"] = timeStr;

  doc["temp"] = bmp.readTemperature();
  doc["pres"] = bmp.readPressure() / 100.0F;
  String json;
  serializeJson(doc, json);
  server.send(200, "application/json", json);
}

httpd_handle_t stream_httpd = NULL;

esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  if (!cameraOK) {
    return httpd_resp_send_404(req);
  }

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if(res != ESP_OK) return res;

  while(true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
    } else {
      size_t hlen = snprintf(part_buf, 64, _STREAM_PART, fb->len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
      if(res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      if(res == ESP_OK) res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
      esp_camera_fb_return(fb);
    }
    if(res != ESP_OK) break;
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
  return res;
}

void startStreamServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  // config.ctrl_port = 81; // Removed as it conflicts with default behavior
  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("Stream server started on port 81");
  }
}

void handleStream() {
  String url = "http://" + WiFi.localIP().toString() + ":81/stream";
  server.sendHeader("Location", url, true);
  server.send(302, "text/plain", "");
}

void handleCommand(char* topic, byte* payload, unsigned int length) {
  String msg = "";
  for (int i = 0; i < length; i++) msg += (char)payload[i];

  if (msg == "RELAY_ON") {
    relayState = true;
    pcf.write(5, HIGH);
    publishStatus();
  } else if (msg == "RELAY_OFF") {
    relayState = false;
    pcf.write(5, LOW);
    publishStatus();
  } else if (msg == "BUZZER_ON") {
    buzzerState = true;
    pcf.write(6, HIGH);
    publishStatus();
  } else if (msg == "BUZZER_OFF") {
    buzzerState = false;
    pcf.write(6, LOW);
    publishStatus();
  } else if (msg == "FLASH_ON") {
    flashState = true;
    digitalWrite(FLASH_LED_GPIO, HIGH);
    publishStatus();
  } else if (msg == "FLASH_OFF") {
    flashState = false;
    digitalWrite(FLASH_LED_GPIO, LOW);
    publishStatus();
  } else if (msg == "CAPTURE") {
    captureImage();
  }
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
  fbConfig.database_url = FIREBASE_HOST;
  fbConfig.api_key = FIREBASE_API_KEY;
  fbAuth.user.email = FIREBASE_USER_EMAIL;
  fbAuth.user.password = FIREBASE_USER_PASSWORD;

  // Ensure host has https://
  String host = FIREBASE_HOST;
  if (!host.startsWith("http")) {
      fbConfig.database_url = "https://" + host;
  }

  Firebase.begin(&fbConfig, &fbAuth);
  Firebase.reconnectWiFi(true);

  String streamPath = String("/") + FIREBASE_NODE;
  if (!Firebase.RTDB.beginStream(&fbdo, streamPath.c_str())) {
    Serial.printf("Firebase Stream begin error, %s\n\n", fbdo.errorReason().c_str());
  }
  Firebase.RTDB.setStreamCallback(&fbdo, handleFirebaseStream, [](bool timeout) {
    if (timeout) Serial.println("Firebase Stream timeout, resuming...");
  });
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Disable brownout detector

  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\n=== SYSTEM BOOTING (Camera Focus) ===");

  // 1. Initialize Camera FIRST
  setupCamera();

  // 2. Initialize I2C (GPIO 13/15)
  Serial.println("Initializing I2C on GPIO 13 (SDA) / 15 (SCL)");
  Wire.begin(I2C_SDA, I2C_SCL);
  pcf.begin();
  for(int i=0; i<5; i++) pcf.write(i, HIGH);
  pcf.write(7, HIGH);

  pinMode(FLASH_LED_GPIO, OUTPUT);
  digitalWrite(FLASH_LED_GPIO, LOW);

  // BMP280 and SD Card IGNORED for now
  Serial.println("BMP280 and SD Card skipped to focus on Camera");

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  configTime(0, 0, "pool.ntp.org");
  setenv("TZ", "IST-5:30", 1);
  tzset();

  mqttClient.setServer(mqttBroker.c_str(), mqttPort);
  mqttClient.setCallback(handleCommand);

  setupFirebase();
  Serial.println("Firebase setup initiated");
  startStreamServer();

  ElegantOTA.begin(&server); // Enable Web-based OTA at /update

  ArduinoOTA.setHostname("esp32-cam-maindoor");
  ArduinoOTA.onStart([]() { Serial.println("OTA Start"); });
  ArduinoOTA.onEnd([]() { Serial.println("\nOTA End"); });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("Error[%u]: ", error);
  });
  ArduinoOTA.begin();

  server.on("/", []() {
    String html = "<html><head><title>ESP32-CAM Smart Door</title>";
    html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
    html += "<style>body{font-family:sans-serif; background:#f4f4f4; text-align:center; margin:0; padding:20px;} ";
    html += ".card{background:white; padding:20px; border-radius:10px; box-shadow:0 4px 8px rgba(0,0,0,0.1); max-width:400px; margin:auto;} ";
    html += "h1{color:#333;} .btn{display:block; width:100%; padding:12px; margin:10px 0; background:#007bff; color:white; text-decoration:none; border-radius:5px; font-weight:bold;} ";
    html += ".btn-on{background:#28a745;} .btn-off{background:#dc3545;} ";
    html += "img{max-width:100%; border-radius:5px; margin-bottom:15px; border:1px solid #ddd;} ";
    html += ".status-val{font-weight:bold; color:#555;}</style></head><body>";
    html += "<div class='card'><h1>Smart Door</h1>";
    html += "<p>Version: " + SW_VERSION + "</p>";
    html += "<img src='/stream' alt='Live Stream'>";
    html += "<div style='text-align:left; margin-bottom:20px;'>";
    html += "<p>Camera: <span class='status-val'>" + String(cameraOK ? "Ready" : "Not Found") + "</span></p>";
    html += "<p>PIR: <span class='status-val'>" + String(lastPIR ? "Detected" : "Clear") + "</span></p>";
    html += "<p>Door: <span class='status-val'>" + String(lastDoor ? "Open" : "Closed") + "</span></p>";
    html += "</div><hr>";
    html += "<a href='/control?cmd=RELAY_ON' class='btn btn-on'>Unlock Door</a>";
    html += "<a href='/control?cmd=RELAY_OFF' class='btn btn-off'>Lock Door</a>";
    html += "<a href='/control?cmd=FLASH_ON' class='btn'>Flash ON</a>";
    html += "<a href='/control?cmd=FLASH_OFF' class='btn'>Flash OFF</a>";
    html += "<a href='/update' class='btn' style='background:#34495e;'>Web OTA Update</a>";
    html += "<a href='/status' class='btn' style='background:#6c757d;'>Refresh Data</a>";
    html += "</div></body></html>";
    server.send(200, "text/html", html);
  });
  server.on("/stream", handleStream);
  server.on("/status", handleStatusReq);
  server.on("/control", []() {
    String cmd = server.arg("cmd");
    if (cmd == "RELAY_ON") { relayState = true; pcf.write(5, HIGH); }
    else if (cmd == "RELAY_OFF") { relayState = false; pcf.write(5, LOW); }
    else if (cmd == "BUZZER_ON") { buzzerState = true; pcf.write(6, HIGH); }
    else if (cmd == "BUZZER_OFF") { buzzerState = false; pcf.write(6, LOW); }
    else if (cmd == "FLASH_ON") { flashState = true; digitalWrite(FLASH_LED_GPIO, HIGH); }
    else if (cmd == "FLASH_OFF") { flashState = false; digitalWrite(FLASH_LED_GPIO, LOW); }
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
  ArduinoOTA.handle();
  server.handleClient();
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttClient.connected()) {
      reconnectMqtt();
    }
  }
  mqttClient.loop();

  bool currentPIR = pcf.read(0);
  bool currentDoor = pcf.read(1);
  bool currentBell = pcf.read(2);
  bool currentLDR = pcf.read(3);
  bool currentPower = pcf.read(4);

  bool changed = false;
  if (currentPIR != lastPIR) { lastPIR = currentPIR; changed = true; if(lastPIR) Serial.println("PIR Event"); }
  if (currentDoor != lastDoor) { lastDoor = currentDoor; changed = true; Serial.println("Door Event"); }
  if (currentBell != lastBell) { lastBell = currentBell; changed = true; Serial.println("Bell Event"); }
  if (currentLDR != lastLDR) {
    lastLDR = currentLDR; changed = true;
    if (lastLDR) {
       pcf.write(5, HIGH); relayState = true;
    } else {
       pcf.write(5, LOW); relayState = false;
    }
  }
  if (currentPower != lastPower) { lastPower = currentPower; changed = true; }

  if (changed || millis() - lastStatusMillis > statusInterval) {
    publishStatus();
    if (changed) publishImage();
    lastStatusMillis = millis();
  }

  delay(10);
}
