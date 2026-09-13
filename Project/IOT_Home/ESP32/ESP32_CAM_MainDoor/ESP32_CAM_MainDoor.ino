#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>
#include <ArduinoOTA.h>

// Compile-time feature flags (Disabled by default)
// #define ENABLE_MQTT
// #define ENABLE_WATCHDOG

#ifdef ENABLE_MQTT
#include <PubSubClient.h>
#endif
#include <Wire.h>

// Resolve sensor_t naming conflict between esp_camera and Adafruit Unified Sensor
#define sensor_t adafruit_sensor_t
#include <Adafruit_BMP280.h>
#undef sensor_t

#include <PCF8574.h>
#include "FS.h"
#include "SD_MMC.h"
#include <ArduinoJson.h>
#include <Firebase_ESP_Client.h>
#include <base64.h>
#include "credentials.h"
#include <time.h>
#include <WebServer.h>
#include <ElegantOTA.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#ifdef ENABLE_WATCHDOG
#include <esp_task_wdt.h>
// Watchdog timeouts
#define WDT_TIMEOUT_NORMAL 10
#define WDT_TIMEOUT_UPDATE 60
#endif

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
#define XCLK_MCLK         20000000
#define I2C_RATE          400000

// Feasible I2C mapping keeping Serial Programming (GPIO 1/3) available
#define I2C_SDA 13
#define I2C_SCL 12
#define FLASH_LED_GPIO 4

// PCF8574 (LCD Adapter Module) Repurposed General I/O Mapping
#define PCF_PIR    0
#define PCF_DOOR   1
#define PCF_BELL   2
#define PCF_LDR    4
#define PCF_POWER  5
#define PCF_RELAY  6
#define PCF_BUZZER 7

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

PCF8574 pcf(0x27); // Default I2C address for common PCF8574 LCD backpack module
Adafruit_BMP280 bmp(&Wire);

#ifdef ENABLE_MQTT
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
#endif
WebServer server(80);

// Firebase Data objects
FirebaseData fbdo;
FirebaseData fbdoPush;
FirebaseAuth fbAuth;
FirebaseConfig fbConfig;

const char* ssid = HOME_NETWORK_SSID;
const char* password = HOME_NETWORK_PASSWORD;

#ifdef ENABLE_MQTT
String mqttBroker = MQTT_BROKER;
int mqttPort = MQTT_PORT;
int currentMqttPortIndex = 0;
const int mqttPorts[] = {1883, 8000, 8883, 8884};
const int numMqttPorts = 4;
String imageTopic = "FrmEsp32/main_door/image";
#endif

String clientId = "ESP32_MainDoor_" + String((uint32_t)ESP.getEfuseMac(), HEX);
String statusTopic = "FrmEsp32/main_door/status";
String cmdTopic = "FrmMobile/main_door/command";

const String SW_VERSION = "1.0.376";

bool lastPIR = false;
bool lastDoor = false;
bool lastBell = false;
bool lastLDR = false;
bool lastPower = false;
bool relayState = false;
bool buzzerState = false;
bool flashState = false;
bool cameraOK = false;
bool otaInProgress = false;
bool isLiveRequested = false;

unsigned long lastClientHeartbeat = 0;
unsigned long lastStatusMillis = 0;
const long statusInterval = 30000; // 30 seconds

#ifdef ENABLE_MQTT
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
#endif

void I2C_ClearBus(int sda, int scl) {
  pinMode(sda, INPUT_PULLUP);
  pinMode(scl, OUTPUT);
  for (int i = 0; i < 9; i++) {
    digitalWrite(scl, LOW);
    delayMicroseconds(10);
    digitalWrite(scl, HIGH);
    delayMicroseconds(10);
  }
  // Send a STOP signal
  pinMode(sda, OUTPUT);
  digitalWrite(sda, LOW);
  delayMicroseconds(10);
  digitalWrite(sda, HIGH);
}

void setupCamera() {
  Serial.println("--- Starting Camera Initialization Sequence ---");

  // Step 0: Clear the Camera I2C bus (Pins 26 and 27)
  I2C_ClearBus(SIOD_GPIO_NUM, SIOC_GPIO_NUM);

  // Step 1: Hard Reset via PowerDown pin
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, HIGH); // Power down
  delay(100);
  digitalWrite(PWDN_GPIO_NUM, LOW);  // Power up
  delay(200);

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

  // Let the driver handle reset if connected, else we already did PWDN
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  // Reduced XCLK for better I2C stability during init
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  if(psramFound()){
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);

  // If failed at 10MHz, try one last time at 20MHz (Standard)
  if (err != ESP_OK) {
      Serial.println("Retrying camera init at standard 20MHz...");
      esp_camera_deinit();
      config.xclk_freq_hz = 20000000;
      err = esp_camera_init(&config);
  }

  if (err != ESP_OK) {
    Serial.printf("[Camera Error] Driver init failed: 0x%x\n", err);
    cameraOK = false;
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
      s->set_brightness(s, 1);
      s->set_saturation(s, -2);
      // Flip for door mounting orientation
      s->set_vflip(s, 1);
      s->set_hmirror(s, 1);
  }

  cameraOK = true;
  Serial.println("[Camera Success] Camera ready.");
}

void logToSD(String event) {
  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char timeStr[25];
  strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &timeinfo);

  String newRow = String(timeStr) + "," + event + "," +
                  String(lastPIR) + "," + String(lastDoor) + "," +
                  String(lastBell) + "," + String(lastLDR) + "," +
                  String(lastPower) + "," + String(relayState) + "," +
                  String(buzzerState) + "\n";

  if (!SD_MMC.exists("/log.csv")) {
    File file = SD_MMC.open("/log.csv", FILE_WRITE);
    if (file) {
      file.print(newRow);
      file.close();
    }
  } else {
    File oldFile = SD_MMC.open("/log.csv", FILE_READ);
    File tmpFile = SD_MMC.open("/log_tmp.csv", FILE_WRITE);
    if (oldFile && tmpFile) {
      tmpFile.print(newRow);
      uint8_t buf[512];
      while (oldFile.available()) {
        int len = oldFile.read(buf, sizeof(buf));
        tmpFile.write(buf, len);
      }
      oldFile.close();
      tmpFile.close();
      SD_MMC.remove("/log.csv");
      SD_MMC.rename("/log_tmp.csv", "/log.csv");
    } else {
      if (oldFile) oldFile.close();
      if (tmpFile) tmpFile.close();
      File file = SD_MMC.open("/log.csv", FILE_APPEND);
      if (file) {
        file.print(newRow);
        file.close();
      }
    }
  }
}

void captureImageToSD() {
  if (!cameraOK) return;
  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char filename[32];
  strftime(filename, sizeof(filename), "/img_%Y%m%d_%H%M%S.jpg", &timeinfo);

  File file = SD_MMC.open(filename, FILE_WRITE);
  if(file) {
    file.write(fb->buf, fb->len);
    file.close();
    Serial.println("Image saved to SD: " + String(filename));
  }
  esp_camera_fb_return(fb);
}

void publishImage() {
  if (!cameraOK) return;
  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb) return;

#ifdef ENABLE_MQTT
  if (mqttClient.connected()) {
    mqttClient.beginPublish(imageTopic.c_str(), fb->len, false);
    mqttClient.write(fb->buf, fb->len);
    mqttClient.endPublish();
  }
#endif

  if (Firebase.ready()) {
    String base64Img = base64::encode(fb->buf, fb->len);
    if (Firebase.RTDB.setString(&fbdoPush, "FrmEsp32/main_door/last_image", base64Img.c_str())) {
      Serial.println("Firebase last_image updated successfully");
    }
  }
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
  doc["temp"] = isnan(temp) ? 0.0 : temp;
  doc["pres"] = isnan(pres) ? 0.0 : pres;
  doc["heap"] = ESP.getFreeHeap();

  char buffer[1024];
  serializeJson(doc, buffer);

#ifdef ENABLE_MQTT
  if (mqttClient.connected()) {
    mqttClient.publish(statusTopic.c_str(), buffer);
  }
#endif

  if (Firebase.ready()) {
    FirebaseJson json;
    json.setJsonData(buffer);
    Firebase.RTDB.setJSON(&fbdoPush, "FrmEsp32/main_door/status", &json);
  }
}

void handleStatusReq() {
  lastClientHeartbeat = millis();
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

  float temp = bmp.readTemperature();
  float pres = bmp.readPressure() / 100.0F;
  doc["temp"] = isnan(temp) ? 0.0 : temp;
  doc["pres"] = isnan(pres) ? 0.0 : pres;

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

  lastClientHeartbeat = millis();
  while(!otaInProgress) {
    if (millis() - lastClientHeartbeat > 30000) {
      Serial.println("[Stream] Client heartbeat lost. Closing session.");
      break;
    }
#ifdef ENABLE_WATCHDOG
    esp_task_wdt_reset();
#endif

    fb = esp_camera_fb_get();
    if (!fb) {
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
    pcf.write(PCF_RELAY, HIGH);
    publishStatus();
  } else if (msg == "RELAY_OFF") {
    relayState = false;
    pcf.write(PCF_RELAY, LOW);
    publishStatus();
  } else if (msg == "BUZZER_ON") {
    buzzerState = true;
    pcf.write(PCF_BUZZER, HIGH);
    publishStatus();
  } else if (msg == "BUZZER_OFF") {
    buzzerState = false;
    pcf.write(PCF_BUZZER, LOW);
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
    captureImageToSD();
    publishImage();
  } else if (msg == "REBOOT") {
    Serial.println("Reboot command received. Restarting...");
    delay(1000);
    ESP.restart();
  } else if (msg.startsWith("CAM_SET:")) {
    // Format: CAM_SET:key:val
    int firstColon = msg.indexOf(':');
    int secondColon = msg.indexOf(':', firstColon + 1);
    if (secondColon != -1) {
      String key = msg.substring(firstColon + 1, secondColon);
      int val = msg.substring(secondColon + 1).toInt();
      sensor_t * s = esp_camera_sensor_get();
      if (s) {
        if (key == "framesize") s->set_framesize(s, (framesize_t)val);
        else if (key == "quality") s->set_quality(s, val);
        else if (key == "brightness") s->set_brightness(s, val);
        else if (key == "contrast") s->set_contrast(s, val);
        else if (key == "saturation") s->set_saturation(s, val);
        else if (key == "hmirror") s->set_hmirror(s, val);
        else if (key == "vflip") s->set_vflip(s, val);
        else if (key == "awb") s->set_whitebal(s, val);
        else if (key == "aec") s->set_exposure_ctrl(s, val);
        else if (key == "agc") s->set_gain_ctrl(s, val);
        else if (key == "special_effect") s->set_special_effect(s, val);
        else if (key == "wb_mode") s->set_wb_mode(s, val);
        Serial.printf("[Camera] Config %s set to %d\n", key.c_str(), val);
        publishStatus();
      }
    }
  }
}

void handleFirebaseStream(FirebaseStream data) {
  String path = data.dataPath();
  if (path == "/command") {
    String msg = data.stringData();
    handleCommand((char*)cmdTopic.c_str(), (byte*)msg.c_str(), msg.length());
  } else if (path == "/live_request") {
    isLiveRequested = data.boolData();
    Serial.printf("[Firebase] Live Request: %s\n", isLiveRequested ? "ON" : "OFF");

    // Adjust resolution to save bandwidth over cloud
    sensor_t * s = esp_camera_sensor_get();
    if (s) {
      if (isLiveRequested) s->set_framesize(s, FRAMESIZE_QVGA);
      else s->set_framesize(s, FRAMESIZE_SVGA);
    }
  }
}

void setupFirebase() {
  fbConfig.database_url = FIREBASE_HOST;
  fbConfig.api_key = FIREBASE_API_KEY;
  fbAuth.user.email = FIREBASE_USER_EMAIL;
  fbAuth.user.password = FIREBASE_USER_PASSWORD;

  String host = FIREBASE_HOST;
  if (!host.startsWith("http")) {
      fbConfig.database_url = "https://" + host;
  }

  Firebase.begin(&fbConfig, &fbAuth);
  Firebase.reconnectWiFi(true);

  // Harmonized Stream Path for Commands and Live Requests (Simplified)
  String streamPath = "FrmMobile/main_door";
  if (Firebase.RTDB.beginStream(&fbdo, streamPath.c_str())) {
    Firebase.RTDB.setStreamCallback(&fbdo, handleFirebaseStream, [](bool timeout) {});
  }
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== MAIN DOOR SECURITY CAM INITIALIZING ===");

  setupCamera();

  // Initialize Feasible I2C Bus on GPIO 13/12
  Wire.begin(I2C_SDA, I2C_SCL);
  pcf.begin();

  // Set PCF8574 Quasi-bidirectional pins for proper inputs
  pcf.write(PCF_PIR, HIGH);
  pcf.write(PCF_DOOR, HIGH);
  pcf.write(PCF_BELL, HIGH);
  pcf.write(PCF_LDR, HIGH);
  pcf.write(PCF_POWER, HIGH);

  pcf.write(PCF_RELAY, LOW);
  pcf.write(PCF_BUZZER, LOW);

  pinMode(FLASH_LED_GPIO, OUTPUT);
  digitalWrite(FLASH_LED_GPIO, LOW);

  // Initialize BMP280 Sensor
  if (!bmp.begin(0x76)) {
    bmp.begin(0x77);
  }

  // Initialize SD Card in 1-Bit mode to keep GPIO 12 & 13 free for I2C
  // This also frees up GPIO 4 (Flash) and GPIO 2
  if(!SD_MMC.begin("/sdcard", true)) {
    Serial.println("SD Card Mount Failed");
  }

  // RE-ASSERT control of Flash LED after SD Card initialization
  pinMode(FLASH_LED_GPIO, OUTPUT);
  digitalWrite(FLASH_LED_GPIO, LOW);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }

  configTime(0, 0, "pool.ntp.org");
  setenv("TZ", "IST-5:30", 1);
  tzset();

#ifdef ENABLE_MQTT
  mqttClient.setServer(mqttBroker.c_str(), mqttPort);
  mqttClient.setCallback(handleCommand);
#endif

  setupFirebase();
  startStreamServer();
  ElegantOTA.begin(&server);
  ElegantOTA.onStart([]() {
    Serial.println("ElegantOTA Update Started - Extending Watchdog to 60s");
    otaInProgress = true;
#ifdef ENABLE_WATCHDOG
    esp_task_wdt_config_t config = {
      .timeout_ms = WDT_TIMEOUT_UPDATE * 1000,
      .idle_core_mask = 0,
      .trigger_panic = true
    };
    esp_task_wdt_reconfigure(&config);
#endif
  });

  ArduinoOTA.onStart([]() {
    Serial.println("ArduinoOTA Start - Extending Watchdog to 60s");
    otaInProgress = true;
#ifdef ENABLE_WATCHDOG
    esp_task_wdt_config_t config = {
      .timeout_ms = WDT_TIMEOUT_UPDATE * 1000,
      .idle_core_mask = 0,
      .trigger_panic = true
    };
    esp_task_wdt_reconfigure(&config);
#endif
  });
  ArduinoOTA.begin();

  server.on("/", []() {
    String html = "<html><head><title>Smart Door Console</title>";
    html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
    html += "<style>body{font-family:sans-serif; background:#f0f2f5; text-align:center; margin:0; padding:15px;} ";
    html += ".card{background:white; padding:20px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.1); max-width:480px; margin:auto;} ";
    html += "h2{color:#1a73e8; margin-top:0;} .btn{display:block; width:100%; padding:14px; margin:10px 0; background:#1a73e8; color:white; text-decoration:none; border-radius:8px; font-weight:bold; border:none; cursor:pointer; box-sizing:border-box;} ";
    html += ".btn-sec{background:#5f6368;} .btn-warn{background:#d93025;} ";
    html += "img{max-width:100%; border-radius:8px; margin-bottom:15px; border:2px solid #e0e0e0; background:#000;} ";
    html += ".grid{display:grid; grid-template-columns: 1fr 1fr; gap:10px; text-align:left; margin:15px 0; font-size:14px;} ";
    html += ".label{font-weight:bold; color:#5f6368;} .val{color:#202124; float:right;}</style></head><body>";

    html += "<div class='card'><h2>Main Door Cam</h2>";
    html += "<p style='font-size:12px; color:#70757a;'>FW: " + SW_VERSION + " | IP: " + WiFi.localIP().toString() + "</p>";

    html += "<img src='/stream' alt='Live Video Feed' onerror=\"this.src='https://via.placeholder.com/400x300?text=Stream+Not+Active'\">";

    html += "<div class='grid'>";
    html += "<div><span class='label'>PIR:</span> <span class='val'>" + String(lastPIR ? "MOTION" : "Clear") + "</span></div>";
    html += "<div><span class='label'>Door:</span> <span class='val'>" + String(lastDoor ? "OPEN" : "Closed") + "</span></div>";
    html += "<div><span class='label'>Bell:</span> <span class='val'>" + String(lastBell ? "RINGING" : "Idle") + "</span></div>";
    html += "<div><span class='label'>LDR:</span> <span class='val'>" + String(lastLDR ? "Dark" : "Light") + "</span></div>";
    html += "<div><span class='label'>Temp:</span> <span class='val'>" + String(bmp.readTemperature(), 1) + " C</span></div>";
    html += "<div><span class='label'>Relay:</span> <span class='val'>" + String(relayState ? "ON" : "OFF") + "</span></div>";
    html += "</div><hr style='border:0; border-top:1px solid #eee;'>";

    html += "<a href='/control?cmd=RELAY_ON' class='btn'>Unlock Door</a>";
    html += "<a href='/control?cmd=RELAY_OFF' class='btn btn-sec'>Lock Door</a>";
    html += "<div style='display:flex; gap:10px;'>";
    html += "<a href='/update' class='btn btn-sec' style='flex:1;'>Update FW</a>";
    html += "<a href='/control?cmd=REBOOT' class='btn btn-warn' style='flex:1;' onclick=\"return confirm('Reboot device?')\">Reboot</a>";
    html += "</div></div></body></html>";

    server.send(200, "text/html", html);
  });
  server.on("/stream", handleStream);
  server.on("/status", handleStatusReq);
  server.on("/control", []() {
    String cmd = server.arg("cmd");
    if (cmd == "RELAY_ON") { relayState = true; pcf.write(PCF_RELAY, HIGH); }
    else if (cmd == "RELAY_OFF") { relayState = false; pcf.write(PCF_RELAY, LOW); }
    else if (cmd == "BUZZER_ON") { buzzerState = true; pcf.write(PCF_BUZZER, HIGH); }
    else if (cmd == "BUZZER_OFF") { buzzerState = false; pcf.write(PCF_BUZZER, LOW); }
    else if (cmd == "FLASH_ON") { flashState = true; digitalWrite(FLASH_LED_GPIO, HIGH); }
    else if (cmd == "FLASH_OFF") { flashState = false; digitalWrite(FLASH_LED_GPIO, LOW); }
    else if (cmd == "REBOOT") {
      server.send(200, "text/plain", "Rebooting...");
      delay(1000);
      ESP.restart();
    }
    server.send(200, "text/plain", "OK");
    publishStatus();
  });
  server.begin();

#ifdef ENABLE_WATCHDOG
  esp_task_wdt_config_t wdt_config = {
      .timeout_ms = WDT_TIMEOUT_NORMAL * 1000,
      .idle_core_mask = 0,
      .trigger_panic = true
  };
  if (esp_task_wdt_init(&wdt_config) != ESP_OK) {
      esp_task_wdt_reconfigure(&wdt_config);
  }
  esp_task_wdt_add(NULL);
#endif
}

void loop() {
#ifdef ENABLE_WATCHDOG
  esp_task_wdt_reset();
#endif
  ArduinoOTA.handle();
  server.handleClient();
#ifdef ENABLE_MQTT
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttClient.connected()) {
      reconnectMqtt();
    }
  }
  mqttClient.loop();
#endif

  bool currentPIR = pcf.read(PCF_PIR);
  bool currentDoor = pcf.read(PCF_DOOR);
  bool currentBell = pcf.read(PCF_BELL);
  bool currentLDR = pcf.read(PCF_LDR);
  bool currentPower = pcf.read(PCF_POWER);

  bool changed = false;
  if (currentPIR != lastPIR) { lastPIR = currentPIR; changed = true; }
  if (currentDoor != lastDoor) { lastDoor = currentDoor; changed = true; }
  if (currentBell != lastBell) { lastBell = currentBell; changed = true; }
  if (currentLDR != lastLDR) {
    lastLDR = currentLDR; changed = true;
    if (lastLDR) {
       pcf.write(PCF_RELAY, HIGH); relayState = true;
    } else {
       pcf.write(PCF_RELAY, LOW); relayState = false;
    }
  }
  if (currentPower != lastPower) { lastPower = currentPower; changed = true; }

  if (changed || millis() - lastStatusMillis > statusInterval) {
    publishStatus();
    if (changed) {
      logToSD("EVENT_CHANGE");
      captureImageToSD();
      publishImage();
    }
    lastStatusMillis = millis();
  }

  // Live Firebase Mode: If isLiveRequested is true, publish frames every 1 second
  static unsigned long lastLiveFirebase = 0;
  if (!otaInProgress && isLiveRequested && Firebase.ready() && millis() - lastLiveFirebase > 1000) {
     publishImage();
     lastLiveFirebase = millis();
  }

  delay(10);
}
