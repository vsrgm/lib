#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoJson.h>
#include <LittleFS.h>
#include <time.h>
#include <ArduinoOTA.h>
#include <ESP8266HTTPUpdateServer.h>
#include <ESP8266mDNS.h>
#include <PubSubClient.h>
#include <WiFiClientSecure.h>
#include <ESP8266httpUpdate.h>
#include <Firebase_ESP_Client.h>
#include "credentials.h"

// Hardware Mapping
#define MQ2_ANALOG_PIN A0
#define MQ2_DIGITAL_PIN 16  // D0
#define RELAY_PIN 5         // D1
#define LDR_PIN 4           // D2
#define SWITCH_PIN 0        // D3
#define BUZZER_PIN 2        // D4

int smokeThresholdOn = 650;
int smokeThresholdOff = 550;
int buzzerThreshold = 800;
int buzzerFreq = 2000;

// Network credentials from credentials.h
const char* ssid = HOME_NETWORK_SSID;
const char* password = HOME_NETWORK_PASSWORD;

WiFiClient wifiClient;
WiFiClientSecure secureClient;
PubSubClient mqttClient;
ESP8266WebServer server(80);
ESP8266HTTPUpdateServer httpUpdater;

// Firebase Data objects
FirebaseData fbdo;
FirebaseAuth fbAuth;
FirebaseConfig fbConfig;

// MQTT Settings
String mqttBroker = MQTT_BROKER;
int mqttPort = MQTT_PORT;
String mqttClientId = "Kitchen-Fan-" + String(ESP.getChipId(), HEX);
String baseTopic = "smart_home/";
String statusTopic = baseTopic + mqttClientId + "/status";
String cmdTopic = baseTopic + mqttClientId + "/commands";
String globalCmdTopic = baseTopic + "all/commands";
String discoveryTopic = baseTopic + "nodes/discovery";
String historyTopic = baseTopic + mqttClientId + "/history";

const String SW_VERSION = "1.0.376";

int currentMqttPortIndex = 0;
const int mqttPorts[] = { 1883, 8000, 8883, 8884 };
const int numMqttPorts = 4;

// State Variables
bool fanActive = false;
bool buzzerActive = false;
bool manualOverride = false;
bool updateMode = false;
int mq2AnalogValue = 0;
int old_mq2AnalogValue = 0;
bool mq2DigitalState = false;
bool old_mq2DigitalState = false;

bool ldrValue = 0;
bool old_ldrValue = 0;
bool smokeDetected = false;

// Web Logging
String webLogs = "";
void addWebLog(String msg) {
  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char tBuf[12];
  strftime(tBuf, sizeof(tBuf), "%H:%M:%S", &timeinfo);

  String logLine = "[" + String(tBuf) + "] " + msg;
  webLogs = logLine + "<br>" + webLogs;
  if (webLogs.length() > 600) webLogs = webLogs.substring(0, 600);
}

const char* logFile = "/fan_log.csv";
const char* settingsFile = "/settings.json";

void loadSettings() {
  Serial.println("Loading Settings...");
  if (LittleFS.exists(settingsFile)) {
    File f = LittleFS.open(settingsFile, "r");
    if (f) {
      StaticJsonDocument<512> doc;
      DeserializationError error = deserializeJson(doc, f);
      if (!error) {
        if (doc.containsKey("mqtt_broker")) mqttBroker = doc["mqtt_broker"].as<String>();
        if (doc.containsKey("manual")) manualOverride = doc["manual"].as<bool>();
        if (doc.containsKey("smoke_th_on")) smokeThresholdOn = doc["smoke_th_on"].as<int>();
        if (doc.containsKey("smoke_th_off")) smokeThresholdOff = doc["smoke_th_off"].as<int>();
        if (doc.containsKey("buzzer_th")) buzzerThreshold = doc["buzzer_th"].as<int>();
        if (doc.containsKey("buzzer_freq")) buzzerFreq = doc["buzzer_freq"].as<int>();
        Serial.println("Settings Loaded.");
      } else {
        Serial.println("Failed to parse settings JSON");
      }
      f.close();
    }
  } else {
    Serial.println("No settings file found.");
  }
}

void saveSettings() {
  File f = LittleFS.open(settingsFile, "w");
  if (f) {
    StaticJsonDocument<512> doc;
    doc["mqtt_broker"] = mqttBroker;
    doc["manual"] = manualOverride;
    doc["smoke_th_on"] = smokeThresholdOn;
    doc["smoke_th_off"] = smokeThresholdOff;
    doc["buzzer_th"] = buzzerThreshold;
    doc["buzzer_freq"] = buzzerFreq;
    serializeJson(doc, f);
    f.close();
  }
}

void setFan(bool on) {
  fanActive = on;
  digitalWrite(RELAY_PIN, on ? LOW : HIGH);
  addWebLog(fanActive ? "Fan: ON" : "Fan: OFF");
}

void setBuzzer(bool on) {
  buzzerActive = on;
  if (on) {
    tone(BUZZER_PIN, buzzerFreq);
  } else {
    noTone(BUZZER_PIN);
    digitalWrite(BUZZER_PIN, HIGH);
  }
}

void publishStatus() {
  DynamicJsonDocument doc(1024);
  doc["fan"] = fanActive ? "ON" : "OFF";
  doc["buzzer"] = buzzerActive ? "ON" : "OFF";
  doc["manual"] = manualOverride ? "ON" : "OFF";
  doc["user_sw"] = (digitalRead(SWITCH_PIN) == LOW ? "State 1" : "State 2");
  doc["mq2_a"] = mq2AnalogValue;
  doc["mq2_d"] = mq2DigitalState ? "SMOKE" : "CLEAR";
  doc["ldr"] = ldrValue ? "DARK" : "LIGHT";
  doc["heap"] = ESP.getFreeHeap();
  doc["ver"] = SW_VERSION;
  doc["ip"] = WiFi.localIP().toString();
  doc["id"] = mqttClientId;
  doc["smoke_th_on"] = smokeThresholdOn;
  doc["smoke_th_off"] = smokeThresholdOff;
  doc["buzzer_th"] = buzzerThreshold;
  doc["buzzer_freq"] = buzzerFreq;

  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char tBuf[20];
  strftime(tBuf, sizeof(tBuf), "%Y-%m-%d %H:%M:%S", &timeinfo);
  doc["ts"] = tBuf;

  String buffer;
  serializeJson(doc, buffer);

  if (mqttClient.connected()) {
    mqttClient.publish(statusTopic.c_str(), buffer.c_str());
  }

  if (Firebase.ready()) {
    FirebaseJson json;
    json.setJsonData(buffer.c_str());
    if (!Firebase.RTDB.setJSON(&fbdo, String(FB_OUTBOX) + "/status", &json)) {
      addWebLog("FB Err: " + fbdo.errorReason());
    }
  }
}

void logData(String eventType) {
  bool exists = LittleFS.exists(logFile);
  File f = LittleFS.open(logFile, "a");
  if (!f) return;

  if (!exists) {
    f.println("Date,Time,Event,Fan,Manual,MQ2_A,LDR");
  }

  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char dBuf[12], tBuf[10];
  strftime(dBuf, sizeof(dBuf), "%Y-%m-%d", &timeinfo);
  strftime(tBuf, sizeof(tBuf), "%H:%M:%S", &timeinfo);

  String entry = String(dBuf) + "," + String(tBuf) + "," + eventType + "," + (fanActive ? "ON" : "OFF") + "," + (manualOverride ? "ON" : "OFF") + "," + String(mq2AnalogValue) + "," + (ldrValue ? "DARK" : "LIGHT") + "\n";

  f.print(entry);
  f.close();

  if (Firebase.ready()) {
    String trimmedEntry = entry;
    trimmedEntry.trim();
    Firebase.RTDB.pushString(&fbdo, String(FB_OUTBOX) + "/history", trimmedEntry);
  }
}

void handleRemoteOTA(String url) {
  url.trim();
  addWebLog("OTA Pending. Restarting...");

  if (Firebase.ready()) {
    Firebase.RTDB.setString(&fbdo, String(FB_OUTBOX) + "/ota_status", "COMMAND_RECEIVED");
  }

  if (url.indexOf("www.dropbox.com") != -1) {
    url.replace("www.dropbox.com", "dl.dropboxusercontent.com");
  }

  if (url.indexOf("github.com") != -1 && url.indexOf("raw.githubusercontent.com") == -1) {
    url.replace("github.com", "raw.githubusercontent.com");
    url.replace("/blob/", "/");
  }

  url.replace("?dl=1", "");
  url.replace("&dl=1", "");
  url.replace("?dl=0", "");
  url.replace("&dl=0", "");

  File f = LittleFS.open("/ota.txt", "w");
  if (f) {
    f.print(url);
    f.close();
    addWebLog("OTA Saved. Restarting...");
    delay(500);
    WiFi.disconnect(true);
    delay(500);
    ESP.restart();
  }
}

void runPendingOTA() {
  String url = "";
  bool isFull = false;

  if (LittleFS.exists("/ota.txt")) {
    File f = LittleFS.open("/ota.txt", "r");
    url = f.readString();
    f.close();
    LittleFS.remove("/ota.txt");
  } else if (LittleFS.exists("/full_ota.txt")) {
    File f = LittleFS.open("/full_ota.txt", "r");
    url = f.readString();
    f.close();
    LittleFS.remove("/full_ota.txt");
    isFull = true;
  }

  url.trim();
  if (url.length() < 10) return;

  addWebLog(isFull ? "Stage 2 OTA: " : "Stage 1 OTA: ");
  addWebLog(url);
  delay(2000);

  ESP.wdtEnable(20000);

  WiFiClientSecure sClient;
  sClient.setInsecure();
  sClient.setBufferSizes(16384, 1024);

  ESPhttpUpdate.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  t_httpUpdate_return ret = ESPhttpUpdate.update(sClient, url);

  if (ret == HTTP_UPDATE_FAILED) {
    addWebLog("OTA Error: " + ESPhttpUpdate.getLastErrorString());
  }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String msg = "";
  for (int i = 0; i < length; i++) msg += (char)payload[i];

  if (msg == "SYNC") {
    publishStatus();
  } else if (msg == "REBOOT") {
    addWebLog("Rebooting...");
    delay(500);
    ESP.restart();
  } else if (msg.startsWith("OTA:")) {
    handleRemoteOTA(msg.substring(4));
  } else if (msg.startsWith("OTA_FULL:")) {
    String url = msg.substring(9);
    File f = LittleFS.open("/full_ota.txt", "w");
    if (f) {
      f.print(url);
      f.close();
      addWebLog("Full OTA URL Saved");
    }
  } else if (msg == "DISCOVER") {
    String discoveryMsg = "{\"ip\":\"" + WiFi.localIP().toString() + "\", \"id\":\"" + mqttClientId + "\", \"ver\":\"" + SW_VERSION + "\"}";
    mqttClient.publish(discoveryTopic.c_str(), discoveryMsg.c_str());
  } else if (msg == "CLEAR" || msg == "clearHistory") {
    LittleFS.remove(logFile);
    addWebLog("History Cleared");
    if (Firebase.ready()) {
      Firebase.RTDB.deleteNode(&fbdo, String(FB_OUTBOX) + "/history");
    }
  } else if (msg.startsWith("CONFIG:")) {
    String jsonStr = msg.substring(7);
    StaticJsonDocument<512> doc;
    deserializeJson(doc, jsonStr);

    if (doc.containsKey("fan")) setFan(doc["fan"].as<bool>());
    if (doc.containsKey("buzzer")) setBuzzer(doc["buzzer"].as<bool>());
    if (doc.containsKey("manual")) manualOverride = doc["manual"].as<bool>();
    if (doc.containsKey("smoke_th_on")) smokeThresholdOn = doc["smoke_th_on"].as<int>();
    if (doc.containsKey("smoke_th_off")) smokeThresholdOff = doc["smoke_th_off"].as<int>();
    if (doc.containsKey("buzzer_th")) buzzerThreshold = doc["buzzer_th"].as<int>();
    if (doc.containsKey("buzzer_freq")) buzzerFreq = doc["buzzer_freq"].as<int>();

    logData("App Config Update");
    saveSettings();
    publishStatus();
  }
}

void reconnectWiFi() {
  static unsigned long lastWiFiRetry = 0;
  if (millis() - lastWiFiRetry > 10000) {
    lastWiFiRetry = millis();
    WiFi.begin(ssid, password);
  }
}

void reconnectMqtt() {
  static unsigned long lastReconnectAttempt = 0;
  if (millis() - lastReconnectAttempt > 5000) {
    lastReconnectAttempt = millis();
    mqttPort = mqttPorts[currentMqttPortIndex];
    currentMqttPortIndex = (currentMqttPortIndex + 1) % numMqttPorts;

    if (mqttPort == 1883) mqttClient.setClient(wifiClient);
    else {
      secureClient.setInsecure();
      mqttClient.setClient(secureClient);
    }
    mqttClient.setServer(mqttBroker.c_str(), mqttPort);

    if (mqttClient.connect(mqttClientId.c_str())) {
      mqttClient.subscribe(cmdTopic.c_str());
      mqttClient.subscribe(globalCmdTopic.c_str());
      String discoveryMsg = "{\"ip\":\"" + WiFi.localIP().toString() + "\", \"id\":\"" + mqttClientId + "\", \"ver\":\"" + SW_VERSION + "\"}";
      mqttClient.publish(discoveryTopic.c_str(), discoveryMsg.c_str());
    }
  }
}

void checkCloudCommands() {
  if (Firebase.ready()) {
    String commandPath = String(FB_INBOX) + "/command";
    fbdo.clear();

    if (Firebase.RTDB.getString(&fbdo, commandPath)) {
      String msg = fbdo.stringData();
      msg.replace("\\\"", "\"");
      if (msg.startsWith("\"")) msg = msg.substring(1);
      if (msg.endsWith("\"")) msg = msg.substring(0, msg.length() - 1);

      if (msg.length() > 0 && msg != "null" && msg != "IDLE") {
        addWebLog("Cmd Recv: " + msg);

        if ((msg.startsWith("http://") || msg.startsWith("https://")) && !msg.startsWith("OTA:") && !msg.startsWith("CONFIG:")) {
          msg = "OTA:" + msg;
        }

        if (msg.startsWith("{") && !msg.startsWith("CONFIG:")) {
          msg = "CONFIG:" + msg;
        }

        Firebase.RTDB.setString(&fbdo, commandPath, "IDLE");
        mqttCallback((char*)cmdTopic.c_str(), (byte*)msg.c_str(), msg.length());
        publishStatus();
      }
    }
  }
}

void setupFirebase() {
  fbConfig.database_url = "https://" + String(FIREBASE_HOST) + "/";
  fbConfig.api_key = FIREBASE_API_KEY;
  fbAuth.user.email = FIREBASE_USER_EMAIL;
  fbAuth.user.password = FIREBASE_USER_PASSWORD;
  fbConfig.signer.test_mode = false;
  Firebase.begin(&fbConfig, &fbAuth);
  Firebase.reconnectWiFi(true);
  addWebLog("Firebase Ready");
}

void setup() {
  delay(200);
  Serial.begin(115200);
  Serial.println("\n\n--- Kitchen Exhaust Fan Starting ---");
  Serial.printf("Version: %s\n", SW_VERSION.c_str());

  digitalWrite(RELAY_PIN, HIGH);
  digitalWrite(BUZZER_PIN, HIGH);

  pinMode(RELAY_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(MQ2_DIGITAL_PIN, INPUT);
  pinMode(LDR_PIN, INPUT);
  pinMode(SWITCH_PIN, INPUT_PULLUP);

  // Check for Factory Reset (Hold switch for 5 seconds during boot)
  if (digitalRead(SWITCH_PIN) == LOW) {
    Serial.println("Reset Button Detected. Hold for 5s to Factory Reset...");
    int count = 0;
    while (digitalRead(SWITCH_PIN) == LOW && count < 50) {
      delay(100);
      count++;
      if (count % 10 == 0) Serial.print(".");
    }
    if (count >= 50) {
      Serial.println("\nFACTORY RESET INITIATED");
      if (LittleFS.begin()) {
        LittleFS.format();
        Serial.println("LittleFS Formatted.");
      }
    }
  }

  if (!LittleFS.begin()) {
    Serial.println("LittleFS Failed");
  } else {
    Serial.println("LittleFS Mounted.");
  }

  loadSettings();

  Serial.printf("Connecting to WiFi: %s\n", ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 40) {
    delay(500);
    timeout++;
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi Connected. IP: " + WiFi.localIP().toString());

    Serial.println("Syncing Time...");
    configTime(5.5 * 3600, 0, "pool.ntp.org", "time.nist.gov");
    time_t now = time(nullptr);
    int retry = 0;
    while (now < 8 * 3600 * 2 && retry < 40) {
      delay(500);
      now = time(nullptr);
      retry++;
      Serial.print(".");
    }
    Serial.println("\nTime Synced.");
  } else {
    Serial.println("\nWiFi Failed. Continuing in Offline Mode.");
  }

  runPendingOTA();

  setupFirebase();
  fbdo.setBSSLBufferSize(2048, 512);
  fbdo.setResponseSize(1024);

  // Version Tracking
  const char* verFile = "/last_ver.txt";
  String lastVer = "";
  if (LittleFS.exists(verFile)) {
    File f = LittleFS.open(verFile, "r");
    lastVer = f.readString();
    f.close();
  }
  if (lastVer != SW_VERSION) {
    if (Firebase.ready()) {
      Firebase.RTDB.setString(&fbdo, String(FB_OUTBOX) + "/ota_status", "SUCCESS: " + SW_VERSION);
    }
    File f = LittleFS.open(verFile, "w");
    f.print(SW_VERSION);
    f.close();
  }

  Serial.println("Setting up Web Server...");

  server.on("/", []() {
    updateMode = false;
    time_t now = time(nullptr);
    struct tm timeinfo;
    localtime_r(&now, &timeinfo);
    char tBuf[20];
    strftime(tBuf, sizeof(tBuf), "%Y-%m-%d %H:%M:%S", &timeinfo);

    server.sendHeader(F("Cache-Control"), F("no-cache"));
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, F("text/html"), "");

    server.sendContent(F("<html><head><meta http-equiv='refresh' content='5'><style>"));
    server.sendContent(F("body{font-family:sans-serif;text-align:center;background:#f4f7f6;padding:10px;}"));
    server.sendContent(F(".c{background:white;padding:15px;border-radius:10px;display:inline-block;text-align:left;min-width:300px;box-shadow:0 4px 6px rgba(0,0,0,0.1);}"));
    server.sendContent(F("p{margin:8px 0;display:flex;justify-content:space-between;}"));
    server.sendContent(F(".s{font-weight:bold;color:#2c3e50;}"));
    server.sendContent(F(".log{background:#2c3e50;color:#2ecc71;padding:10px;border-radius:5px;font-family:monospace;font-size:0.8em;margin-top:10px;max-height:150px;overflow:auto;}"));
    server.sendContent(F("</style></head><body><div class='c'><h3>Kitchen Exhaust Fan</h3>"));

    server.sendContent(F("<p>Fan: <span class='s'>"));
    server.sendContent(fanActive ? F("ON") : F("OFF"));
    server.sendContent(F("</span></p>"));
    server.sendContent(F("<p>MQ2 A: <span class='s'>"));
    server.sendContent(String(mq2AnalogValue));
    server.sendContent(F("</span></p>"));
    server.sendContent(F("<p>LDR: <span class='s'>"));
    server.sendContent(ldrValue ? F("DARK") : F("LIGHT"));
    server.sendContent(F("</span></p>"));
    server.sendContent(F("<p>Manual: <span class='s'>"));
    server.sendContent(manualOverride ? F("ON") : F("OFF"));
    server.sendContent(F("</span></p>"));
    server.sendContent(F("<p>Heap: <span class='s'>"));
    server.sendContent(String(ESP.getFreeHeap()));
    server.sendContent(F("</span></p>"));
    server.sendContent(F("<p>Ver: <span class='s'>"));
    server.sendContent(SW_VERSION);
    server.sendContent(F("</span></p>"));
    server.sendContent(F("<p>Time: <span class='s'>"));
    server.sendContent(tBuf);
    server.sendContent(F("</span></p>"));
    server.sendContent(F("<div class='log'>"));
    server.sendContent(webLogs);
    server.sendContent(F("</div>"));
    server.sendContent(F("<hr><p style='text-align:center;display:block;'><a href='/update'>Firmware Update</a> | <a href='/reboot' onclick=\"return confirm('Reboot device?')\">Reboot</a></p>"));
    server.sendContent(F("</div></body></html>"));
    server.sendContent("");
  });

  server.on("/status", []() {
    StaticJsonDocument<512> doc;
    doc["fan"] = fanActive ? "ON" : "OFF";
    doc["manual"] = manualOverride ? "ON" : "OFF";
    doc["mq2_a"] = mq2AnalogValue;
    doc["ver"] = SW_VERSION;
    doc["ip"] = WiFi.localIP().toString();
    doc["id"] = mqttClientId;

    time_t now = time(nullptr);
    struct tm timeinfo;
    localtime_r(&now, &timeinfo);
    char tBuf[20];
    strftime(tBuf, sizeof(tBuf), "%Y-%m-%d %H:%M:%S", &timeinfo);
    doc["ts"] = tBuf;

    String response;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
  });

  server.on("/update", []() {
    updateMode = true;
    ESP.wdtEnable(20000);
    String html = "<html><head><meta name='viewport' content='width=device-width, initial-scale=1'>",targetContent:
                  "<style>body{font-family:sans-serif;text-align:center;padding:20px;background:#f4f7f6;}"
                  ".c{background:white;padding:30px;border-radius:10px;display:inline-block;box-shadow:0 4px 6px rgba(0,0,0,0.1);max-width:90%;}"
                  "h2{color:#d35400;} .btn{background:#27ae60;color:white;padding:12px 24px;text-decoration:none;border-radius:5px;display:inline-block;margin-top:20px;}"
                  "</style></head>"
                  "<body><div class='c'><h2>Firmware Update Mode</h2>"
                  "<p>Background services (Firebase/MQTT) have been stopped to free memory.</p>"
                  "<p>Free Heap: <b>" + String(ESP.getFreeHeap()) + "</b> bytes</p>"
                  "<a href='/update_now' class='btn'>Open Update Tool</a>"
                  "<p style='margin-top:20px;font-size:0.8em;color:gray;'>After update, device will restart automatically.</p>"
                  "</div></body></html>";
    server.send(200, "text/html", html);
  });

  server.on("/reboot", []() {
    server.send(200, "text/html", "<html><head><meta http-equiv='refresh' content='10;url=/'></head><body><h3>Rebooting...</h3><p>Redirecting to home in 10s...</p></body></html>");
    addWebLog("Web Reboot Init");
    delay(500);
    ESP.restart();
  });

  httpUpdater.setup(&server, "/update_now");

if (WiFi.status() == WL_CONNECTED) {
  MDNS.begin(mqttClientId.c_str());
}

server.begin();
MDNS.addService("http", "tcp", 80);

mqttClient.setServer(mqttBroker.c_str(), mqttPort);
mqttClient.setCallback(mqttCallback);
mqttClient.setBufferSize(1024);
ESP.wdtEnable(WDTO_8S);
publishStatus();
}

void loop() {
  ESP.wdtFeed();
  server.handleClient();
  MDNS.update();

  if (updateMode) {
    delay(10);
    return;
  }

  int statechanged = 0;

  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttClient.connected()) {
      reconnectMqtt();
    } else {
      mqttClient.loop();
    }

    if (Firebase.ready()) {
      static unsigned long lastCmdCheck = 0;
      if (millis() - lastCmdCheck > 2000) {
        lastCmdCheck = millis();
        checkCloudCommands();
      }
    }
  } else {
    reconnectWiFi();
  }

  static unsigned long lastAdc = 0;
  if (millis() - lastAdc > 500) {
    lastAdc = millis();
    mq2AnalogValue = analogRead(A0);
    if (abs(mq2AnalogValue - old_mq2AnalogValue) > 15) {
      old_mq2AnalogValue = mq2AnalogValue;
      statechanged = 1;
    }
  }

  mq2DigitalState = (digitalRead(MQ2_DIGITAL_PIN) == LOW);
  if (mq2DigitalState != old_mq2DigitalState) {
    old_mq2DigitalState = mq2DigitalState;
    statechanged = 1;
  }

  ldrValue = digitalRead(LDR_PIN);
  if (ldrValue != old_ldrValue) {
    old_ldrValue = ldrValue;
    statechanged = 1;
  }

  if (mq2DigitalState || mq2AnalogValue > smokeThresholdOn) {
    if (!smokeDetected) {
      smokeDetected = true;
      logData("SMOKE ALERT");
      statechanged = 1;
    }
    // Safety persistence: Ensure fan is ON if smoke is detected, even if it was manually toggled OFF
    if (!fanActive) {
      setFan(true);
      statechanged = 1;
    }
  } else if (smokeDetected && !mq2DigitalState && mq2AnalogValue < smokeThresholdOff) {
    smokeDetected = false;
    if (!manualOverride) {
      setFan(false);
    }
    logData("SMOKE CLEARED");
    statechanged = 1;
  }

  // Buzzer Control Logic
  if (mq2AnalogValue >= buzzerThreshold) {
    if (!buzzerActive) {
      setBuzzer(true);
      statechanged = 1;
    }
  } else if (buzzerActive && !manualOverride && mq2AnalogValue < buzzerThreshold) {
    setBuzzer(false);
    statechanged = 1;
  }

  static int lastSwitchState = -1;
  int currentSwitchState = digitalRead(SWITCH_PIN);
  if (lastSwitchState == -1) lastSwitchState = currentSwitchState;
  if (currentSwitchState != lastSwitchState) {
    delay(50);
    if (digitalRead(SWITCH_PIN) == currentSwitchState) {
      lastSwitchState = currentSwitchState;
      setFan(!fanActive);
      statechanged = 1;
    }
  }

  static unsigned long lastPeriodicPublish = 0;
  if (millis() - lastPeriodicPublish > 30000) {
    lastPeriodicPublish = millis();
    statechanged = 1;
  }
  static unsigned long lastUpdate = 0;
  if (statechanged == 1 && millis() - lastUpdate > 1000) {
    statechanged = 0;
    lastUpdate = millis();
    publishStatus();
  }
}
