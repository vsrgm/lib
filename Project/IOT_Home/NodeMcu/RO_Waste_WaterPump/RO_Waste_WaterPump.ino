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

// Hardware Mapping for NodeMCU
#define RELAY_PIN D1      // GPIO 5
#define FLOAT_SWITCH_PIN D6 // GPIO 12
#define RELAY_ACTIVE_LOW true

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
String mqttClientId = "RO_Pump_" + String(ESP.getChipId(), HEX);
String baseTopic = "smart_home/";
String statusTopic = baseTopic + mqttClientId + "/status";
String cmdTopic = baseTopic + mqttClientId + "/commands";
String globalCmdTopic = baseTopic + "all/commands";
String discoveryTopic = baseTopic + "nodes/discovery";
String historyTopic = baseTopic + mqttClientId + "/history";

const String SW_VERSION = "1.0.304";

int currentMqttPortIndex = 0;
const int mqttPorts[] = {1883, 8000, 8883, 8884};
const int numMqttPorts = 4;

// State Variables
bool pumpActive = false;
bool waterLevelOk = false;
bool manualOverride = false;
unsigned long pumpStartTime = 0;
const unsigned long MAX_RUN_TIME = 600000; // 10 minutes safety timeout

// Web Logging
String webLogs = "";
void addWebLog(String msg) {
  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char tBuf[12];
  strftime(tBuf, sizeof(tBuf), "%H:%M:%S", &timeinfo);

  String logLine = "[" + String(tBuf) + "] " + msg;
  Serial.println(logLine);
  webLogs = logLine + "<br>" + webLogs;
  if (webLogs.length() > 800) webLogs = webLogs.substring(0, 800);
}

const char* logFile = "/ro_pump_log.csv";
const char* settingsFile = "/settings.json";

void loadSettings() {
  if (LittleFS.exists(settingsFile)) {
    File f = LittleFS.open(settingsFile, "r");
    if (f) {
      StaticJsonDocument<256> doc;
      deserializeJson(doc, f);
      if (doc.containsKey("mqtt_broker")) mqttBroker = doc["mqtt_broker"].as<String>();
      if (doc.containsKey("manual_override")) manualOverride = doc["manual_override"].as<bool>();
      f.close();
    }
  }
}

void saveSettings() {
  File f = LittleFS.open(settingsFile, "w");
  if (f) {
    StaticJsonDocument<256> doc;
    doc["mqtt_broker"] = mqttBroker;
    doc["manual_override"] = manualOverride;
    serializeJson(doc, f);
    f.close();
  }
}

void setPump(bool on) {
    if (on && !waterLevelOk) {
        addWebLog("Refused: Water level LOW");
        return;
    }
    pumpActive = on;
    if (pumpActive) pumpStartTime = millis();
    else pumpStartTime = 0;

    digitalWrite(RELAY_PIN, (RELAY_ACTIVE_LOW ? !on : on));
    addWebLog(pumpActive ? "Pump: ON" : "Pump: OFF");
}

void publishStatus() {
  StaticJsonDocument<512> doc;
  doc["pump"] = pumpActive ? "ON" : "OFF";
  doc["level"] = waterLevelOk ? "OK" : "LOW";
  doc["manual"] = manualOverride ? "ON" : "OFF";
  doc["heap"] = ESP.getFreeHeap();
  doc["ver"] = SW_VERSION;
  doc["ip"] = WiFi.localIP().toString();
  doc["id"] = mqttClientId;

  char buffer[400];
  serializeJson(doc, buffer);

  if (mqttClient.connected()) {
    mqttClient.publish(statusTopic.c_str(), buffer);
  }

  if (Firebase.ready()) {
    FirebaseJson json;
    json.setJsonData(buffer);
    Firebase.RTDB.setJSON(&fbdo, String(FB_OUTBOX) + "/status", &json);
  }
}

void logData(String eventType) {
  bool exists = LittleFS.exists(logFile);
  File f = LittleFS.open(logFile, "a");
  if (!f) return;

  if (!exists) {
    f.println("Date,Time,Event,Pump,Level,Manual,FreeHeap");
  }

  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char dBuf[12], tBuf[10];
  strftime(dBuf, sizeof(dBuf), "%Y-%m-%d", &timeinfo);
  strftime(tBuf, sizeof(tBuf), "%H:%M:%S", &timeinfo);

  String entry = String(dBuf) + "," + String(tBuf) + "," +
                 eventType + "," +
                 (pumpActive ? "ON" : "OFF") + "," +
                 (waterLevelOk ? "OK" : "LOW") + "," +
                 (manualOverride ? "ON" : "OFF") + "," +
                 String(ESP.getFreeHeap()) + "\n";

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
    delay(1000);
    ESP.restart();
  } else {
    addWebLog("FS Error: Could not save OTA");
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

  Serial.println("Starting OTA: " + url);
  addWebLog(isFull ? "Stage 2 OTA: " : "Stage 1 OTA: ");
  addWebLog(url);

  delay(5000);
  system_update_cpu_freq(160);

  WiFiClientSecure sClient;
  sClient.setInsecure();
  sClient.setBufferSizes(16384, 1024);

  ESPhttpUpdate.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  ESPhttpUpdate.rebootOnUpdate(true);

  t_httpUpdate_return ret = ESPhttpUpdate.update(sClient, url);

  system_update_cpu_freq(80);

  String err = "OTA Fail: " + ESPhttpUpdate.getLastErrorString();
  addWebLog(err);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String msg = "";
  for (int i = 0; i < length; i++) msg += (char)payload[i];

  if (msg == "SYNC") {
    publishStatus();
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
    StaticJsonDocument<128> doc;
    deserializeJson(doc, jsonStr);

    if (doc.containsKey("pump")) {
        setPump(doc["pump"].as<bool>());
    }
    if (doc.containsKey("manual")) {
        manualOverride = doc["manual"].as<bool>();
    }

    logData("App Config Update");
    saveSettings();
    publishStatus();
  } else if (msg == "HISTORY") {
    if (LittleFS.exists(logFile)) {
      File f = LittleFS.open(logFile, "r");
      while (f.available()) {
        String line = f.readStringUntil('\n');
        if (line.length() > 0) {
          mqttClient.publish(historyTopic.c_str(), line.c_str());
        }
      }
      f.close();
      mqttClient.publish(historyTopic.c_str(), "EOF");
    }
  }
}

void reconnectWiFi() {
  static unsigned long lastWiFiRetry = 0;
  if (millis() - lastWiFiRetry > 10000) {
    lastWiFiRetry = millis();
    WiFi.begin(ssid, password);
    Serial.println("Retrying WiFi connection...");
  }
}

void reconnectMqtt() {
  static unsigned long lastReconnectAttempt = 0;
  if (millis() - lastReconnectAttempt > 5000) {
    lastReconnectAttempt = millis();

    mqttPort = mqttPorts[currentMqttPortIndex];
    currentMqttPortIndex = (currentMqttPortIndex + 1) % numMqttPorts;

    if (mqttPort == 1883) {
      mqttClient.setClient(wifiClient);
    } else {
      secureClient.setInsecure();
      mqttClient.setClient(secureClient);
    }
    mqttClient.setServer(mqttBroker.c_str(), mqttPort);

    if (mqttClient.connect(mqttClientId.c_str())) {
      mqttClient.subscribe(cmdTopic.c_str());
      mqttClient.subscribe(globalCmdTopic.c_str());
      addWebLog("MQTT Connected: Port " + String(mqttPort));
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

        if ((msg.startsWith("http://") || msg.startsWith("https://")) &&
            !msg.startsWith("OTA:") && !msg.startsWith("CONFIG:")) {
            msg = "OTA:" + msg;
        }

        if (msg.startsWith("{") && !msg.startsWith("CONFIG:")) {
            msg = "CONFIG:" + msg;
        }

        Firebase.RTDB.setString(&fbdo, commandPath, "IDLE");
        mqttCallback((char*)cmdTopic.c_str(), (byte*)msg.c_str(), msg.length());
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
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  setPump(false);

  // Float switch input
  pinMode(FLOAT_SWITCH_PIN, INPUT_PULLUP);

  if (!LittleFS.begin()) Serial.println("LittleFS Failed");
  loadSettings();

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 40) {
    delay(500);
    timeout++;
  }

  configTime(5.5 * 3600, 0, "pool.ntp.org", "time.nist.gov");

  // Wait for time sync for Firebase
  time_t now = time(nullptr);
  int retry = 0;
  while (now < 8 * 3600 * 2 && retry < 40) {
    delay(500);
    now = time(nullptr);
    retry++;
  }

  if (now >= 8 * 3600 * 2) {
    runPendingOTA();
  }

  setupFirebase();
  fbdo.setBSSLBufferSize(2048, 512);
  fbdo.setResponseSize(1024);

  // Version tracking
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

  server.on("/", []() {
    server.sendHeader("Cache-Control", "no-cache");
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "text/html", "");

    server.sendContent("<html><head><meta http-equiv='refresh' content='5'><style>");
    server.sendContent("body{font-family:sans-serif;text-align:center;background:#f4f7f6;padding:10px;}");
    server.sendContent(".c{background:white;padding:15px;border-radius:10px;display:inline-block;text-align:left;min-width:300px;box-shadow:0 4px 6px rgba(0,0,0,0.1);}");
    server.sendContent("p{margin:8px 0;display:flex;justify-content:space-between;}");
    server.sendContent(".s{font-weight:bold;color:#2c3e50;}");
    server.sendContent(".log{background:#2c3e50;color:#2ecc71;padding:10px;border-radius:5px;font-family:monospace;font-size:0.8em;margin-top:10px;max-height:150px;overflow:auto;}");
    server.sendContent("</style></head><body><div class='c'><h3>RO Pump Monitor</h3>");

    server.sendContent("<p>Pump: <span class='s'>" + String(pumpActive ? "ON" : "OFF") + "</span></p>");
    server.sendContent("<p>Level: <span class='s'>" + String(waterLevelOk ? "OK" : "LOW") + "</span></p>");
    server.sendContent("<p>Manual: <span class='s'>" + String(manualOverride ? "ON" : "OFF") + "</span></p>");
    server.sendContent("<p>Heap: <span class='s'>" + String(ESP.getFreeHeap()) + "</span></p>");
    server.sendContent("<p>Ver: <span class='s'>" + SW_VERSION + "</span></p>");
    server.sendContent("<div class='log'>" + webLogs + "</div>");
    server.sendContent("<hr><p style='text-align:center;display:block;'><a href='/update'>Firmware Update</a></p>");
    server.sendContent("</div></body></html>");
    server.sendContent("");
  });

  server.on("/status", []() {
    StaticJsonDocument<256> doc;
    doc["pump"] = pumpActive ? "ON" : "OFF";
    doc["level"] = waterLevelOk ? "OK" : "LOW";
    doc["manual"] = manualOverride ? "ON" : "OFF";
    doc["heap"] = ESP.getFreeHeap();
    doc["ver"] = SW_VERSION;
    String response;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
  });

  server.on("/sync", []() {
    if (!LittleFS.exists(logFile)) {
      server.send(200, "text/plain", "No history\n");
      return;
    }
    File f = LittleFS.open(logFile, "r");
    server.streamFile(f, "text/csv");
    f.close();
  });

  server.on("/update", HTTP_GET, []() {
    mqttClient.disconnect();
    fbdo.clear();
    server.send(200, "text/html", "Update Mode. <a href='/update_now'>Upload</a>");
  });

  httpUpdater.setup(&server, "/update_now");

  if (WiFi.status() == WL_CONNECTED) {
    MDNS.begin(mqttClientId.c_str());
  }

  server.begin();
  MDNS.addService("http", "tcp", 80);

  mqttClient.setServer(mqttBroker.c_str(), mqttPort);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setBufferSize(512);

  publishStatus();
}

void loop() {
  server.handleClient();
  MDNS.update();

  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttClient.connected()) {
      reconnectMqtt();
    } else {
      mqttClient.loop();
    }
    checkCloudCommands();
  } else {
    reconnectWiFi();
  }

  static unsigned long lastSensorRead = 0;
  if (millis() - lastSensorRead > 1000) {
    lastSensorRead = millis();

    // Read float switch (Active LOW)
    bool currentLevel = (digitalRead(FLOAT_SWITCH_PIN) == LOW);

    if (currentLevel != waterLevelOk) {
        waterLevelOk = currentLevel;
        addWebLog(waterLevelOk ? "Level: OK" : "Level: LOW");

        // Auto-stop if level goes LOW even in manual override
        if (!waterLevelOk && pumpActive) {
            setPump(false);
            addWebLog("Auto-Stop: Low Water");
            logData("Auto-Stop");
        }
        publishStatus();
    }
  }

  // Safety Timeout
  if (pumpActive && (millis() - pumpStartTime > MAX_RUN_TIME)) {
      setPump(false);
      addWebLog("Safety Timeout!");
      logData("Safety Stop");
      publishStatus();
  }

  static unsigned long lastPeriodicPublish = 0;
  if (millis() - lastPeriodicPublish > 30000) {
    lastPeriodicPublish = millis();
    publishStatus();
  }
}
