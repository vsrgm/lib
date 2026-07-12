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
#include <DHT.h>
#include <Firebase_ESP_Client.h>
#include "credentials.h"

#define LDR_PIN A0
#define EMERGENCY_LIGHT_PIN D2
#define DHTPIN D3
#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);

// Network credentials
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
String mqttClientId = "StudyRoomNode_" + String(ESP.getChipId(), HEX);
String baseTopic = "smart_home/";
String statusTopic = baseTopic + mqttClientId + "/status";
String cmdTopic = baseTopic + mqttClientId + "/commands";
String globalCmdTopic = baseTopic + "all/commands";
String discoveryTopic = baseTopic + "nodes/discovery";
String historyTopic = baseTopic + mqttClientId + "/history";

const String SW_VERSION = "1.0.178";

int currentMqttPortIndex = 0;
const int mqttPorts[] = {1883, 8000, 8883, 8884};
const int numMqttPorts = 4;

// Sensor and State Variables
float dhtTemp = 0.0;
float dhtHum = 0.0;
int lightRawValue = 0;
bool emergencyLightOn = false;
bool manualOverride = false;

// Sensor Enable/Disable Flags
bool enDht = true;
bool enLight = true;
bool enEmer = true;

// Web Logging
String webLogs = "";
void addWebLog(String msg) {
  Serial.println("LOG: " + msg);
  webLogs = msg + "<br>" + webLogs;
  if (webLogs.length() > 800) webLogs = webLogs.substring(0, 800);
}

const char* logFile = "/study_room_log.csv";
const char* settingsFile = "/settings.json";

void loadSettings() {
  if (LittleFS.exists(settingsFile)) {
    File f = LittleFS.open(settingsFile, "r");
    if (f) {
      StaticJsonDocument<256> doc;
      deserializeJson(doc, f);
      if (doc.containsKey("mqtt_broker")) mqttBroker = doc["mqtt_broker"].as<String>();
      if (doc.containsKey("mqtt_port")) mqttPort = doc["mqtt_port"].as<int>();
      if (doc.containsKey("en_dht")) enDht = doc["en_dht"].as<bool>();
      if (doc.containsKey("en_light")) enLight = doc["en_light"].as<bool>();
      if (doc.containsKey("en_emer")) enEmer = doc["en_emer"].as<bool>();
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
    doc["mqtt_port"] = mqttPort;
    doc["en_dht"] = enDht;
    doc["en_light"] = enLight;
    doc["en_emer"] = enEmer;
    doc["manual_override"] = manualOverride;
    serializeJson(doc, f);
    f.close();
  }
}

void publishStatus() {
  StaticJsonDocument<512> doc; // Increased size to accommodate IP and ID
  doc["dht_temp"] = String(dhtTemp, 1);
  doc["dht_hum"] = String(dhtHum, 1);
  doc["light_raw"] = lightRawValue;
  doc["emer"] = emergencyLightOn ? "ON" : "OFF";
  doc["manual_override"] = manualOverride ? "ON" : "OFF";
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
    // Write to the OUTBOX
    Firebase.RTDB.setJSON(&fbdo, String(FB_OUTBOX) + "/status", &json);
  }
}

void logData(String eventType) {
  bool exists = LittleFS.exists(logFile);
  File f = LittleFS.open(logFile, "a");
  if (!f) return;

  if (!exists) {
    f.println("Date,Time,Reason,Manual Override,DHT_T,DHT_H,Light,Emer,FreeSpace");
  }

  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char dBuf[12], tBuf[10];
  strftime(dBuf, sizeof(dBuf), "%Y-%m-%d", &timeinfo);
  strftime(tBuf, sizeof(tBuf), "%H:%M:%S", &timeinfo);

  FSInfo fs_info;
  long freeSpace = 0;
  if (LittleFS.info(fs_info)) {
    freeSpace = fs_info.totalBytes - fs_info.usedBytes;
  }

  String entry = String(dBuf) + "," + String(tBuf) + "," +
                 eventType + "," +
                 (manualOverride ? "ON" : "OFF") + "," +
                 String(dhtTemp, 1) + "," +
                 String(dhtHum, 1) + "," +
                 String(lightRawValue) + "," +
                 (emergencyLightOn ? "ON" : "OFF") + "," +
                 String(freeSpace) + "\n";

  f.print(entry);
  f.close();

  // Parallel Push to Firebase History
  if (Firebase.ready()) {
    String trimmedEntry = entry;
    trimmedEntry.trim();
    // Write history to OUTBOX
    Firebase.RTDB.pushString(&fbdo, String(FB_OUTBOX) + "/history", trimmedEntry);
  }
}

void handleRemoteOTA(String url) {
  url.trim();
  addWebLog("OTA Pending. Restarting...");

  // Report to Firebase before restarting
  if (Firebase.ready()) {
    Firebase.RTDB.setString(&fbdo, String(FB_OUTBOX) + "/ota_status", "COMMAND_RECEIVED");
  }

  // Dropbox Direct Download Fix
  if (url.indexOf("www.dropbox.com") != -1) {
      url.replace("www.dropbox.com", "dl.dropboxusercontent.com");
  }

  // GitHub Direct Download Fix
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
  if (!LittleFS.exists("/ota.txt")) return;

  File f = LittleFS.open("/ota.txt", "r");
  String url = f.readString();
  f.close();
  LittleFS.remove("/ota.txt");
  url.trim();

  if (url.length() < 10) return;

  Serial.println("Starting Fresh OTA: " + url);
  addWebLog("Boot-OTA Attempt: " + url);

  // CRITICAL: Wait for network to be fully stable
  delay(5000);

  // Boost CPU speed to 160MHz for the heavy download
  system_update_cpu_freq(160);

  WiFiClientSecure sClient;
  sClient.setInsecure();
  // 16384 (16KB) is the absolute max SSL fragment size.
  sClient.setBufferSizes(16384, 1024);

  HTTPClient http;
  http.begin(sClient, url);
  http.setUserAgent("Mozilla/5.0 (ESP8266)");
  http.setTimeout(180000);

  ESPhttpUpdate.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  ESPhttpUpdate.rebootOnUpdate(true); // Standard mode: reboot immediately on success

  t_httpUpdate_return ret = ESPhttpUpdate.update(http);

  // If we reach here, update failed (otherwise it would have rebooted)
  http.end();
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
  } else if (msg == "DISCOVER") {
    String discoveryMsg = "{\"ip\":\"" + WiFi.localIP().toString() + "\", \"id\":\"" + mqttClientId + "\", \"ver\":\"" + SW_VERSION + "\"}";
    mqttClient.publish(discoveryTopic.c_str(), discoveryMsg.c_str());
  } else if (msg == "CLEAR") {
    LittleFS.remove(logFile);
  } else if (msg.startsWith("CONFIG:")) {
    String jsonStr = msg.substring(7);
    StaticJsonDocument<128> doc; // Reduced from 256
    deserializeJson(doc, jsonStr);

    if (doc.containsKey("manual_override")) {
        manualOverride = doc["manual_override"].as<bool>();
    }

    if (doc.containsKey("en_emer")) {
        emergencyLightOn = doc["en_emer"].as<bool>();
        digitalWrite(EMERGENCY_LIGHT_PIN, emergencyLightOn ? HIGH : LOW);
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
      String discoveryMsg = "{\"ip\":\"" + WiFi.localIP().toString() + "\", \"id\":\"" + mqttClientId + "\", \"ver\":\"" + SW_VERSION + "\"}";
      mqttClient.publish(discoveryTopic.c_str(), discoveryMsg.c_str());
    }
  }
}

void checkCloudCommands() {
  if (Firebase.ready()) {
    // Read from the INBOX
    String commandPath = String(FB_INBOX) + "/command";

    // Clear fbdo before request to ensure fresh data
    fbdo.clear();

    if (Firebase.RTDB.getString(&fbdo, commandPath)) {
      String msg = fbdo.stringData();

      // FIX 1: Remove double-escaping and quotes
      msg.replace("\\\"", "\"");
      if (msg.startsWith("\"")) msg = msg.substring(1);
      if (msg.endsWith("\"")) msg = msg.substring(0, msg.length() - 1);

      if (msg.length() > 0 && msg != "null" && msg != "IDLE") {
        addWebLog("Cmd Recv: " + msg);

        // Auto-detect OTA URLs if they don't have the prefix
        if ((msg.startsWith("http://") || msg.startsWith("https://")) &&
            !msg.startsWith("OTA:") && !msg.startsWith("CONFIG:")) {
            msg = "OTA:" + msg;
            addWebLog("Auto-prefixed OTA");
        }

        // Detect JSON and prefix with CONFIG: if missing
        if (msg.startsWith("{") && !msg.startsWith("CONFIG:")) {
            msg = "CONFIG:" + msg;
        }

        // Reset the INBOX to IDLE *before* executing, so user sees the "transaction"
        Firebase.RTDB.setString(&fbdo, commandPath, "IDLE");

        mqttCallback((char*)cmdTopic.c_str(), (byte*)msg.c_str(), msg.length());
      }
    } else {
      // Optional: log error if getString failed
      // Serial.println("FB Error: " + fbdo.errorReason());
    }
  }
}

void setupFirebase() {
  fbConfig.database_url = "https://" + String(FIREBASE_HOST) + "/";
  fbConfig.api_key = FIREBASE_API_KEY;
  fbAuth.user.email = FIREBASE_USER_EMAIL;
  fbAuth.user.password = FIREBASE_USER_PASSWORD;

  // OPTIMIZATION: Use larger buffers to handle long URLs and metadata
  fbConfig.signer.test_mode = false;
  // Buffers are now set per-context (Boot vs Main) to save RAM

  Firebase.begin(&fbConfig, &fbAuth);
  Firebase.reconnectWiFi(true);
  addWebLog("Firebase Ready");
}

void setup() {
  Serial.begin(115200);
  pinMode(EMERGENCY_LIGHT_PIN, OUTPUT);
  digitalWrite(EMERGENCY_LIGHT_PIN, LOW);

  if (!LittleFS.begin()) Serial.println("LittleFS Failed");
  loadSettings();

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 40) {
    delay(500);
    Serial.print(".");
    timeout++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi Connected. IP: " + WiFi.localIP().toString());
  }

  // CRUCIAL: Wait for NTP Time Sync (Required for Firebase SSL)
  configTime(5.5 * 3600, 0, "pool.ntp.org", "time.nist.gov");
  Serial.print("Waiting for NTP time sync: ");
  time_t now = time(nullptr);
  int retry = 0;
  while (now < 8 * 3600 * 2 && retry < 40) {
    delay(500);
    Serial.print(".");
    now = time(nullptr);
    retry++;
  }
  if (now < 8 * 3600 * 2) addWebLog("Time sync FAILED!");
  else {
    addWebLog("Time Synchronized.");
    runPendingOTA(); // Execute any update saved before restart
  }

  setupFirebase();
  // Reduced buffers for normal operation to keep HTTP server stable
  fbdo.setBSSLBufferSize(2048, 512);
  fbdo.setResponseSize(1024);

  // VERSION TRACKER: Detect if we just booted into a new version
  const char* verFile = "/last_ver.txt";
  String lastVer = "";
  if (LittleFS.exists(verFile)) {
    File f = LittleFS.open(verFile, "r");
    lastVer = f.readString();
    f.close();
  }

  if (lastVer != SW_VERSION) {
    // New version detected! Report success
    if (Firebase.ready()) {
      Firebase.RTDB.setString(&fbdo, String(FB_OUTBOX) + "/ota_status", "SUCCESS: " + SW_VERSION);
      addWebLog("New Firmware Detected: " + SW_VERSION);
    }
    // Save current version for next boot
    File f = LittleFS.open(verFile, "w");
    f.print(SW_VERSION);
    f.close();
  }

  server.on("/", []() {
    server.sendHeader("Cache-Control", "no-cache");
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "text/html", "");

    server.sendContent("<html><head><meta http-equiv='refresh' content='5'><style>");
    server.sendContent("body{font-family:sans-serif;text-align:center;background:#eef2f7;padding:10px;}");
    server.sendContent(".c{background:white;padding:15px;border-radius:10px;display:inline-block;text-align:left;min-width:300px;box-shadow:0 5px 15px rgba(0,0,0,0.1);}");
    server.sendContent("p{margin:8px 0;display:flex;justify-content:space-between; font-size: 0.9em;}");
    server.sendContent(".s{font-weight:bold;padding:2px 8px;border-radius:10px;font-size:0.8em;background:#2ecc71;color:white;}");
    server.sendContent(".log{background:#2c3e50;color:#0f0;padding:10px;border-radius:5px;font-family:monospace;font-size:0.75em;margin-top:10px;max-height:100px;overflow:auto;word-wrap:break-word;}");
    server.sendContent("</style></head><body><div class='c'><h3>Study Monitor</h3>");

    server.sendContent("<p>Ver: <span>" + SW_VERSION + "</span></p>");
    server.sendContent("<p>Heap: <span>" + String(ESP.getFreeHeap()) + "</span></p>");
    server.sendContent("<hr>");
    server.sendContent("<p>Temp: <span>" + String(dhtTemp, 1) + " C</span></p>");
    server.sendContent("<p>Hum: <span>" + String(dhtHum, 1) + " %</span></p>");
    server.sendContent("<p>Light: <span>" + String(lightRawValue) + "</span></p>");
    server.sendContent("<p>Manual: <span>" + String(manualOverride ? "ON" : "OFF") + "</span></p>");
    server.sendContent("<div class='log'>" + webLogs + "</div>");
    server.sendContent("<hr><p style='text-align:center;display:block;'><a href='/update'>Firmware Update</a></p>");
    server.sendContent("</div></body></html>");
    server.sendContent("");
  });

  // REST OF SETUP
  server.on("/status", []() {
    StaticJsonDocument<400> doc;
    doc["dht_temp"] = String(dhtTemp, 1);
    doc["dht_hum"] = String(dhtHum, 1);
    doc["light_raw"] = lightRawValue;
    doc["emer"] = emergencyLightOn ? "ON" : "OFF";
    doc["manual_override"] = manualOverride ? "ON" : "OFF";
    doc["heap"] = ESP.getFreeHeap();
    FSInfo fs_info;
    if (LittleFS.info(fs_info)) doc["fs_free"] = fs_info.totalBytes - fs_info.usedBytes;
    doc["ver"] = SW_VERSION;
    String response;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
  });
  server.on("/sync", []() {
    if (!LittleFS.exists(logFile)) {
      server.send(200, "text/plain", "No history available\n");
      return;
    }
    File f = LittleFS.open(logFile, "r");
    server.streamFile(f, "text/csv");
    f.close();
  });
  server.on("/config", HTTP_POST, []() {
    if (server.hasArg("plain")) {
      String body = server.arg("plain");
      StaticJsonDocument<256> doc;
      deserializeJson(doc, body);
      if (doc.containsKey("mqtt_broker")) mqttBroker = doc["mqtt_broker"].as<String>();
      if (doc.containsKey("mqtt_port")) mqttPort = doc["mqtt_port"].as<int>();
      saveSettings();
      server.send(200, "application/json", "{\"status\":\"success\"}");
    } else {
      server.send(400, "text/plain", "Bad Request");
    }
  });

  // PREPARE FOR UPDATE: Stop background tasks to free RAM before updating
  server.on("/update", HTTP_GET, []() {
    addWebLog("System entering update mode...");
    // Stop memory-intensive services
    mqttClient.disconnect();
    fbdo.clear();
    
    String html = "<html><head><style>body{font-family:sans-serif;text-align:center;padding:50px;} .btn{background:#3498db;color:white;padding:15px 30px;text-decoration:none;border-radius:5px;font-weight:bold;}</style></head><body>";
    html += "<h1>Firmware Update Mode</h1><p>RAM has been freed for a stable update.</p>";
    html += "<br><br><a class='btn' href='/update_now'>Proceed to Upload Page &rarr;</a>";
    html += "</body></html>";
    server.send(200, "text/html", html);
  });

  httpUpdater.setup(&server, "/update_now");

  if (WiFi.status() == WL_CONNECTED) {
    if (MDNS.begin(mqttClientId.c_str())) {
      Serial.println("mDNS responder started: http://" + mqttClientId + ".local");
    }
  }

  server.begin();
  MDNS.addService("http", "tcp", 80);

  mqttClient.setServer(mqttBroker.c_str(), mqttPort);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setBufferSize(512); // Increased to 512 to avoid truncation of long URLs

  dht.begin();

  // Publish initial status
  publishStatus();
}

void loop()
{
  static float old_dhtTemp = 0, old_dhtHum = 0;
  static int old_lightRawValue = 0;

  server.handleClient();
  MDNS.update();

  if (WiFi.status() != WL_CONNECTED) {
    reconnectWiFi();
  } else {
    if (!mqttClient.connected()) {
      reconnectMqtt();
    } else {
      mqttClient.loop();
    }
  }

  static unsigned long lastSensorRead = 0;
  static unsigned long lastPeriodicPublish = 0;

  if (millis() - lastSensorRead > 2000) {
    lastSensorRead = millis();

    if (enDht) {
      dhtTemp = dht.readTemperature();
      dhtHum = dht.readHumidity();
    }
  }
  checkCloudCommands();

  if (enLight) {
    lightRawValue = analogRead(LDR_PIN);
    if (!manualOverride) {
      bool shouldBeOn = (lightRawValue < 500);  // Threshold for dark
      if (shouldBeOn != emergencyLightOn) {
        emergencyLightOn = shouldBeOn;
        digitalWrite(EMERGENCY_LIGHT_PIN, emergencyLightOn ? HIGH : LOW);
        addWebLog(emergencyLightOn ? "Auto Light ON" : "Auto Light OFF");
        logData("Auto Light Change");
      }
    }
  }

  static unsigned long lastLog = 0;
  if (millis() - lastLog > 3600000) {
    lastLog = millis();
    logData("Hourly Log");
  }

  if ((old_dhtTemp != dhtTemp) ||
      (old_dhtHum != dhtHum) ||
      (old_lightRawValue != lightRawValue)) {
    old_lightRawValue = lightRawValue;
    old_dhtTemp = dhtTemp;
    old_dhtHum = dhtHum;
    publishStatus();
    lastPeriodicPublish = millis();
  }

  // Periodic publish every 30 seconds even if no change
  if (millis() - lastPeriodicPublish > 30000) {
    lastPeriodicPublish = millis();
    publishStatus();
  }
}
