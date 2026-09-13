#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
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
#include <IRremoteESP8266.h>
#include <IRrecv.h>
#include <IRutils.h>
#include "credentials.h"

// Compile-time option to enable/disable MQTT
//#define ENABLE_MQTT

#define LDR_PIN A0
#define EMERGENCY_LIGHT_PIN D2
#define IR_RECEIVER_PIN D4
#define DHTPIN D6
#define DHTTYPE DHT11
#define BUZZER_PIN D8

DHT dht(DHTPIN, DHTTYPE);
IRrecv irrecv(IR_RECEIVER_PIN);
decode_results irResults;

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

const String SW_VERSION = "1.0.416";

WiFiUDP udpDiscovery;
const int UDP_DISCOVERY_PORT = 8888;

void setupUdpDiscovery() {
  udpDiscovery.begin(UDP_DISCOVERY_PORT);
}

void checkUdpDiscovery() {
  int packetSize = udpDiscovery.parsePacket();
  if (packetSize) {
    char buf[255];
    int len = udpDiscovery.read(buf, 255);
    if (len > 0) buf[len] = 0;
    String msg = String(buf);
    msg.trim();
    if (msg == "DISCOVER" || msg.indexOf("DISCOVER") != -1) {
      String response = "{\"ip\":\"" + WiFi.localIP().toString() + "\",\"id\":\"" + mqttClientId + "\",\"name\":\"study\",\"ver\":\"" + SW_VERSION + "\"}";
      udpDiscovery.beginPacket(udpDiscovery.remoteIP(), udpDiscovery.remotePort());
      udpDiscovery.print(response);
      udpDiscovery.endPacket();
    }
  }
}

int currentMqttPortIndex = 0;
const int mqttPorts[] = {1883, 8000, 8883, 8884};
const int numMqttPorts = 4;

// Sensor and State Variables
float dhtTemp = 0.0;
float dhtHum = 0.0;
int lightRawValue = 0;
bool emergencyLightOn = false;
bool manualOverride = false;
bool buzzerOn = false;
int buzzerFreq = 2000;
bool irReceiveMode = false;
unsigned long lastHeartbeatTime = 0;

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
      if (doc.containsKey("buzzer_freq")) buzzerFreq = doc["buzzer_freq"].as<int>();
      if (doc.containsKey("ir_receive_mode")) irReceiveMode = doc["ir_receive_mode"].as<bool>();
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
    doc["buzzer_freq"] = buzzerFreq;
    doc["ir_receive_mode"] = irReceiveMode;
    serializeJson(doc, f);
    f.close();
  }
}

void publishStatus() {
  StaticJsonDocument<512> doc;
  doc["dht_temp"] = String(dhtTemp, 1);
  doc["dht_hum"] = String(dhtHum, 1);
  doc["light_raw"] = lightRawValue;
  doc["emer"] = emergencyLightOn ? "ON" : "OFF";
  doc["buzzer"] = buzzerOn ? "ON" : "OFF";
  doc["buzzer_freq"] = buzzerFreq;
  doc["manual_override"] = manualOverride ? "ON" : "OFF";
  doc["ir_receive_mode"] = irReceiveMode ? "ON" : "OFF";
  doc["heap"] = ESP.getFreeHeap();
  doc["ver"] = SW_VERSION;
  doc["ip"] = WiFi.localIP().toString();
  doc["id"] = mqttClientId;

  char buffer[400];
  serializeJson(doc, buffer);

#ifdef ENABLE_MQTT
  if (mqttClient.connected()) {
    mqttClient.publish(statusTopic.c_str(), buffer);
  }
#endif

  if (Firebase.ready()) {
    if (millis() - lastHeartbeatTime < 60000) {
      FirebaseJson json;
      json.setJsonData(buffer);
      Firebase.RTDB.setJSON(&fbdo, String(FB_OUTBOX) + "/status", &json);
    }
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

  if (Firebase.ready()) {
    if (millis() - lastHeartbeatTime < 60000) {
      String trimmedEntry = entry;
      trimmedEntry.trim();
      Firebase.RTDB.pushString(&fbdo, String(FB_OUTBOX) + "/history", trimmedEntry);
    }
  }
}

void handleRemoteOTA(String url) {
  url.trim();
  addWebLog("OTA Pending. Restarting...");
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
  }
}

void runPendingOTA() {
  String url = "";
  if (LittleFS.exists("/ota.txt")) {
    File f = LittleFS.open("/ota.txt", "r");
    url = f.readString();
    f.close();
    LittleFS.remove("/ota.txt");
  }
  url.trim();
  if (url.length() < 10) return;

  delay(5000);
  system_update_cpu_freq(160);
  ESP.wdtEnable(20000);

  WiFiClientSecure sClient;
  sClient.setInsecure();
  sClient.setBufferSizes(16384, 1024);

  ESPhttpUpdate.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
  ESPhttpUpdate.rebootOnUpdate(true);
  ESPhttpUpdate.update(sClient, url);
  system_update_cpu_freq(80);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String msg = "";
  for (int i = 0; i < length; i++) msg += (char)payload[i];

  if (msg == "SYNC") {
    publishStatus();
  } else if (msg == "HEARTBEAT") {
    lastHeartbeatTime = millis();
    publishStatus();
  } else if (msg.startsWith("OTA:")) {
    handleRemoteOTA(msg.substring(4));
  } else if (msg == "DISCOVER") {
    String discoveryMsg = "{\"ip\":\"" + WiFi.localIP().toString() + "\", \"id\":\"" + mqttClientId + "\", \"ver\":\"" + SW_VERSION + "\"}";
#ifdef ENABLE_MQTT
    mqttClient.publish(discoveryTopic.c_str(), discoveryMsg.c_str());
#endif
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

    if (doc.containsKey("manual_override")) {
        manualOverride = doc["manual_override"].as<bool>();
    }
    if (doc.containsKey("en_emer")) {
        emergencyLightOn = doc["en_emer"].as<bool>();
        digitalWrite(EMERGENCY_LIGHT_PIN, emergencyLightOn ? HIGH : LOW);
    }
    if (doc.containsKey("buzzer")) {
        buzzerOn = doc["buzzer"].as<bool>();
    }
    if (doc.containsKey("buzzer_freq")) {
        buzzerFreq = doc["buzzer_freq"].as<int>();
    }
    if (doc.containsKey("ir_receive_mode")) {
        irReceiveMode = doc["ir_receive_mode"].as<bool>();
        if (irReceiveMode) {
            irrecv.enableIRIn();
            addWebLog("IR Receive Enabled");
        } else {
            irrecv.disableIRIn();
            addWebLog("IR Receive Disabled");
        }
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
#ifdef ENABLE_MQTT
          mqttClient.publish(historyTopic.c_str(), line.c_str());
#endif
        }
      }
      f.close();
#ifdef ENABLE_MQTT
      mqttClient.publish(historyTopic.c_str(), "EOF");
#endif
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
      }
    }
  }
}

void setupFirebase() {
  fbConfig.database_url = "https://" + String(FIREBASE_HOST) + "/";
  fbConfig.api_key = FIREBASE_API_KEY;
  fbAuth.user.email = FIREBASE_USER_EMAIL;
  fbAuth.user.password = FIREBASE_USER_PASSWORD;
  Firebase.begin(&fbConfig, &fbAuth);
  Firebase.reconnectWiFi(true);
  addWebLog("Firebase Ready");
}

void setup() {
  Serial.begin(115200);
  pinMode(EMERGENCY_LIGHT_PIN, OUTPUT);
  digitalWrite(EMERGENCY_LIGHT_PIN, LOW);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

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
    server.sendContent("</style></head><body><div class='c'><h3>Study Room Monitor</h3>");

    server.sendContent("<p>Ver: <span>" + SW_VERSION + "</span></p>");
    server.sendContent("<p>Heap: <span>" + String(ESP.getFreeHeap()) + "</span></p>");
    server.sendContent("<hr>");
    server.sendContent("<p>Temp: <span>" + String(dhtTemp, 1) + " C</span></p>");
    server.sendContent("<p>Hum: <span>" + String(dhtHum, 1) + " %</span></p>");
    server.sendContent("<p>Light: <span>" + String(lightRawValue) + "</span></p>");
    server.sendContent("<p>Manual: <span>" + String(manualOverride ? "ON" : "OFF") + "</span></p>");
    server.sendContent("<p>IR Receive: <span>" + String(irReceiveMode ? "ON" : "OFF") + "</span></p>");
    server.sendContent("<p>Buzzer: <span>" + String(buzzerOn ? "ON" : "OFF") + " (" + String(buzzerFreq) + " Hz)</span></p>");
    server.sendContent("<div class='log'>" + webLogs + "</div>");
    server.sendContent("<hr><p style='text-align:center;display:block;'><a href='/update'>Firmware Update</a></p>");
    server.sendContent("</div></body></html>");
    server.sendContent("");
  });

  server.on("/status", []() {
    StaticJsonDocument<400> doc;
    doc["dht_temp"] = String(dhtTemp, 1);
    doc["dht_hum"] = String(dhtHum, 1);
    doc["light_raw"] = lightRawValue;
    doc["emer"] = emergencyLightOn ? "ON" : "OFF";
    doc["buzzer"] = buzzerOn ? "ON" : "OFF";
    doc["buzzer_freq"] = buzzerFreq;
    doc["manual_override"] = manualOverride ? "ON" : "OFF";
    doc["ir_receive_mode"] = irReceiveMode ? "ON" : "OFF";
    doc["heap"] = ESP.getFreeHeap();
    doc["ver"] = SW_VERSION;
    doc["ip"] = WiFi.localIP().toString();
    doc["id"] = mqttClientId;
    String response;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
  });

  server.on("/control", []() {
    if (server.hasArg("cmd")) {
      String cmd = server.arg("cmd");
      if (cmd == "HEARTBEAT") {
        lastHeartbeatTime = millis();
      }
      StaticJsonDocument<400> doc;
      doc["dht_temp"] = String(dhtTemp, 1);
      doc["dht_hum"] = String(dhtHum, 1);
      doc["light_raw"] = lightRawValue;
      doc["emer"] = emergencyLightOn ? "ON" : "OFF";
      doc["buzzer"] = buzzerOn ? "ON" : "OFF";
      doc["buzzer_freq"] = buzzerFreq;
      doc["manual_override"] = manualOverride ? "ON" : "OFF";
      doc["ir_receive_mode"] = irReceiveMode ? "ON" : "OFF";
      doc["heap"] = ESP.getFreeHeap();
      doc["ver"] = SW_VERSION;
      doc["ip"] = WiFi.localIP().toString();
      doc["id"] = mqttClientId;
      String response;
      serializeJson(doc, response);
      server.send(200, "application/json", response);
    } else {
      server.send(400, "text/plain", "Bad Request");
    }
  });

  server.on("/clear", []() {
    LittleFS.remove(logFile);
    addWebLog("History Cleared via IP");
    if (Firebase.ready()) {
      Firebase.RTDB.deleteNode(&fbdo, String(FB_OUTBOX) + "/history");
    }
    server.send(200, "text/plain", "History Cleared");
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

  server.on("/update", HTTP_GET, []() {
    addWebLog("System entering update mode...");
    ESP.wdtEnable(20000);
    fbdo.clear();
    String html = "<html><head><style>body{font-family:sans-serif;text-align:center;padding:50px;} .btn{background:#3498db;color:white;padding:15px 30px;text-decoration:none;border-radius:5px;font-weight:bold;}</style></head><body>";
    html += "<h1>Firmware Update Mode</h1><p>RAM has been freed for a stable update.</p>";
    html += "<br><br><a class='btn' href='/update_now'>Proceed to Upload Page &rarr;</a>";
    html += "</body></html>";
    server.send(200, "text/html", html);
  });

  httpUpdater.setup(&server, "/update_now");

  server.begin();
  setupUdpDiscovery();
  dht.begin();
  if (irReceiveMode) {
    irrecv.enableIRIn();
  }
  ESP.wdtEnable(WDTO_8S);
  publishStatus();
}

void loop() {
  static unsigned long lastIrCaptureTime = 0;
  if (irReceiveMode) {
    if (irrecv.decode(&irResults)) {
      String hexStr = resultToHexidecimal(&irResults);
      String protocol = typeToString(irResults.decode_type);

      // Filter out common ambient noise or partial trash, and rate limit captures (minimum 300ms gap)
      if (irResults.bits >= 8 && (millis() - lastIrCaptureTime > 300)) {
          lastIrCaptureTime = millis();
          String logMsg = "IR: " + hexStr + " (" + protocol + " " + String(irResults.bits) + "b)";

          if (irResults.decode_type == UNKNOWN) {
              // Add raw timings for debugging UNKNOWN signals
              logMsg += " RAW[" + String(irResults.rawlen) + "]: ";
              for (uint16_t i = 1; i < irResults.rawlen; i++) {
                  logMsg += String(irResults.rawbuf[i] * kRawTick) + ",";
                  if (i > 10) { logMsg += "..."; break; }
              }
          }
          addWebLog(logMsg);

          if (Firebase.ready()) {
            FirebaseJson json;
            json.set("hex", hexStr);
            json.set("protocol", protocol);
            json.set("bits", irResults.bits);
            String rawStr = "";
            for (uint16_t i = 1; i < irResults.rawlen; i++) {
                rawStr += String(rawStr.length() > 0 ? "," : "") + String(irResults.rawbuf[i] * kRawTick);
            }
            json.set("raw", rawStr);
            Firebase.RTDB.pushJSON(&fbdo, String(FB_OUTBOX) + "/captured_ir", &json);
          }
      }
      irrecv.resume();
    }
  }

  ESP.wdtFeed();
  checkUdpDiscovery();
  server.handleClient();

  if (buzzerOn) {
    tone(BUZZER_PIN, buzzerFreq);
  } else {
    noTone(BUZZER_PIN);
  }

  static unsigned long lastSensorRead = 0;
  static unsigned long lastPeriodicPublish = 0;

  if (millis() - lastSensorRead > 2000) {
    lastSensorRead = millis();
    float new_dhtTemp = dht.readTemperature();
    float new_dhtHum = dht.readHumidity();
    int new_lightRawValue = analogRead(LDR_PIN);

    bool changed = false;
    if (abs(new_dhtTemp - dhtTemp) >= 0.2 ||
        abs(new_dhtHum - dhtHum) >= 1.0 ||
        abs(new_lightRawValue - lightRawValue) > 20) {
      changed = true;
    }

    dhtTemp = new_dhtTemp;
    dhtHum = new_dhtHum;
    lightRawValue = new_lightRawValue;

    if (!manualOverride) {
      bool shouldBeOn = (lightRawValue < 500);
      if (shouldBeOn != emergencyLightOn) {
        emergencyLightOn = shouldBeOn;
        digitalWrite(EMERGENCY_LIGHT_PIN, emergencyLightOn ? HIGH : LOW);
        addWebLog(emergencyLightOn ? "Auto Light ON" : "Auto Light OFF");
        logData("Auto Light Change");
        changed = true;
      }
    }

    if (changed) {
      publishStatus();
    }
  }

  checkCloudCommands();

  if (millis() - lastPeriodicPublish > 30000) {
    lastPeriodicPublish = millis();
    publishStatus();
  }
}
