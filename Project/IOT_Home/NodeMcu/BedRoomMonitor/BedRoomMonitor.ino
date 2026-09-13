#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <WiFiUdp.h>
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
#include <Wire.h>
#include <Firebase_ESP_Client.h>
#include <IRremoteESP8266.h>
#include <IRsend.h>
#include <IRutils.h>
#include "credentials.h"

#define MQ135_ANALOG_PIN A0
#define MQ135_DIGITAL_PIN D0
#define IR_TRANSMITTER_PIN D5
#define LDR_DIGITAL_PIN D3
#define DHTPIN D6
#define DHTTYPE DHT11
#define PIR_PIN D7
#define BUZZER_PIN D8

DHT dht(DHTPIN, DHTTYPE);
IRsend irsend(IR_TRANSMITTER_PIN);

const char* ssid = HOME_NETWORK_SSID;
const char* password = HOME_NETWORK_PASSWORD;

WiFiClient wifiClient;
WiFiClientSecure secureClient;
PubSubClient mqttClient;
ESP8266WebServer server(80);
ESP8266HTTPUpdateServer httpUpdater;

FirebaseData fbdo;
FirebaseAuth fbAuth;
FirebaseConfig fbConfig;

String mqttBroker = MQTT_BROKER;
int mqttPort = MQTT_PORT;
String mqttClientId = "BedRoomNode_" + String(ESP.getChipId(), HEX);
String baseTopic = "smart_home/";
String statusTopic = baseTopic + mqttClientId + "/status";
String cmdTopic = baseTopic + mqttClientId + "/commands";
String globalCmdTopic = baseTopic + "all/commands";
String discoveryTopic = baseTopic + "nodes/discovery";
String historyTopic = baseTopic + mqttClientId + "/history";

const String SW_VERSION = "1.0.385";

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
      String response = "{\"ip\":\"" + WiFi.localIP().toString() + "\",\"id\":\"" + mqttClientId + "\",\"name\":\"bedroom\",\"ver\":\"" + SW_VERSION + "\"}";
      udpDiscovery.beginPacket(udpDiscovery.remoteIP(), udpDiscovery.remotePort());
      udpDiscovery.print(response);
      udpDiscovery.endPacket();
    }
  }
}

int currentMqttPortIndex = 0;
const int mqttPorts[] = {1883, 8000, 8883, 8884};
const int numMqttPorts = 4;

// Preset Raw IR signals from tested hardware (ir.ino)
const uint16_t RAW_FAN_ON[20] PROGMEM = {868, 824, 1740, 1638, 866, 822, 1738, 824, 866, 1642, 1738, 820, 868, 1640, 866, 824, 1738, 822, 868};
const uint16_t RAW_FAN_OFF[22] PROGMEM = {868, 824, 866, 824, 864, 824, 866, 824, 1736, 824, 866, 1644, 1734, 824, 866, 1642, 866, 824, 1736, 824, 864};
const uint16_t RAW_FAN_SPEED1[20] PROGMEM = {862, 828, 1708, 1672, 836, 852, 1736, 826, 864, 1644, 836, 854, 1706, 852, 836, 854, 862, 1644, 1736};
const uint16_t RAW_FAN_SPEED2[22] PROGMEM = {868, 822, 868, 822, 866, 822, 868, 822, 1738, 822, 866, 1642, 868, 820, 1740, 820, 868, 1640, 868, 822, 1740};
const uint16_t RAW_FAN_SPEED3[20] PROGMEM = {810, 880, 1680, 1698, 810, 880, 1682, 878, 808, 1698, 1680, 880, 810, 1698, 810, 880, 806, 882, 1680};
const uint16_t RAW_FAN_SPEED4[24] PROGMEM = {868, 824, 864, 824, 866, 822, 866, 824, 1736, 822, 868, 1640, 1736, 824, 866, 1642, 866, 822, 866, 824, 866, 822, 866};
const uint16_t RAW_FAN_SPEED5[24] PROGMEM = {866, 824, 866, 824, 866, 822, 866, 822, 1738, 824, 864, 1642, 868, 822, 866, 820, 868, 822, 1738, 1640, 868, 822, 866};

const uint16_t RAW_AC_ON[228] PROGMEM = {3302, 1382, 472, 1130, 472, 1130, 472, 434, 494, 408, 470, 432, 472, 1130, 474, 428, 474, 430, 472, 1130, 474, 1130, 472, 430, 472, 1130, 472, 432, 474, 432, 470, 1132, 472, 1132, 474, 430, 472, 1132, 496, 1110, 474, 432, 472, 432, 472, 1130, 468, 432, 466, 432, 474, 1134, 464, 434, 444, 454, 466, 434, 442, 454, 464, 434, 466, 436, 468, 460, 446, 438, 464, 436, 462, 436, 470, 430, 442, 456, 442, 454, 444, 452, 466, 434, 444, 1154, 468, 1132, 444, 1154, 444, 454, 442, 456, 442, 1158, 442, 454, 466, 434, 442, 1154, 444, 1154, 444, 456, 442, 458, 472, 430, 442, 454, 444, 454, 444, 456, 444, 454, 464, 1134, 442, 1154, 444, 452, 444, 454, 444, 456, 442, 456, 442, 454, 444, 1152, 444, 454, 444, 1156, 442, 1156, 442, 1154, 444, 1158, 442, 456, 442, 454, 442, 456, 442, 454, 444, 452, 444, 456, 468, 432, 444, 454, 444, 454, 442, 456, 444, 454, 442, 454, 442, 456, 442, 456, 442, 454, 442, 456, 442, 454, 466, 432, 444, 454, 442, 454, 444, 454, 464, 434, 442, 456, 442, 454, 442, 454, 474, 430, 442, 456, 442, 456, 466, 432, 444, 456, 442, 1154, 442, 454, 442, 454, 444, 454, 442, 456, 442, 1154, 442, 456, 444, 454, 442, 1154, 442, 454, 442, 456, 466, 1134, 442};
const uint16_t RAW_AC_OFF[228] PROGMEM = {3274, 1406, 466, 1134, 468, 1132, 466, 434, 466, 438, 468, 434, 472, 1130, 472, 430, 474, 430, 472, 1132, 496, 1106, 498, 406, 472, 1132, 472, 432, 496, 434, 500, 1082, 472, 1132, 474, 432, 496, 1106, 466, 1136, 468, 432, 442, 456, 464, 1134, 444, 456, 464, 434, 464, 1132, 466, 434, 462, 434, 464, 434, 468, 434, 520, 382, 474, 432, 472, 430, 442, 456, 464, 436, 442, 456, 464, 434, 464, 434, 466, 434, 442, 454, 466, 432, 498, 1104, 466, 1132, 466, 434, 464, 434, 466, 434, 466, 1134, 464, 434, 464, 434, 470, 1130, 464, 1134, 464, 434, 464, 434, 442, 454, 464, 436, 466, 432, 464, 434, 442, 1154, 442, 1154, 442, 1156, 462, 1134, 442, 454, 442, 456, 442, 454, 444, 454, 442, 456, 442, 456, 464, 436, 442, 1154, 442, 1154, 442, 1156, 468, 430, 442, 452, 442, 458, 464, 432, 476, 430, 442, 454, 442, 456, 462, 434, 442, 454, 442, 454, 442, 454, 442, 456, 442, 454, 442, 456, 468, 432, 442, 454, 442, 454, 442, 454, 442, 456, 442, 454, 464, 434, 494, 408, 464, 434, 442, 454, 442, 454, 442, 454, 442, 456, 442, 454, 442, 1154, 442, 454, 442, 456, 468, 432, 442, 454, 442, 1154, 464, 434, 442, 456, 464, 1132, 494, 406, 442, 454, 444, 1154, 442};
const uint16_t RAW_AC_26C[228] PROGMEM = {3358, 1328, 526, 1078, 524, 1078, 526, 380, 524, 378, 526, 378, 526, 1078, 522, 380, 496, 406, 524, 1080, 524, 1078, 524, 378, 526, 1078, 466, 430, 470, 430, 468, 1128, 468, 1130, 468, 430, 468, 1130, 468, 1130, 470, 428, 468, 430, 468, 1154, 502, 384, 468, 430, 468, 1130, 468, 430, 468, 430, 494, 430, 502, 380, 494, 404, 468, 428, 470, 428, 468, 430, 470, 430, 522, 376, 468, 428, 468, 430, 466, 430, 468, 430, 468, 430, 468, 1130, 468, 1128, 470, 1128, 468, 428, 470, 428, 470, 1130, 468, 430, 522, 378, 468, 1130, 468, 1128, 468, 428, 468, 428, 468, 426, 468, 430, 466, 428, 468, 432, 526, 1076, 468, 430, 468, 1128, 470, 428, 466, 428, 468, 428, 468, 428, 468, 428, 468, 1128, 470, 430, 524, 1076, 468, 1128, 468, 1128, 468, 1128, 468, 428, 468, 428, 468, 430, 494, 406, 522, 378, 494, 408, 468, 428, 468, 428, 468, 430, 468, 430, 468, 430, 468, 428, 468, 430, 468, 428, 468, 428, 468, 428, 468, 428, 468, 428, 468, 432, 526, 374, 468, 430, 468, 428, 468, 430, 468, 428, 468, 430, 468, 430, 468, 428, 468, 428, 468, 428, 466, 428, 468, 1132, 468, 428, 468, 428, 468, 428, 468, 1128, 468, 430, 468, 428, 468, 428, 468, 1128, 466, 430, 468, 428, 468, 1132, 524};

enum IrCommandType { CMD_PROTOCOL, CMD_RAW, CMD_PRESET };

// IR Queue
struct IrCommand {
  IrCommandType type;
  decode_type_t protocol;
  uint64_t code;
  int bits;
  String rawDataStr;
  String presetName;
  bool pending;
};
IrCommand irQueue = {CMD_PROTOCOL, UNKNOWN, 0, 0, "", "", false};

// Sensor variables
float dhtTemp = 0.0;
float dhtHum = 0.0;
int mq135Analog = 0;
int mq135Digital = 0;
int ldrDigital = 0;
int pirState = 0;
bool buzzerOn = false;
int buzzerFreq = 2000;
bool manualOverride = false;
bool irTransmitterMode = false;
bool isTransmitting = false;
unsigned long lastHeartbeatTime = 0;

String webLogs = "";
void addWebLog(String msg) {
  Serial.println("LOG: " + msg);
  webLogs = msg + "<br>" + webLogs;
  if (webLogs.length() > 800) webLogs = webLogs.substring(0, 800);
}

const char* logFile = "/bedroom_log.csv";
const char* settingsFile = "/settings.json";

void loadSettings() {
  if (LittleFS.exists(settingsFile)) {
    File f = LittleFS.open(settingsFile, "r");
    if (f) {
      StaticJsonDocument<256> doc;
      deserializeJson(doc, f);
      if (doc.containsKey("mqtt_broker")) mqttBroker = doc["mqtt_broker"].as<String>();
      if (doc.containsKey("mqtt_port")) mqttPort = doc["mqtt_port"].as<int>();
      if (doc.containsKey("manual_override")) manualOverride = doc["manual_override"].as<bool>();
      if (doc.containsKey("buzzer_freq")) buzzerFreq = doc["buzzer_freq"].as<int>();
      if (doc.containsKey("ir_transmitter_mode")) irTransmitterMode = doc["ir_transmitter_mode"].as<bool>();
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
    doc["manual_override"] = manualOverride;
    doc["buzzer_freq"] = buzzerFreq;
    doc["ir_transmitter_mode"] = irTransmitterMode;
    serializeJson(doc, f);
    f.close();
  }
}

void updateDisplay() {
  char displayStr[9];
  if (mq135Digital == 1) {
    strcpy(displayStr, "GAS ALRT");
  } else if (pirState == 1) {
    strcpy(displayStr, "PIR ACTV");
  } else {
    int t = (int)dhtTemp;
    int h = (int)dhtHum;
    snprintf(displayStr, sizeof(displayStr), "%2dC  %2dH", t, h);
  }

  Wire.beginTransmission(0x40);
  Wire.write(displayStr);
  Wire.endTransmission();
}

void publishStatus() {
  StaticJsonDocument<512> doc;
  doc["dht_temp"] = String(dhtTemp, 1);
  doc["dht_hum"] = String(dhtHum, 1);
  doc["mq135_analog"] = String(mq135Analog);
  doc["mq135_digital"] = String(mq135Digital);
  doc["light_raw"] = String(ldrDigital);
  doc["pir"] = String(pirState);
  doc["buzzer"] = buzzerOn ? "ON" : "OFF";
  doc["buzzer_freq"] = buzzerFreq;
  doc["manual_override"] = manualOverride ? "ON" : "OFF";
  doc["ir_transmitter_mode"] = irTransmitterMode ? "ON" : "OFF";
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
    f.println("Date,Time,Reason,Manual Override,MQ135 A,MQ135 D,DHT_T,DHT_H,Light,PIR,Buzzer");
  }

  time_t now = time(nullptr);
  struct tm timeinfo;
  localtime_r(&now, &timeinfo);
  char dBuf[12], tBuf[10];
  strftime(dBuf, sizeof(dBuf), "%Y-%m-%d", &timeinfo);
  strftime(tBuf, sizeof(tBuf), "%H:%M:%S", &timeinfo);

  String entry = String(dBuf) + "," + String(tBuf) + "," +
                 eventType + "," +
                 (manualOverride ? "ON" : "OFF") + "," +
                 String(mq135Analog) + "," +
                 String(mq135Digital) + "," +
                 String(dhtTemp, 1) + "," +
                 String(dhtHum, 1) + "," +
                 String(ldrDigital) + "," +
                 String(pirState) + "," +
                 (buzzerOn ? "ON" : "OFF") + "\n";

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

void sendPresetIR(String name) {
  name.trim();
  name.toUpperCase();
  uint16_t buffer[230];

  if (name == "FAN_ON" || name == "FANON") {
    for (int i = 0; i < 20; i++) buffer[i] = pgm_read_word(&RAW_FAN_ON[i]);
    irsend.sendRaw(buffer, 20, 38);
  } else if (name == "FAN_OFF" || name == "FANOFF") {
    for (int i = 0; i < 22; i++) buffer[i] = pgm_read_word(&RAW_FAN_OFF[i]);
    irsend.sendRaw(buffer, 22, 38);
  } else if (name == "FAN_SPEED1" || name == "FANSPEED1") {
    for (int i = 0; i < 20; i++) buffer[i] = pgm_read_word(&RAW_FAN_SPEED1[i]);
    irsend.sendRaw(buffer, 20, 38);
  } else if (name == "FAN_SPEED2" || name == "FANSPEED2") {
    for (int i = 0; i < 22; i++) buffer[i] = pgm_read_word(&RAW_FAN_SPEED2[i]);
    irsend.sendRaw(buffer, 22, 38);
  } else if (name == "FAN_SPEED3" || name == "FANSPEED3") {
    for (int i = 0; i < 20; i++) buffer[i] = pgm_read_word(&RAW_FAN_SPEED3[i]);
    irsend.sendRaw(buffer, 20, 38);
  } else if (name == "FAN_SPEED4" || name == "FANSPEED4") {
    for (int i = 0; i < 24; i++) buffer[i] = pgm_read_word(&RAW_FAN_SPEED4[i]);
    irsend.sendRaw(buffer, 24, 38);
  } else if (name == "FAN_SPEED5" || name == "FANSPEED5") {
    for (int i = 0; i < 24; i++) buffer[i] = pgm_read_word(&RAW_FAN_SPEED5[i]);
    irsend.sendRaw(buffer, 24, 38);
  } else if (name == "AC_ON" || name == "ACON") {
    for (int i = 0; i < 228; i++) buffer[i] = pgm_read_word(&RAW_AC_ON[i]);
    irsend.sendRaw(buffer, 228, 38);
  } else if (name == "AC_OFF" || name == "ACOFF") {
    for (int i = 0; i < 228; i++) buffer[i] = pgm_read_word(&RAW_AC_OFF[i]);
    irsend.sendRaw(buffer, 228, 38);
  } else if (name == "AC_26C" || name == "AC26C") {
    for (int i = 0; i < 228; i++) buffer[i] = pgm_read_word(&RAW_AC_26C[i]);
    irsend.sendRaw(buffer, 228, 38);
  }
}

void sendRawStringIR(String rawStr) {
  uint16_t rawBuf[256];
  int count = 0;
  int startIdx = 0;
  while (startIdx < rawStr.length() && count < 256) {
    int commaIdx = rawStr.indexOf(',', startIdx);
    if (commaIdx == -1) commaIdx = rawStr.length();
    String valStr = rawStr.substring(startIdx, commaIdx);
    valStr.trim();
    if (valStr.length() > 0) {
      rawBuf[count++] = (uint16_t)valStr.toInt();
    }
    startIdx = commaIdx + 1;
  }
  if (count > 0) {
    irsend.sendRaw(rawBuf, count, 38);
  }
}

decode_type_t parseProtocol(String str) {
  str.trim();
  str.toUpperCase();
  if (str == "1" || str == "3" || str == "NEC") return NEC;     // 3 in IRremoteESP8266
  if (str == "2" || str == "4" || str == "SONY") return SONY;   // 4 in IRremoteESP8266
  if (str == "RC5") return RC5;
  if (str == "RC6") return RC6;
  int val = str.toInt();
  if (val > 0) return (decode_type_t)val;
  return UNKNOWN;
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String msg = "";
  for (int i = 0; i < length; i++) msg += (char)payload[i];

  if (msg == "SYNC") {
    publishStatus();
  } else if (msg == "HEARTBEAT") {
    lastHeartbeatTime = millis();
  } else if (msg.startsWith("OTA:")) {
    handleRemoteOTA(msg.substring(4));
  } else if (msg.startsWith("IR_SEND_RAW:")) {
    irQueue.type = CMD_RAW;
    irQueue.rawDataStr = msg.substring(12);
    irQueue.pending = true;
    addWebLog("Queue Raw IR");
  } else if (msg.startsWith("IR_SEND_PRESET:")) {
    irQueue.type = CMD_PRESET;
    irQueue.presetName = msg.substring(15);
    irQueue.pending = true;
    addWebLog("Queue Preset IR: " + irQueue.presetName);
  } else if (msg.startsWith("IR_SEND:")) {
    String data = msg.substring(8);
    if (data.startsWith("PRESET:")) {
      irQueue.type = CMD_PRESET;
      irQueue.presetName = data.substring(7);
      irQueue.pending = true;
      addWebLog("Queue Preset IR: " + irQueue.presetName);
    } else if (data.startsWith("RAW:")) {
      irQueue.type = CMD_RAW;
      irQueue.rawDataStr = data.substring(4);
      irQueue.pending = true;
      addWebLog("Queue Raw IR");
    } else {
      int comma1 = data.indexOf(',');
      int comma2 = data.lastIndexOf(',');
      if (comma1 != -1 && comma2 != -1) {
        String protocolStr = data.substring(0, comma1);
        String hexStr = data.substring(comma1 + 1, comma2);
        int bits = data.substring(comma2 + 1).toInt();

        uint64_t code = strtoull(hexStr.c_str(), NULL, 16);
        decode_type_t protocol = parseProtocol(protocolStr);

        irQueue.type = CMD_PROTOCOL;
        irQueue.protocol = protocol;
        irQueue.code = code;
        irQueue.bits = bits;
        irQueue.pending = true;

        addWebLog("Queue IR: " + hexStr + " (" + String((int)protocol) + ")");
      } else if (data.length() > 0) {
        irQueue.type = CMD_PRESET;
        irQueue.presetName = data;
        irQueue.pending = true;
        addWebLog("Queue Preset IR: " + irQueue.presetName);
      }
    }
  } else if (msg == "DISCOVER") {
    String discoveryMsg = "{\"ip\":\"" + WiFi.localIP().toString() + "\", \"id\":\"" + mqttClientId + "\", \"ver\":\"" + SW_VERSION + "\"}";
#ifdef ENABLE_MQTT
    mqttClient.publish(discoveryTopic.c_str(), discoveryMsg.c_str());
#endif
  } else if (msg == "CLEAR") {
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
    if (doc.containsKey("buzzer")) {
        buzzerOn = doc["buzzer"].as<bool>();
    }
    if (doc.containsKey("buzzer_freq")) {
        buzzerFreq = doc["buzzer_freq"].as<int>();
    }
    if (doc.containsKey("ir_transmitter_mode")) {
        irTransmitterMode = doc["ir_transmitter_mode"].as<bool>();
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
  pinMode(MQ135_DIGITAL_PIN, INPUT);
  pinMode(LDR_DIGITAL_PIN, INPUT);
  pinMode(PIR_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Wire.begin(D2, D1); // SDA = D2, SCL = D1 as per requirement layout

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
    server.sendContent("</style></head><body><div class='c'><h3>Bed Room Monitor</h3>");

    server.sendContent("<p>Ver: <span>" + SW_VERSION + "</span></p>");
    server.sendContent("<p>Heap: <span>" + String(ESP.getFreeHeap()) + "</span></p>");
    server.sendContent("<hr>");
    server.sendContent("<p>Temp: <span>" + String(dhtTemp, 1) + " C</span></p>");
    server.sendContent("<p>Hum: <span>" + String(dhtHum, 1) + " %</span></p>");
    server.sendContent("<p>MQ135 Analog: <span>" + String(mq135Analog) + "</span></p>");
    server.sendContent("<p>MQ135 Digital: <span>" + String(mq135Digital == 1 ? "ALERT" : "NORMAL") + "</span></p>");
    server.sendContent("<p>Light: <span>" + String(ldrDigital) + "</span></p>");
    server.sendContent("<p>PIR: <span>" + String(pirState == 1 ? "ACTIVE" : "IDLE") + "</span></p>");
    server.sendContent("<p>Manual: <span>" + String(manualOverride ? "ON" : "OFF") + "</span></p>");
    server.sendContent("<p>IR Transmit: <span>" + String(irTransmitterMode ? "ON" : "OFF") + "</span></p>");
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
    doc["mq135_analog"] = String(mq135Analog);
    doc["mq135_digital"] = String(mq135Digital);
    doc["light_raw"] = String(ldrDigital);
    doc["pir"] = String(pirState);
    doc["buzzer"] = buzzerOn ? "ON" : "OFF";
    doc["buzzer_freq"] = buzzerFreq;
    doc["manual_override"] = manualOverride ? "ON" : "OFF";
    doc["ir_transmitter_mode"] = irTransmitterMode ? "ON" : "OFF";
    doc["heap"] = ESP.getFreeHeap();
    doc["ver"] = SW_VERSION;
    doc["ip"] = WiFi.localIP().toString();
    doc["id"] = mqttClientId;
    String response;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
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

  server.on("/control", []() {
    if (server.hasArg("cmd")) {
      String cmd = server.arg("cmd");
      mqttCallback((char*)cmdTopic.c_str(), (byte*)cmd.c_str(), cmd.length());
      server.send(200, "text/plain", "OK");
    } else {
      server.send(400, "text/plain", "Missing cmd param");
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
  dht.begin();
  irsend.begin();
  setupUdpDiscovery();
  ESP.wdtEnable(WDTO_8S);
  publishStatus();
}

void loop() {
  ESP.wdtFeed();
  server.handleClient();
  checkUdpDiscovery();

  if (irQueue.pending) {
    isTransmitting = true;
    if (irQueue.type == CMD_PRESET) {
      addWebLog("TX Preset IR: " + irQueue.presetName);
      sendPresetIR(irQueue.presetName);
    } else if (irQueue.type == CMD_RAW) {
      addWebLog("TX Raw IR");
      sendRawStringIR(irQueue.rawDataStr);
    } else if (irQueue.type == CMD_PROTOCOL) {
      addWebLog("TX IR Code: 0x" + String((uint32_t)irQueue.code, HEX));

      if (irQueue.protocol == NEC) {
        // Send NEC code with 1 repeat frame (matching physical remote output)
        irsend.sendNEC(irQueue.code, irQueue.bits, 1);
      } else if (irQueue.protocol == SONY) {
        irsend.sendSony(irQueue.code, irQueue.bits, 2);
      } else {
        irsend.send(irQueue.protocol, irQueue.code, irQueue.bits);
      }
    }
    irQueue.pending = false;
    isTransmitting = false;
  }

  if (buzzerOn) {
    tone(BUZZER_PIN, buzzerFreq);
  } else {
    noTone(BUZZER_PIN);
  }

  static unsigned long lastSensorRead = 0;
  static unsigned long lastPeriodicPublish = 0;

  if (!isTransmitting && millis() - lastSensorRead > 2000) {
    lastSensorRead = millis();
    float new_dhtTemp = dht.readTemperature();
    float new_dhtHum = dht.readHumidity();
    int new_mq135Analog = analogRead(MQ135_ANALOG_PIN);
    int new_mq135Digital = digitalRead(MQ135_DIGITAL_PIN);
    int new_ldrDigital = digitalRead(LDR_DIGITAL_PIN);
    int new_pirState = digitalRead(PIR_PIN);

    bool changed = false;
    if (abs(new_dhtTemp - dhtTemp) >= 0.2 ||
        abs(new_dhtHum - dhtHum) >= 1.0 ||
        abs(new_mq135Analog - mq135Analog) > 20 ||
        new_mq135Digital != mq135Digital ||
        new_ldrDigital != ldrDigital ||
        new_pirState != pirState) {
      changed = true;
    }

    dhtTemp = new_dhtTemp;
    dhtHum = new_dhtHum;
    mq135Analog = new_mq135Analog;
    mq135Digital = new_mq135Digital;
    ldrDigital = new_ldrDigital;
    pirState = new_pirState;

    if (!manualOverride) {
      if (mq135Digital == 1) {
        if (!buzzerOn) {
          buzzerOn = true;
          logData("Auto Gas Alarm ON");
          changed = true;
        }
      } else {
        if (buzzerOn) {
          buzzerOn = false;
          logData("Auto Gas Alarm OFF");
          changed = true;
        }
      }
    }
    updateDisplay();

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
