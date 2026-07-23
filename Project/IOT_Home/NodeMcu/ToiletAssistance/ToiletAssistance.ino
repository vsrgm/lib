#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoJson.h>
#include <LittleFS.h>
#include <time.h>
#include <ArduinoOTA.h>
#include <ESP8266HTTPUpdateServer.h>
#include <ESP8266mDNS.h>
#include <PubSubClient.h>
#include <DHT.h>
#include <Firebase_ESP_Client.h>
#include "credentials.h"

// Pin Definitions
#define LDR_PIN A0
#define FAN_PIN D0
#define PIR_PIN D1
#define LIGHT_PIN D2
#define DHT_PIN D3
#define BUZZER_PIN D4

#define DHTTYPE DHT11
DHT dht(DHT_PIN, DHTTYPE);

// WiFi & MQTT
const char* ssid = HOME_NETWORK_SSID;
const char* password = HOME_NETWORK_PASSWORD;

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
ESP8266WebServer server(80);
ESP8266HTTPUpdateServer httpUpdater;

// Firebase Data objects
FirebaseData fbdo;
FirebaseAuth fbAuth;
FirebaseConfig fbConfig;

String mqttBroker = MQTT_BROKER;
int mqttPort = MQTT_PORT;
int currentMqttPortIndex = 0;
const int mqttPorts[] = {1883, 8000, 8883, 8884};
const int numMqttPorts = 4;

String mqttClientId = "ToiletAssistance_" + String(ESP.getChipId(), HEX);
String baseTopic = "smart_home/toilet/";
String statusTopic = baseTopic + "status";
String cmdTopic = baseTopic + "commands";

const String SW_VERSION = "1.0.304";

// System State
float temperature = 0.0;
float humidity = 0.0;
int ldrValue = 0;
bool pirState = false;
bool fanState = false;
bool lightState = false;
bool buzzerState = false;

unsigned long lastPirMovement = 0;
unsigned long fanTurnOnTime = 0;
unsigned long lightTurnOnTime = 0;
bool buzzerActive = false;

const char* logFile = "/toilet_log.csv";

void handleFirebaseStream(FirebaseStream data) {
    if (data.dataPath() == "/command") {
        String msg = data.stringData();
        Serial.println("Firebase Command Received: " + msg);
        handleCommand(msg);
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

    pinMode(FAN_PIN, OUTPUT);
    pinMode(LIGHT_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(PIR_PIN, INPUT);

    digitalWrite(FAN_PIN, LOW);
    digitalWrite(LIGHT_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);

    if (!LittleFS.begin()) {
        Serial.println("LittleFS Mount Failed");
    }

    dht.begin();
    setupWiFi();
    setupTime();

    mqttClient.setServer(mqttBroker.c_str(), mqttPort);
    mqttClient.setCallback(mqttCallback);

    setupFirebase();

    // Publish initial status
    publishStatus();

    httpUpdater.setup(&server);
    setupRoutes();
    server.begin();

    ArduinoOTA.setHostname("ToiletNode");
    ArduinoOTA.begin();

    logData("System Boot");
}

void setupWiFi() {
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
        server.handleClient();
    }
    Serial.println("\nWiFi Connected. IP: " + WiFi.localIP().toString());
}

void setupTime() {
    configTime(5.5 * 3600, 0, "pool.ntp.org", "time.nist.gov");
    Serial.print("Waiting for NTP time sync: ");
    time_t now = time(nullptr);
    while (now < 8 * 3600 * 2) {
        delay(500);
        Serial.print(".");
        now = time(nullptr);
        server.handleClient();
    }
    Serial.println("\nTime Synchronized.");
}

void setupRoutes() {
    server.on("/status", HTTP_GET, []() {
        StaticJsonDocument<512> doc;
        doc["pir"] = pirState;
        doc["ldr"] = ldrValue < 300; // Low light threshold
        doc["fan"] = fanState;
        doc["light"] = lightState;
        doc["buzzer"] = buzzerState;
        doc["temp"] = temperature;
        doc["hum"] = humidity;
        doc["ver"] = SW_VERSION;

        FSInfo fs_info;
        if (LittleFS.info(fs_info)) {
            doc["fs_free"] = fs_info.totalBytes - fs_info.usedBytes;
        }

        String response;
        serializeJson(doc, response);
        server.send(200, "application/json", response);
    });

    server.on("/control", HTTP_GET, []() {
        String cmd = server.arg("cmd");
        handleCommand(cmd);
        server.send(200, "text/plain", "OK");
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
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
    String msg = "";
    for (int i = 0; i < length; i++) msg += (char)payload[i];
    handleCommand(msg);
}

void handleCommand(String cmd) {
    if (cmd == "FAN_ON") { fanState = true; digitalWrite(FAN_PIN, HIGH); }
    else if (cmd == "FAN_OFF") { fanState = false; digitalWrite(FAN_PIN, LOW); }
    else if (cmd == "LIGHT_ON") { lightState = true; digitalWrite(LIGHT_PIN, HIGH); }
    else if (cmd == "LIGHT_OFF") { lightState = false; digitalWrite(LIGHT_PIN, LOW); }
    else if (cmd == "BUZZER_ON") { buzzerState = true; digitalWrite(BUZZER_PIN, HIGH); }
    else if (cmd == "BUZZER_OFF") { buzzerState = false; digitalWrite(BUZZER_PIN, LOW); }

    publishStatus();
    logData("Manual Command: " + cmd);
}

void publishStatus() {
    StaticJsonDocument<512> doc;
    doc["pir"] = pirState;
    doc["ldr"] = ldrValue < 300;
    doc["fan"] = fanState;
    doc["light"] = lightState;
    doc["buzzer"] = buzzerState;
    doc["temp"] = temperature;
    doc["hum"] = humidity;
    doc["ver"] = SW_VERSION;

    FSInfo fs_info;
    if (LittleFS.info(fs_info)) {
        doc["fs_free"] = fs_info.totalBytes - fs_info.usedBytes;
    }

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

void logData(String reason) {
    File f = LittleFS.open(logFile, "a");
    if (!f) return;

    time_t now = time(nullptr);
    struct tm* timeinfo = localtime(&now);
    char timestamp[25];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", timeinfo);

    f.printf("%s,%d,%d,%d,%d,%.1f,%.1f,%s\n",
        timestamp, ldrValue, fanState, pirState, lightState, temperature, humidity, reason.c_str());
    f.close();

    // Parallel Push to Firebase History
    if (Firebase.ready()) {
        String entry = String(timestamp) + "," + reason;
        Firebase.RTDB.pushString(&fbdo, FIREBASE_NODE "/history", entry);
    }
}

void loop() {
    server.handleClient();
    mqttClient.loop();
    ArduinoOTA.handle();

    if (WiFi.status() == WL_CONNECTED) {
        if (!mqttClient.connected()) reconnectMqtt();
    }

    static unsigned long lastMeasure = 0;
    static unsigned long lastPeriodicPublish = 0;

    if (millis() - lastMeasure > 2000) {
        lastMeasure = millis();
        readSensors();
        checkAutomation();
    }

    if (millis() - lastPeriodicPublish > 30000) {
        lastPeriodicPublish = millis();
        publishStatus();
    }
}

void readSensors() {
    temperature = dht.readTemperature();
    humidity = dht.readHumidity();
    ldrValue = analogRead(LDR_PIN);
    bool currentPir = digitalRead(PIR_PIN);

    if (currentPir != pirState) {
        pirState = currentPir;
        if (pirState) lastPirMovement = millis();
        publishStatus();
        logData("PIR Change");
    }
}

void checkAutomation() {
    unsigned long now = millis();

    // 0.1.2. PIR -> Fan ON for 1 min
    if (pirState) {
        if (!fanState) {
            fanState = true;
            digitalWrite(FAN_PIN, HIGH);
            fanTurnOnTime = now;
            publishStatus();
            logData("Auto Fan ON");
        } else {
            fanTurnOnTime = now; // Reset timer while moving
        }
    } else if (fanState && (now - fanTurnOnTime > 60000)) {
        fanState = false;
        digitalWrite(FAN_PIN, LOW);
        publishStatus();
        logData("Auto Fan OFF");
    }

    // 0.1.3. PIR + Low Light -> Light ON for 1 min
    if (pirState && (ldrValue < 300)) {
        if (!lightState) {
            lightState = true;
            digitalWrite(LIGHT_PIN, HIGH);
            lightTurnOnTime = now;
            publishStatus();
            logData("Auto Light ON");
        } else {
            lightTurnOnTime = now;
        }
    } else if (lightState && (now - lightTurnOnTime > 60000)) {
        lightState = false;
        digitalWrite(LIGHT_PIN, LOW);
        publishStatus();
        logData("Auto Light OFF");
    }

    // 0.1.4. No movement > 5 mins + Light ON (LDR High) -> Buzzer ON
    if (!pirState && (now - lastPirMovement > 300000) && (ldrValue > 700)) {
        if (!buzzerState) {
            buzzerState = true;
            digitalWrite(BUZZER_PIN, HIGH);
            publishStatus();
            logData("Buzzer Alert (Light left ON)");
        }
    } else if (buzzerState && (pirState || ldrValue <= 700)) {
        buzzerState = false;
        digitalWrite(BUZZER_PIN, LOW);
        publishStatus();
        logData("Buzzer OFF");
    }
}

void reconnectMqtt() {
    static unsigned long lastReconnectAttempt = 0;
    if (millis() - lastReconnectAttempt > 5000) {
        lastReconnectAttempt = millis();

        // Cycle through ports
        mqttPort = mqttPorts[currentMqttPortIndex];
        currentMqttPortIndex = (currentMqttPortIndex + 1) % numMqttPorts;

        Serial.print("Attempting MQTT connection on port ");
        Serial.print(mqttPort);
        Serial.print("... ");

        mqttClient.setServer(mqttBroker.c_str(), mqttPort);

        if (mqttClient.connect(mqttClientId.c_str())) {
            Serial.println("connected");
            mqttClient.subscribe(cmdTopic.c_str());
            publishStatus();
        } else {
          Serial.print("failed, rc=");
          Serial.println(mqttClient.state());
        }
    }
}
