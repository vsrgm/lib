#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#include <ESP8266WebServer.h>
#include <ArduinoOTA.h>
#include <Firebase_ESP_Client.h>
#include "credentials.h"

const String SW_VERSION = "1.0.416";
WiFiUDP udpDiscovery;
const int UDP_DISCOVERY_PORT = 8888;

void setupUdpDiscovery() {
  udpDiscovery.begin(UDP_DISCOVERY_PORT);
}

void checkUdpDiscovery() {
  if (WiFi.status() != WL_CONNECTED) return;

  int packetSize = udpDiscovery.parsePacket();
  if (packetSize) {
    char buf[255];
    int len = udpDiscovery.read(buf, sizeof(buf) - 1);
    if (len > 0) buf[len] = 0;
    String msg = String(buf);
    msg.trim();
    if (msg == "DISCOVER" || msg.indexOf("DISCOVER") != -1) {
      IPAddress remoteIp = udpDiscovery.remoteIP();
      uint16_t remotePort = udpDiscovery.remotePort();
      if (remotePort == 0) remotePort = UDP_DISCOVERY_PORT;

      String response = "{\"ip\":\"" + WiFi.localIP().toString() + "\",\"id\":\"rccar\",\"name\":\"rccar\",\"ver\":\"" + SW_VERSION + "\"}";

      udpDiscovery.beginPacket(remoteIp, remotePort);
      udpDiscovery.print(response);
      udpDiscovery.endPacket();

      if (remotePort != UDP_DISCOVERY_PORT) {
        udpDiscovery.beginPacket(remoteIp, UDP_DISCOVERY_PORT);
        udpDiscovery.print(response);
        udpDiscovery.endPacket();
      }

      IPAddress broadcastIp = WiFi.broadcastIP();
      udpDiscovery.beginPacket(broadcastIp, UDP_DISCOVERY_PORT);
      udpDiscovery.print(response);
      udpDiscovery.endPacket();
    }
  }
}

int fl = D0;
int bl = D1;
int fr = D2;
int br = D3;

String command;    // String to store app command state.
ESP8266WebServer server(80);  // Create a webserver object that listens for HTTP request on port 80

// Firebase Data objects
FirebaseData fbdo;
FirebaseAuth fbAuth;
FirebaseConfig fbConfig;

unsigned long previousMillis = 0;

String sta_ssid = HOME_NETWORK_SSID;
String sta_password = HOME_NETWORK_PASSWORD;

int WIFI_CAR_MODE = WIFI_STA;

void handleFirebaseStream(FirebaseStream data) {
  if (data.dataPath() == "/command") {
    String cmd = data.stringData();
    Serial.println("Firebase Command Received: " + cmd);
    handleCarCommand(cmd);
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

void handleCarCommand(String cmd) {
  cmd.toUpperCase();
  if (cmd == "F" || cmd == "FORWARD") Forward();
  else if (cmd == "B" || cmd == "BACKWARD") Backward();
  else if (cmd == "R" || cmd == "RIGHT" || cmd == "TURNRIGHT") TurnRight();
  else if (cmd == "L" || cmd == "LEFT" || cmd == "TURNLEFT") TurnLeft();
  else if (cmd == "G" || cmd == "FORWARDLEFT") ForwardLeft();
  else if (cmd == "H" || cmd == "BACKWARDLEFT") BackwardLeft();
  else if (cmd == "I" || cmd == "FORWARDRIGHT") ForwardRight();
  else if (cmd == "J" || cmd == "BACKWARDRIGHT") BackwardRight();
  else if (cmd == "S" || cmd == "STOP") Stop();
  else if (cmd == "V" || cmd == "HORN" || cmd == "BEEP") BeepHorn();
  else if (cmd == "W" || cmd == "LIGHT_ON") TurnLightOn();
  else if (cmd == "w" || cmd == "LIGHT_OFF") TurnLightOff();
}

void setup() {
  pinMode(fl, OUTPUT);
  pinMode(fr, OUTPUT);
  pinMode(bl, OUTPUT);
  pinMode(br, OUTPUT);
  digitalWrite(fl, HIGH);
  digitalWrite(fr, HIGH);
  digitalWrite(bl, HIGH);
  digitalWrite(br, HIGH);

  Serial.begin(115200);  // set up Serial library at 115200 bps
  Serial.println();
  Serial.println("*WiFi Robot Remote Control Mode - L298N 2A*");
  Serial.println("------------------------------------------------");

  // set NodeMCU Wifi hostname based on chip mac address
  String chip_id = String(ESP.getChipId(), HEX);
  int i = chip_id.length() - 4;
  chip_id = chip_id.substring(i);
  chip_id = "WiFi_RC_Car-" + chip_id;
  String hostname(chip_id);

  Serial.println();
  Serial.println("Hostname: " + hostname);

  if (WIFI_CAR_MODE == WIFI_STA)
  {
    // first, set NodeMCU as STA mode to connect with a Wifi network
    WiFi.mode(WIFI_STA);
    WiFi.begin(sta_ssid.c_str(), sta_password.c_str());
    Serial.println("");
    Serial.print("Connecting to: ");
    Serial.println(sta_ssid);
    Serial.print("Password: ");
    Serial.println(sta_password);

    // try to connect with Wifi network about 10 seconds
    unsigned long currentMillis = millis();
    previousMillis = currentMillis;
    while (WiFi.status() != WL_CONNECTED && currentMillis - previousMillis <= 10000) {
      delay(500);
      Serial.print(".");
      currentMillis = millis();
    }
    Serial.println("");
    Serial.println("*WiFi-STA-Mode*");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  }
  else
  {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(hostname.c_str());
    IPAddress myIP = WiFi.softAPIP();
    Serial.println("");
    Serial.println("WiFi failed connected to " + sta_ssid);
    Serial.println("");
    Serial.println("*WiFi-AP-Mode*");
    Serial.print("AP IP address: ");
    Serial.println(myIP);
  }
  delay(300);

  server.on("/", HTTP_handleRoot);     // call the 'handleRoot' function when a client requests URI "/"
  server.onNotFound(HTTP_handleRoot);  // when a client requests an unknown URI (i.e. something other than "/"), call function "handleNotFound"
  server.begin();                      // actually start the server

  ArduinoOTA.onStart([]() {
    ESP.wdtEnable(20000);
  });
  ArduinoOTA.begin();  // enable to receive update/uploade firmware via Wifi OTA

  ESP.wdtEnable(WDTO_8S);
  setupFirebase();
  setupUdpDiscovery();
}

void loop() {
  ESP.wdtFeed();
  checkUdpDiscovery();
  ArduinoOTA.handle();    // listen for update OTA request from clients
  server.handleClient();  // listen for HTTP requests from clients
}

// function prototypes for HTTP handlers
void HTTP_handleRoot(void) {
  String response = "{\"ip\":\"" + WiFi.localIP().toString() + "\",\"id\":\"rccar\",\"name\":\"rccar\",\"ver\":\"" + SW_VERSION + "\"}";
  server.send(200, "application/json", response);

  String cmd = "";
  if (server.hasArg("State")) {
    cmd = server.arg("State");
  } else if (server.hasArg("cmd")) {
    cmd = server.arg("cmd");
  }
  if (cmd != "") {
    Serial.println("HTTP Command Received: " + cmd);
    handleCarCommand(cmd);
  }
}

void handleNotFound() {
  HTTP_handleRoot();
}

// function to move forward
void Forward() {
  digitalWrite(bl, HIGH);
  digitalWrite(br, HIGH);

  delay(100);
  digitalWrite(fl, LOW);
  digitalWrite(fr, LOW);

}

// function to move backward
void Backward() {
  digitalWrite(fl, HIGH);
  digitalWrite(fr, HIGH);
  delay(100);
  digitalWrite(bl, LOW);
  digitalWrite(br, LOW);
}

// function to turn right
void TurnRight() {
  digitalWrite(fr, HIGH);
  digitalWrite(fl, LOW);
  delay(100);
  digitalWrite(bl, HIGH);
  digitalWrite(br, LOW);
}

// function to turn left
void TurnLeft() {
  digitalWrite(fl, HIGH);
  digitalWrite(fr, LOW);
  delay(100);
  digitalWrite(bl, LOW);
  digitalWrite(br, HIGH);
}

// function to move forward left
void ForwardLeft() {

}

// function to move backward left
void BackwardLeft() {

}

// function to move forward right
void ForwardRight() {

}

// function to move backward right
void BackwardRight() {

}

// function to stop motors
void Stop() {
  digitalWrite(fl, HIGH);
  digitalWrite(fr, HIGH);
  digitalWrite(bl, HIGH);
  digitalWrite(br, HIGH);
}

// function to beep a buzzer
void BeepHorn() {

}

// function to turn on LED
void TurnLightOn() {
}

// function to turn off LED
void TurnLightOff() {
}
