#include <ESP8266WiFi.h>
#include "HTTPSRedirect.h"
#include "DebugMacros.h"
#include <NTPClient.h>
#include <WiFiUdp.h>

// Fill ssid and password with your network credentials
const char* ssid = "ANANTH";
const char* password = "1q2w3e4r";

const char* host = "script.google.com";
// Replace with your own script id to make server side changes
const char *GScriptId = "AKfycbwQZ-bSbgwng3KwkbMQsv7vUXKiWFLEeSqHvI5YkudF0wWflqI";

const int httpsPort = 443;
HTTPSRedirect* client = nullptr;
String url = String("/macros/s/") + GScriptId + "/exec?cal";

String payload_base =  "{\"command\": \"cell\", \
			\"sheet_name\": \"Temp1\", \
			\"values\": ";

String payload_read =  "{\"command\": \"read\", \
			\"sheet_name\": \"Temp1\", \
			\"values\": ";
String payload = "";

String url3 = String("/macros/s/") + GScriptId + "/exec?read";


const long utcOffsetInSeconds = 19800;
char daysOfTheWeek[7][12] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};

// Define NTP Client to get time
WiFiUDP ntpUDP;

// You can specify the time server pool and the offset (in seconds, can be
// changed later with setTimeOffset() ). Additionaly you can specify the
// update interval (in milliseconds, can be changed using setUpdateInterval() ).
NTPClient timeClient(ntpUDP, "europe.pool.ntp.org", utcOffsetInSeconds, 60000);
void setup()
{
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  // Use HTTPSRedirect class to create a new TLS connection
  client = new HTTPSRedirect(httpsPort);
  client->setInsecure();
  client->setPrintResponseBody(true);
  client->setContentTypeHeader("application/json");
  timeClient.begin();
}

void loop()
{
  int retval = client->connect(host, httpsPort);
  payload = payload_base + "\"" + "A1" + "," + "Ananth" + "\"}";
  client->POST(url, host, payload, false);

  payload = payload_base + "\"" + "B1" + "," + "Family" + "\"}";
  client->POST(url, host, payload, false);

  payload = payload_base + "\"" + "C1" + "," + "Fine" + "\"}";
  client->POST(url, host, payload, false);


  /* read sequence */
#if 1
  payload = payload_read + "\"" + "A2" + "\"}";
  client->POST(url, host, payload, false);
#endif
  client->GET(url3, host, false);
  String sample = client->getResponseBody();

  timeClient.update();


  Serial.print(daysOfTheWeek[timeClient.getDay()]);
  Serial.print(", ");
  Serial.print(timeClient.getHours());
  Serial.print(":");
  Serial.print(timeClient.getMinutes());
  Serial.print(":");
  Serial.println(timeClient.getSeconds());
  
  Serial.print("Received text = ");
  Serial.print(sample);
  Serial.print("\r\n");
}
