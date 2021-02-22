/*  HTTPS on ESP8266 with follow redirects, chunked encoding support
    Version 3.0
    Author: Sujay Phadke
    Github: @electronicsguy
    Copyright (C) 2018 Sujay Phadke <electronicsguy123@gmail.com>
    All rights reserved.

    Example Arduino program
*/

#include <ESP8266WiFi.h>
#include "HTTPSRedirect.h"
#include "DebugMacros.h"

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
String payload = "";

String url3 = String("/macros/s/") + GScriptId + "/exec?read";

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
}
