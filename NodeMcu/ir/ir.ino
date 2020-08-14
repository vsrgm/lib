#ifndef UNIT_TEST
#include <Arduino.h>
#endif
#include <IRremoteESP8266.h>
#include <IRsend.h>
#include <IRrecv.h>
#include <ESP8266WiFi.h>
#include <ESP8266mDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ESP8266Ping.h>

#include "DHT.h"
#include "Adafruit_MQTT.h"
#include "Adafruit_MQTT_Client.h"

#include "inc/key.h"
#include "inc/plants.h"

IRsend irsend(14); // An IR LED is controlled by GPIO pin 14 (D5)
uint16_t RECV_PIN = D4;
const int sensorPin = A0;
IRrecv irrecv(RECV_PIN);

/************ Global State (you don't need to change this!) ******************/

// Create an ESP8266 WiFiClient class to connect to the MQTT server.
WiFiClient client;
// or... use WiFiFlientSecure for SSL
//WiFiClientSecure client;

// Setup the MQTT client class by passing in the WiFi client and MQTT server and login details.
Adafruit_MQTT_Client mqtt(&client, AIO_SERVER, AIO_SERVERPORT, AIO_USERNAME, AIO_KEY);

/****************************** Feeds ***************************************/


// Setup a feed called 'onoff' for subscribing to changes.
Adafruit_MQTT_Subscribe plant = Adafruit_MQTT_Subscribe(&mqtt, AIO_USERNAME"/feeds/cmnplant"); // FeedName
Adafruit_MQTT_Subscribe mqttir = Adafruit_MQTT_Subscribe(&mqtt, AIO_USERNAME"/feeds/irdisp"); // FeedName
Adafruit_MQTT_Publish status = Adafruit_MQTT_Publish(&mqtt, AIO_USERNAME "/feeds/irdisp");

// Set the LCD address to 0x27/0x3f for a 16 chars and 2 line display
LiquidCrystal_I2C lcd(0x3f, 16, 2);
DHT dht;

const char* ssid = WLAN_SSID;
const char* password = WLAN_PASS;

void setup()
{
  // initialize IR
  irsend.begin();
  irrecv.enableIRIn();  // Start the receiver
  //pinMode(sensorPin, INPUT);


  Serial.begin(115200);
  Serial.println("Booting");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.waitForConnectResult() != WL_CONNECTED)
  {
    Serial.println("Connection Failed! Rebooting...");
    delay(5000);
    ESP.restart();
  }

  // Port defaults to 8266
  // ArduinoOTA.setPort(8266);

  //Hostname defaults to esp8266-[ChipID]
  ArduinoOTA.setHostname("ACFANMonitor");

  // No authentication by default
  // ArduinoOTA.setPassword("admin");

  // Password can be set with it's md5 value as well
  // MD5(admin) = 21232f297a57a5a743894a0e4a801fc3
  // ArduinoOTA.setPasswordHash("21232f297a57a5a743894a0e4a801fc3");

  ArduinoOTA.onStart([]() {
    String type;
    if (ArduinoOTA.getCommand() == U_FLASH) {
      type = "sketch";
    } else { // U_FS
      type = "filesystem";
    }

    // NOTE: if updating FS this would be the place to unmount FS using FS.end()
    Serial.println("Start updating " + type);
  });
  ArduinoOTA.onEnd([]() {
    Serial.println("\nEnd");
  });
  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
  });
  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("Error[%u]: ", error);
    if (error == OTA_AUTH_ERROR) {
      Serial.println("Auth Failed");
    } else if (error == OTA_BEGIN_ERROR) {
      Serial.println("Begin Failed");
    } else if (error == OTA_CONNECT_ERROR) {
      Serial.println("Connect Failed");
    } else if (error == OTA_RECEIVE_ERROR) {
      Serial.println("Receive Failed");
    } else if (error == OTA_END_ERROR) {
      Serial.println("End Failed");
    }
  });
  ArduinoOTA.begin();
  Serial.println("Ready");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // initialize the LCD
  lcd.begin();
  lcd.clear();
  lcd.noBacklight();

  dht.setup(D3);   /* D1 is used for data communication */
  mqtt.subscribe(&mqttir);
  mqtt.subscribe(&plant);
}

void loop()
{
#define MAX_COUNT 100
  float avgTemp = 0, count = 0, avgV = 0, LcdTemp = 0;

  int ret, type, value1, value2, value3, value4, levelb, levelt, mqttcmd = 0;
  char publish[100];
  char instring[100];
  unsigned long irhash = 0, ircode, switchcmd, lcdon = 0, checknetconnectivity = MAX_COUNT, netconnectivity = 0;

  decode_results results;
  Adafruit_MQTT_Subscribe *subscription;
  decode_results *presults = &results;
  while (1)
  {
    ArduinoOTA.handle();
    ret = MQTT_connect();
    if (checknetconnectivity == 0)
    {
      netconnectivity = !(Ping.ping("www.google.com"));
      checknetconnectivity = MAX_COUNT;
    } else
    {
      checknetconnectivity--;
    }

    if ((ret == 0) || netconnectivity)
    {
      ESP.restart();
      lcd.noBacklight();
    }

    if (irrecv.decode(&results))
    {
      irhash = dump(&results);
      ircode = results.value;
      Serial.println(ircode);
      //irrecv.resume();  // Receive the next value
    }

    switchcmd = mqttcmd ? mqttcmd : irhash;
    switch (switchcmd)
    {
      case 0x01: //AC OFF
        {
          // Hash:
          // 3790E56F
          // For IR Scope/IrScrutinizer:
          // +3274 -1406 +466 -1134 +468 -1132 +466 -434 +466 -438 +468 -434 +472 -1130 +472 -430 +474 -430 +472 -1132 +496 -1106 +498 -406 +472 -1132 +472 -432 +496 -434 +500 -1082 +472 -1132 +474 -432 +496 -1106 +466 -1136 +468 -432 +442 -456 +464 -1134 +444 -456 +464 -434 +464 -1132 +466 -434 +462 -434 +464 -434 +468 -434 +520 -382 +474 -432 +472 -430 +442 -456 +464 -436 +442 -456 +464 -434 +464 -434 +466 -434 +442 -454 +466 -432 +498 -1104 +466 -1132 +466 -434 +464 -434 +466 -434 +466 -1134 +464 -434 +464 -434 +470 -1130 +464 -1134 +464 -434 +464 -434 +442 -454 +464 -436 +466 -432 +464 -434 +442 -1154 +442 -1154 +442 -1156 +462 -1134 +442 -454 +442 -456 +442 -454 +444 -454 +442 -456 +442 -456 +464 -436 +442 -1154 +442 -1154 +442 -1156 +468 -430 +442 -452 +442 -458 +464 -432 +476 -430 +442 -454 +442 -456 +462 -434 +442 -454 +442 -454 +442 -454 +442 -456 +442 -454 +442 -456 +442 -454 +442 -456 +468 -432 +442 -454 +442 -454 +442 -454 +442 -456 +442 -454 +464 -434 +494 -408 +464 -434 +442 -454 +442 -454 +442 -454 +442 -456 +442 -454 +442 -1154 +442 -454 +442 -456 +468 -432 +442 -454 +442 -1154 +464 -434 +442 -456 +464 -1132 +494 -406 +442 -454 +444 -1154 +442 -127976
          // For Arduino sketch:
          uint16_t raw[228] = {3274, 1406, 466, 1134, 468, 1132, 466, 434, 466, 438, 468, 434, 472, 1130, 472, 430, 474, 430, 472, 1132, 496, 1106, 498, 406, 472, 1132, 472, 432, 496, 434, 500, 1082, 472, 1132, 474, 432, 496, 1106, 466, 1136, 468, 432, 442, 456, 464, 1134, 444, 456, 464, 434, 464, 1132, 466, 434, 462, 434, 464, 434, 468, 434, 520, 382, 474, 432, 472, 430, 442, 456, 464, 436, 442, 456, 464, 434, 464, 434, 466, 434, 442, 454, 466, 432, 498, 1104, 466, 1132, 466, 434, 464, 434, 466, 434, 466, 1134, 464, 434, 464, 434, 470, 1130, 464, 1134, 464, 434, 464, 434, 442, 454, 464, 436, 466, 432, 464, 434, 442, 1154, 442, 1154, 442, 1156, 462, 1134, 442, 454, 442, 456, 442, 454, 444, 454, 442, 456, 442, 456, 464, 436, 442, 1154, 442, 1154, 442, 1156, 468, 430, 442, 452, 442, 458, 464, 432, 476, 430, 442, 454, 442, 456, 462, 434, 442, 454, 442, 454, 442, 454, 442, 456, 442, 454, 442, 456, 442, 454, 442, 456, 468, 432, 442, 454, 442, 454, 442, 454, 442, 456, 442, 454, 464, 434, 494, 408, 464, 434, 442, 454, 442, 454, 442, 454, 442, 456, 442, 454, 442, 1154, 442, 454, 442, 456, 468, 432, 442, 454, 442, 1154, 464, 434, 442, 456, 464, 1132, 494, 406, 442, 454, 444, 1154, 442,};
          irsend.sendRaw(raw, 228, 38);

        }
        break;

      case 0x02: //AC ON
        {
          // Hash:
          // 9ADEA691
          // For IR Scope/IrScrutinizer:
          // +3302 -1382 +472 -1130 +472 -1130 +472 -434 +494 -408 +470 -432 +472 -1130 +474 -428 +474 -430 +472 -1130 +474 -1130 +472 -430 +472 -1130 +472 -432 +474 -432 +470 -1132 +472 -1132 +474 -430 +472 -1132 +496 -1110 +474 -432 +472 -432 +472 -1130 +468 -432 +466 -432 +474 -1134 +464 -434 +444 -454 +466 -434 +442 -454 +464 -434 +466 -436 +468 -460 +446 -438 +464 -436 +462 -436 +470 -430 +442 -456 +442 -454 +444 -452 +466 -434 +444 -1154 +468 -1132 +444 -1154 +444 -454 +442 -456 +442 -1158 +442 -454 +466 -434 +442 -1154 +444 -1154 +444 -456 +442 -458 +472 -430 +442 -454 +444 -454 +444 -456 +444 -454 +464 -1134 +442 -1154 +444 -452 +444 -454 +444 -456 +442 -456 +442 -454 +444 -1152 +444 -454 +444 -1156 +442 -1156 +442 -1154 +444 -1158 +442 -456 +442 -454 +442 -456 +442 -454 +444 -452 +444 -456 +468 -432 +444 -454 +444 -454 +442 -456 +444 -454 +442 -454 +442 -456 +442 -456 +442 -454 +442 -456 +442 -454 +466 -432 +444 -454 +442 -454 +444 -454 +464 -434 +442 -456 +442 -454 +442 -454 +474 -430 +442 -456 +442 -456 +466 -432 +444 -456 +442 -1154 +442 -454 +442 -454 +444 -454 +442 -456 +442 -1154 +442 -456 +444 -454 +442 -1154 +442 -454 +442 -456 +466 -1134 +442 -127976
          // For Arduino sketch:
          uint16_t raw[228] = {3302, 1382, 472, 1130, 472, 1130, 472, 434, 494, 408, 470, 432, 472, 1130, 474, 428, 474, 430, 472, 1130, 474, 1130, 472, 430, 472, 1130, 472, 432, 474, 432, 470, 1132, 472, 1132, 474, 430, 472, 1132, 496, 1110, 474, 432, 472, 432, 472, 1130, 468, 432, 466, 432, 474, 1134, 464, 434, 444, 454, 466, 434, 442, 454, 464, 434, 466, 436, 468, 460, 446, 438, 464, 436, 462, 436, 470, 430, 442, 456, 442, 454, 444, 452, 466, 434, 444, 1154, 468, 1132, 444, 1154, 444, 454, 442, 456, 442, 1158, 442, 454, 466, 434, 442, 1154, 444, 1154, 444, 456, 442, 458, 472, 430, 442, 454, 444, 454, 444, 456, 444, 454, 464, 1134, 442, 1154, 444, 452, 444, 454, 444, 456, 442, 456, 442, 454, 444, 1152, 444, 454, 444, 1156, 442, 1156, 442, 1154, 444, 1158, 442, 456, 442, 454, 442, 456, 442, 454, 444, 452, 444, 456, 468, 432, 444, 454, 444, 454, 442, 456, 444, 454, 442, 454, 442, 456, 442, 456, 442, 454, 442, 456, 442, 454, 466, 432, 444, 454, 442, 454, 444, 454, 464, 434, 442, 456, 442, 454, 442, 454, 474, 430, 442, 456, 442, 456, 466, 432, 444, 456, 442, 1154, 442, 454, 442, 454, 444, 454, 442, 456, 442, 1154, 442, 456, 444, 454, 442, 1154, 442, 454, 442, 456, 466, 1134, 442,};
          irsend.sendRaw(raw, 228, 38);

        }
        break;

      case 0x03: //AC Set Temperature to 26
        {

        }
        break;

      case 0x04: //FAN ON
      case 0xD7CB28C0:
        {
          //D7CB28C0
          //For IR Scope/IrScrutinizer:
          //+868 -824 +1740 -1638 +866 -822 +1738 -824 +866 -1642 +1738 -820 +868 -1640 +866 -824 +1738 -822 +868 -127976
          //For Arduino sketch:
          uint16_t raw[20] = {868, 824, 1740, 1638, 866, 822, 1738, 824, 866, 1642, 1738, 820, 868, 1640, 866, 824, 1738, 822, 868,};
          irsend.sendRaw(raw, 20, 38);

          Serial.println("FAN ON");

        }
        break;

      case 0x05: // FAN OFF
      case 0xD546E1D2:
        {
          //D546E1D2
          //For IR Scope/IrScrutinizer:
          //+868 -824 +866 -824 +864 -824 +866 -824 +1736 -824 +866 -1644 +1734 -824 +866 -1642 +866 -824 +1736 -824 +864 -127976
          //For Arduino sketch:
          uint16_t raw[22] = {868, 824, 866, 824, 864, 824, 866, 824, 1736, 824, 866, 1644, 1734, 824, 866, 1642, 866, 824, 1736, 824, 864,};
          irsend.sendRaw(raw, 22, 38);

          Serial.println("FAN OFF");
        }
        break;

      default:
        {

        }
        break;
    }
    mqttcmd = irhash = 0;
    irrecv.resume();
#if 0
    delay(dht.getMinimumSamplingPeriod()); /* Delay of amount equal to sampling period */
    int humidity = dht.getHumidity();/* Get humidity value */
    int temperature = dht.getTemperature();/* Get temperature value */

    if (!strncmp(dht.getStatusString(), "OK", 2))
    {
      lcd.setCursor(0, 0); // Cursor0 , Linea0
      lcd.print("H:");
      lcd.print(humidity);
      lcd.print("% ");

      //lcd.setCursor(0, 1); // Cursor0 , Linea0
      lcd.print("T:");
      lcd.print(temperature);
      lcd.print("C");
    }
#elif 1
    delay(100);

    float sensorValue;
    float voltageOut;

    float temperatureC;
    float temperatureF;
    float temperatureK;

    sensorValue = analogRead(sensorPin);
    voltageOut = ((sensorValue * 3280) / 1024) * 0.958;

    // calculate temperature for LM335
    temperatureK = voltageOut / 10;
    temperatureC = temperatureK - 273;
    temperatureF = (temperatureC * 1.8) + 32;

    avgTemp += temperatureC;
    avgV += voltageOut;
    count++;

    if (count == MAX_COUNT)
    {
      Serial.print("Temperature in C = ");
      Serial.print(avgTemp / MAX_COUNT);
      Serial.print(" = ");
      Serial.println(avgV / MAX_COUNT);
      LcdTemp = avgTemp / MAX_COUNT;
      count = avgTemp = avgV = 0;
    }
#endif
    while ((subscription = mqtt.readSubscription(100)))
    {
      if (subscription == &plant)
      {
        char printformat[17];
        sscanf((char *)plant.lastread, "%d,%d,%d,%d,%d,%d,%d",
               &type, &value1, &value2, &value3, &value4,
               &levelb, &levelt);
        if (type == REPORT)
        {
          lcd.clear();
          lcd.setCursor(0, 0);
          lcd.print("T:");
          lcd.print(LcdTemp);
          lcd.print("C");
          lcd.setCursor(12, 0);
          lcd.print("B");
          lcd.print(levelb);
          lcd.print("T");
          lcd.print(levelt);

          lcd.setCursor(0, 1);
          lcd.print("B");
          sprintf(printformat, "%02d", value1);
          lcd.print(printformat);
          lcd.print("/");
          sprintf(printformat, "%02d", value2);
          lcd.print(printformat);
          lcd.print(" ");
          lcd.print("T");
          sprintf(printformat, "%02d", value3);
          lcd.print(printformat);
          lcd.print("/");
          sprintf(printformat, "%02d   ", value4);
          lcd.print(printformat);
          lcdon = 100;
        }
      } else if (subscription == &mqttir)
      {
        sscanf((char *)mqttir.lastread, "%d", &mqttcmd);
        lcdon = 100;
        lcd.clear();
        lcd.setCursor(0, 1);
        lcd.print("Value = ");
        lcd.print(mqttcmd);
      }
    }
    if (lcdon)
    {
      lcdon--;
      lcd.backlight();

    } else
    {
      lcd.noBacklight();
    }

  }
}


unsigned char MQTT_connect()
{
  int8_t ret;

  // Stop if already connected.
  if (mqtt.connected())
  {
    return 1;
  }

  Serial.print("Connecting to MQTT... ");

  uint8_t retries = 3;

  while ((ret = mqtt.connect()) != 0)
  { // connect will return 0 for connected
    Serial.println(mqtt.connectErrorString(ret));
    Serial.println("Retrying MQTT connection in 5 seconds...");
    mqtt.disconnect();
    delay(5000);  // wait 5 seconds
    retries--;
    if (retries == 0)
    {
      return 0;
    }
  }
  return 1;
}

int compare(unsigned int oldval, unsigned int newval) {
  if (newval < oldval * .8) {
    return 0;
  }
  else if (oldval < newval * .8) {
    return 2;
  }
  else {
    return 1;
  }
}

#define USECPERTICK 2 //50
#define FNV_PRIME_32 16777619
#define FNV_BASIS_32 2166136261

unsigned long decodeHash(decode_results * results) {
  unsigned long hash = FNV_BASIS_32;
  for (int i = 1; i + 2 < results->rawlen; i++) {
    int value =  compare(results->rawbuf[i], results->rawbuf[i + 2]);
    // Add value into the hash
    hash = (hash * FNV_PRIME_32) ^ value;
  }
  return hash;
}

int c = 1;
unsigned long dump(decode_results *results)
{
  int count = results->rawlen;
  uint16 lbuf[200];
  Serial.println(c);
  c++;
  Serial.println("Hash: ");
  unsigned long hash = decodeHash(results);
  Serial.println(hash, HEX);
  Serial.println("For IR Scope/IrScrutinizer: ");
  for (int i = 1; i < count; i++) {

    if ((i % 2) == 1) {
      Serial.print("+");
      Serial.print(results->rawbuf[i]*USECPERTICK, DEC);
    }
    else {
      Serial.print(-(int)results->rawbuf[i]*USECPERTICK, DEC);
    }
    Serial.print(" ");
  }
  Serial.println("-127976");
  Serial.println("For Arduino sketch: ");
  Serial.print("unsigned int raw[");
  Serial.print(count, DEC);
  Serial.print("] = {");
  for (int i = 1; i < count; i++) {

    if ((i % 2) == 1) {
      Serial.print(results->rawbuf[i]*USECPERTICK, DEC);
    }
    else {
      Serial.print((int)results->rawbuf[i]*USECPERTICK, DEC);
    }
    Serial.print(",");
  }
  Serial.print("};");
  Serial.println("");
  Serial.print("irsend.sendRaw(raw,");
  Serial.print(count, DEC);
  Serial.print(",38);");
  Serial.println("");
  Serial.println("");
  return hash;
}
