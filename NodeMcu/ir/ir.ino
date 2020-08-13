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

  dht.setup(D3);   /* D1 is used for data communication */
  lcd.clear();
  mqtt.subscribe(&mqttir);
  mqtt.subscribe(&plant);
}

void loop()
{
#define MAX_COUNT 100
  float avgTemp = 0, count = 0, avgV = 0;

  int ret, type, value1, value2, value3, value4, levelb, levelt, mqttcmd = 0;
  char publish[100];
  char instring[100];
  unsigned long irhash = 0, ircode, switchcmd, lcdon = 0, checknetconnectivity = MAX_COUNT, netconnectivity = 0;

  decode_results results;
  Adafruit_MQTT_Subscribe *subscription;

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
      irrecv.resume();  // Receive the next value
    }

    switchcmd = mqttcmd ? mqttcmd : irhash;
    Serial.print("FAN  = ");
    Serial.println(switchcmd);

    switch (switchcmd)
    {
      case 0x01: //AC OFF
        {
          //Hash:
          //E809B3D2
          //For IR Scope/IrScrutinizer:
          //+83150 -34350 +11800 -28050 +11800 -28050 +11800 -10550 +11800 -10550 +11800 -10650 +11800 -28100 +11800 -10550 +11800 -10500 +12450 -27550 +11800 -28
          //050 +11850 -10550 +12400 -27500 +11800 -10500 +11800 -10500 +11800 -28050 +11800 -28050 +11800 -10550 +11800 -28000 +11850 -28000 +11800 -10500 +11800
          // -10500 +11800 -28000 +11800 -10500 +11800 -10550 +11800 -28000 +11800 -10500 +11850 -10500 +11800 -10550 +11800 -10500 +11800 -10500 +12350 -10000 +1
          //1800 -10500 +11850 -10500 +11800 -10500 +11800 -10500 +11850 -10500 +11800 -10500 +11850 -10450 +11800 -10500 +11800 -10500 +11850 -28000 +11800 -2800
          //0 +11800 -10500 +11800 -10550 +11800 -10550 +11800 -28000 +11800 -10550 +11750 -10600 +11800 -127976
          //For Arduino sketch:
          uint16_t raw[100] = {
            83150, 34350, 11800, 28050, 11800, 28050, 11800, 10550, 11800, 10550, 11800, 10650, 11800, 28100, 11800, 10550, 11800, 10500, 12450, 27550, 11800,
            28050, 11850, 10550, 12400, 27500, 11800, 10500, 11800, 10500, 11800, 28050, 11800, 28050, 11800, 10550, 11800, 28000, 11850, 28000, 11800, 10500,
            11800, 10500, 11800, 28000, 11800, 10500, 11800, 10550, 11800, 28000, 11800, 10500, 11850, 10500, 11800, 10550, 11800, 10500, 11800, 10500, 12350,
            10000, 11800, 10500, 11850, 10500, 11800, 10500, 11800, 10500, 11850, 10500, 11800, 10500, 11850, 10450, 11800, 10500, 11800, 10500, 11850, 28000,
            11800, 28000, 11800, 10500, 11800, 10550, 11800, 10550, 11800, 28000, 11800, 10550, 11750, 10600, 11800
          };
          irsend.sendRaw(raw, 100, 38);
        }
        break;

      case 0x02: //AC ON
        {
          //Hash:
          //911F7E46
          //For IR Scope/IrScrutinizer:
          //+84550 -33300 +13100 -27050 +13100 -27000 +12500 -10100 +13100 -9500 +12500 -10100 +13100 -27000 +13100 -9500 +13100 -9500 +12500 -27600 +12500 -27600
          // +12500 -10100 +12500 -27600 +12500 -10050 +12450 -10050 +12450 -27550 +12450 -27500 +12450 -10050 +12450 -27550 +12450 -27450 +12450 -9900 +12450 -99
          //50 +12450 -27450 +12450 -9950 +12450 -9950 +12400 -27450 +12450 -10150 +13100 -9500 +12450 -10050 +12450 -9950 +12400 -9950 +12450 -9900 +12450 -9950
          //+12450 -9900 +12450 -9950 +12450 -9900 +12450 -9900 +12450 -10000 +13100 -9450 +12450 -10000 +12450 -10000 +12450 -27500 +12450 -27500 +12450 -27500 +
          //12450 -10000 +12450 -10000 +12450 -27500 +12450 -10000 +12450 -10000 +12450 -127976
          //For Arduino sketch:
          uint16_t raw[100] = {
            84550, 33300, 13100, 27050, 13100, 27000, 12500, 10100, 13100, 9500, 12500, 10100, 13100, 27000, 13100, 9500, 13100, 9500,
            12500, 27600, 12500, 27600, 12500, 10100, 12500, 27600, 12500, 10050, 12450, 10050, 12450, 27550, 12450, 27500, 12450,
            10050, 12450, 27550, 12450, 27450, 12450, 9900, 12450, 9950, 12450, 27450, 12450, 9950, 12450, 9950, 12400, 27450,
            12450, 10150, 13100, 9500, 12450, 10050, 12450, 9950, 12400, 9950, 12450, 9900, 12450, 9950, 12450, 9900, 12450, 9950,
            12450, 9900, 12450, 9900, 12450, 10000, 13100, 9450, 12450, 10000, 12450, 10000, 12450, 27500, 12450, 27500, 12450, 27500,
            12450, 10000, 12450, 10000, 12450, 27500, 12450, 10000, 12450, 10000, 12450
          };
          irsend.sendRaw(raw, 100, 38);
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
          //+472 -384 +898 -781 +473 -383 +883 -397 +448 -806 +897 -383 +461 -792 +461 -384 + 883 - 397 + 462 - 127976
          //For Arduino sketch:
          uint16_t raw[20] = {472, 384, 898, 781, 473, 383, 883, 397, 448, 806, 897, 383, 461, 792, 461, 384, 883, 397, 462,};
          irsend.sendRaw(raw, 20, 38);

          Serial.println("FAN ON");

        }
        break;

      case 0x05: // FAN OFF
      case 0xD546E1D2:
        {
          //D546E1D2
          //For IR Scope/IrScrutinizer:
          //+458 -397 +448 -397 +447 -397 +447 -398 +883 -397 +447 -798 +892 -397 +448 -797 +457 -398 +883 -397 +447 -127976
          //For Arduino sketch:
          uint16_t raw[22] = {458, 397, 448, 397, 447, 397, 447, 398, 883, 397, 447, 798, 892, 397, 448, 797, 457, 398, 883, 397, 447,};
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
#elif 0
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

#define USECPERTICK 1 //50
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
