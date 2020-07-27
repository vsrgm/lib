//TODO : Implement WatchDog to safe reboot in case if any hang in the board
#include <ESP8266WiFi.h>
#include <ESP8266mDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <Wire.h>
#include <EEPROM.h>
#include <ESP8266WiFi.h>
#include "Adafruit_MQTT.h"
#include "Adafruit_MQTT_Client.h"
#include "key.h"

#define RELAY1		      D1
#define RELAY2		      D2
#define RELAY3		      D3
#define RELAY4		      D4
#define WATERLEVELTERRACE     D6
#define WATERLEVELBALCONY     D7
// define the number of bytes you want to access
#define EEPROM_SIZE 256

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
Adafruit_MQTT_Publish status = Adafruit_MQTT_Publish(&mqtt, AIO_USERNAME "/feeds/cmnplant");
unsigned char MQTT_connect();

int bcnt = 0, tcnt = 0, btotal = 0, ttotal = 0;

enum eeromoffset
{
  MAGIC = 1,
  BCNT,
  BTOTAL,
  TCNT,
  TTOTAL,
};

enum feeds
{
  BPLANTS = 0,
  TPLANTS,
  RESETREPORT,
  REPORT,
  GETREPORT,
};
const char* ssid = WLAN_SSID;
const char* password = WLAN_PASS;

void setup()
{
  int Magic, retry_count = 5;
  pinMode(RELAY1, OUTPUT);
  digitalWrite(RELAY1, ENABLE_HIGH);
  pinMode(RELAY2, OUTPUT);
  digitalWrite(RELAY2, ENABLE_HIGH);
  pinMode(RELAY3, OUTPUT);
  digitalWrite(RELAY3, ENABLE_HIGH);
  pinMode(RELAY4, OUTPUT);
  digitalWrite(RELAY4, ENABLE_HIGH);
  pinMode(WATERLEVELTERRACE, INPUT_PULLUP);
  pinMode(WATERLEVELBALCONY, INPUT_PULLUP);


  Serial.begin(115200);
  Serial.println("Booting");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.waitForConnectResult() != WL_CONNECTED) {
    Serial.println("Connection Failed! Rebooting...");
    delay(5000);
    ESP.restart();
  }

  // Port defaults to 8266
  // ArduinoOTA.setPort(8266);

  //Hostname defaults to esp8266-[ChipID]
  ArduinoOTA.setHostname("Waterpump");

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


  // initialize EEPROM with predefined size
  EEPROM.begin(EEPROM_SIZE);
  while (retry_count)
  {
    Magic = EEPROM.read(MAGIC);
    if (Magic == 0xAB)
      break;
    retry_count--;
    delay(1000);
  }

  if (Magic == 0xAB)
  {
    bcnt = EEPROM.read(BCNT);
    tcnt = EEPROM.read(TCNT);
    btotal = EEPROM.read(BTOTAL);
    ttotal = EEPROM.read(TTOTAL);
  } else
  {
    resetEEROM(0, 0, 0, 0);
  }

  // Connect to WiFi access point.
  Serial.println(); Serial.println();
  Serial.print("Connecting to ");
  Serial.println(WLAN_SSID);

  WiFi.begin(WLAN_SSID, WLAN_PASS);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());


  // Setup MQTT subscription for onoff feed.
  mqtt.subscribe(&plant);
}

void resetEEROM(int bcnt, int btotal, int tcnt, int ttotal)
{
  EEPROM.write(MAGIC, 0xAB);
  EEPROM.write(BCNT, bcnt);
  EEPROM.write(BTOTAL, btotal);
  EEPROM.write(TCNT, tcnt);
  EEPROM.write(TTOTAL, ttotal);
  EEPROM.commit();
}

unsigned int getwaterlevel(unsigned int pin)
{
  unsigned int count = 0, level = 0, idx;
  for (idx = 0; idx < 100; idx++)
  {
    level = digitalRead(pin);
    count += level;
    delay(10);
  }
  return count;
}

void loop()
{
  int count = 0, ret;
  int balwaterlevellow = 0, terracewaterlevellow = 0;
  char instring[100];
  char publish[100];
  int type, value1, value2, value3, value4;

  Adafruit_MQTT_Subscribe *subscription;

  ArduinoOTA.handle();
  ret = MQTT_connect();
  if (ret == 0)
  {
    ESP.restart();
  }

  while ((subscription = mqtt.readSubscription(5000)))
  {
    if (subscription == &plant)
    {
      Serial.print(F("GotB: "));
      Serial.println((char *)plant.lastread);
      sscanf((char *)plant.lastread, "%d,%d,%d,%d,%d", &type, &value1, &value2, &value3, &value4);
      balwaterlevellow = getwaterlevel(WATERLEVELBALCONY);
      terracewaterlevellow = getwaterlevel(WATERLEVELTERRACE);

      if (type == BPLANTS)
      {
        if (value1)// && (balwaterlevellow < 50))
        {
          digitalWrite(RELAY3, !ENABLE_HIGH);
          digitalWrite(RELAY4, !ENABLE_HIGH);

          for (count = 0; count < value1; count++)
            delay(1000);

          digitalWrite(RELAY3, ENABLE_HIGH);
          digitalWrite(RELAY4, ENABLE_HIGH);
          bcnt++;

          EEPROM.write(BCNT, bcnt);
          if (bcnt > btotal)
          {
            btotal = bcnt;
            EEPROM.write(BTOTAL, btotal);
          }
        } else
        {
          bcnt = 0;
          EEPROM.write(BCNT, bcnt);
        }
      } else if (type == TPLANTS)
      {
        if (value1 && (terracewaterlevellow < 50))
        {
          digitalWrite(RELAY1, !ENABLE_HIGH);
          digitalWrite(RELAY2, !ENABLE_HIGH);

          for (count = 0; count < value1; count++)
            delay(1000);

          digitalWrite(RELAY1, ENABLE_HIGH);
          digitalWrite(RELAY2, ENABLE_HIGH);

          tcnt++;
          EEPROM.write(TCNT, tcnt);
          if (tcnt > ttotal)
          {
            ttotal = tcnt;
            EEPROM.write(TTOTAL, ttotal);
          }
        } else
        {
          tcnt = 0;
          EEPROM.write(TCNT, tcnt);
        }
      } else if (type == RESETREPORT)
      {
        bcnt = value1;
        btotal = value2;
        tcnt = value3;
        ttotal = value4;
        resetEEROM(value1, value2, value3, value4);
      }

      switch (type)
      {
        case RESETREPORT:
        case TPLANTS:
        case BPLANTS:
          EEPROM.commit();
        //break;

        case GETREPORT:
          type = REPORT;
          bcnt = EEPROM.read(BCNT);
          btotal = EEPROM.read(BTOTAL);
          tcnt = EEPROM.read(TCNT);
          ttotal = EEPROM.read(TTOTAL);
          memset(publish, 0, sizeof(publish));
          sprintf(publish, "%d,%d,%d,%d,%d,%d,%d,V=%s-%s,%d,%d", type, bcnt, btotal, tcnt,
                  ttotal, (balwaterlevellow < 50), (terracewaterlevellow < 50),
                  __DATE__, __TIME__, balwaterlevellow, terracewaterlevellow);
          status.publish(publish);
          break;

        default:
          break;
      }
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
  Serial.println("MQTT Connected!");
  return 1;

}
