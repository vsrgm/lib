#include <ESP8266WiFi.h>
#include <ESP8266mDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <Wire.h>
#include <EEPROM.h>
#include <ESP8266WiFi.h>
#include <ESP8266Ping.h>
#include <ESP8266WebServer.h>

#include "Adafruit_MQTT.h"
#include "Adafruit_MQTT_Client.h"
#include "inc/key.h"
#include "inc/plants.h"

#define RELAY1		      D1
#define RELAY2		      D2
#define RELAY3		      D3
#define RELAY4		      D4
#define WATERLEVELTERRACE     D6
#define WATERLEVELBALCONY     D7
// define the number of bytes you want to access
#define EEPROM_SIZE 512
#define STRING_SIZE 1024

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
Adafruit_MQTT_Publish debug = Adafruit_MQTT_Publish(&mqtt, AIO_USERNAME "/feeds/debug");

unsigned char MQTT_connect();

int bcnt = 0, tcnt = 0, btotal = 0, ttotal = 0;

const char* ssid = WLAN_SSID;
const char* password = WLAN_PASS;

void setup()
{
  ESP.wdtDisable();

  int Magic, retry_count = 10;
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
  watchdogfeed();

  Serial.begin(115200);
  Serial.println("Booting");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.waitForConnectResult() != WL_CONNECTED) {
    Serial.println("Connection Failed! Rebooting...");
    delay(5000);
    ESP.restart();
  }
  watchdogfeed();

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

  watchdogfeed();

  // initialize EEPROM with predefined size
  EEPROM.begin(EEPROM_SIZE);
  while (retry_count)
  {
    watchdogfeed();

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
  watchdogfeed();

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

  watchdogfeed();

  // Setup MQTT subscription for onoff feed.
  mqtt.subscribe(&plant);
  watchdogfeed();
}

void watchdogfeed()
{
  ESP.wdtFeed();
  ESP.wdtDisable();
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

int countcomma(char *feedstring)
{
  int count = 0, inc = 0;
  for (inc = 0; feedstring[inc]; inc++)
  {
    if (feedstring[inc] == ',')
      count++;
  }

  return count;
}

void get_input_feeddata(char *feedstring, struct feedinput *feed)
{
  int feedlength;
  feedlength = strlen(feedstring);
  sscanf(feedstring, "%d", &feed->type);
  Serial.printf("%s %d \r\n", __FUNCTION__, __LINE__);

  switch (feed->type)
  {
    case REPORT:
    case GETREPORT:
      {
        break;
      }

    case BPLANTS:
    case TPLANTS:
    case RESETREPORT:
    case RESETDATA:
      {
        switch (countcomma(feedstring))
        {
          case 0:
            {
              Serial.printf("%s %d \r\n", __FUNCTION__, __LINE__);
              break;
            }
          case 4:
            {
              sscanf(feedstring, "%d,%d,%d,%d,%d", &feed->type, &feed->value1, &feed->value2, &feed->value3, &feed->value4);
              Serial.printf("%s %d \r\n", __FUNCTION__, __LINE__);
              break;
            }
          default :
            {
              Serial.printf("%s %d \r\n", __FUNCTION__, __LINE__);
              sscanf(feedstring, "%d,%d,%d,%d,%d,%s %d,%d at %d:%2d%2s",
                     &feed->type, &feed->value1, &feed->value2, &feed->value3, &feed->value4,
                     feed->date.month, &feed->date.date, &feed->date.year, &feed->date.timehour, &feed->date.timemin, feed->date.meridies);
              Serial.printf("%d %s \r\n", feed->date.timehour, feed->date.meridies);

            }
        }
      }
  }

  Serial.printf("%s %d \r\n", __FUNCTION__, __LINE__);
  return ;
}

int compute_water_pattern(struct feedinput *feed)
{
  if (strstr(feed->date.meridies, "PM"))
  {
    if (feed->date.timehour != 12)
      feed->date.timehour += 12;
  } else
  {
    if (feed->date.timehour == 12)
      feed->date.timehour = 0;
  }
  Serial.printf("%s %d \r\n", __FUNCTION__, __LINE__);
  return waterpattern(feed->value2, feed->date.timehour, (feed->date.timehour < 12), feed->value3, feed->value4);
}

void loop()
{
  int count = 0, ret;
  int balwaterlevellow = 0, terracewaterlevellow = 0;
  char instring[STRING_SIZE];
  char publish[STRING_SIZE];

  struct feedinput feed;

  int pingret = 1;
  Adafruit_MQTT_Subscribe *subscription;

  memset(&feed, 0x00, sizeof(feed));
  balwaterlevellow = getwaterlevel(WATERLEVELBALCONY);
  terracewaterlevellow = getwaterlevel(WATERLEVELTERRACE);

  while (1)
  {
    watchdogfeed();

    ArduinoOTA.handle();

    pingret = Ping.ping("www.google.com", 2);

    ret = MQTT_connect();
    if ((ret == 0) || !pingret)
    {
      ESP.restart();
    }

    watchdogfeed();
    debug.publish("Start");

    while ((subscription = mqtt.readSubscription(5000)))
    {
      watchdogfeed();

      if (subscription == &plant)
      {
        Serial.print(F("GotB: "));
        Serial.println((char *)plant.lastread);
        get_input_feeddata((char *)plant.lastread, &feed);

        switch (feed.type)
        {
          case BPLANTS:
            {
              balwaterlevellow = getwaterlevel(WATERLEVELBALCONY);
              if (compute_water_pattern(&feed))
                waterplants(feed.value1, RELAY3, RELAY4, balwaterlevellow, BCNT, BTOTAL);
            }
            break;

          case TPLANTS:
            {
              terracewaterlevellow = getwaterlevel(WATERLEVELTERRACE);

              if (compute_water_pattern(&feed))
                waterplants(feed.value1, RELAY1, RELAY2, terracewaterlevellow, TCNT, TTOTAL);
            }
            break;

          case RESETREPORT:
            {
              bcnt = feed.value1;
              btotal = feed.value2;
              tcnt = feed.value3;
              ttotal = feed.value4;
              resetEEROM(bcnt, btotal, tcnt, ttotal);
            }
            break;

          case RESETDATA:
            {
              bcnt = tcnt = 0;
              resetEEROM(bcnt, btotal, tcnt, ttotal);
            }
            break;

          case GETREPORT:
            {
              balwaterlevellow = getwaterlevel(WATERLEVELBALCONY);
              terracewaterlevellow = getwaterlevel(WATERLEVELTERRACE);
            }
            break;
        }

        watchdogfeed();

        switch (feed.type)
        {
          case RESETREPORT:
          case TPLANTS:
          case BPLANTS:
          case RESETDATA:
            EEPROM.commit();
          //break;

          case GETREPORT:
            {

              long millisecs;

              feed.type = REPORT;
              bcnt = EEPROM.read(BCNT);
              btotal = EEPROM.read(BTOTAL);
              tcnt = EEPROM.read(TCNT);
              ttotal = EEPROM.read(TTOTAL);
              memset(publish, 0, sizeof(publish));

              millisecs = millis();
              int systemUpTimeMn = int((millisecs / (1000 * 60)) % 60);
              int systemUpTimeHr = int((millisecs / (1000 * 60 * 60)) % 24);
              int systemUpTimeDy = int((millisecs / (1000 * 60 * 60 * 24)) % 365);


              sprintf(publish, "%d,%d,%d,%d,%d,%d,%d,V=%s-%s,%d,%d,%s,%s-%d-%d-%d:%d,%d-%d-%d",
                      feed.type, bcnt, btotal, tcnt, ttotal,
                      (balwaterlevellow < 50), (terracewaterlevellow < 50),
                      __DATE__, __TIME__, balwaterlevellow, terracewaterlevellow,
                      ESP.getResetReason().c_str(),
                      feed.date.month, feed.date.date, feed.date.year, feed.date.timehour, feed.date.timemin,
                      systemUpTimeDy, systemUpTimeHr, systemUpTimeMn);
              status.publish(publish);
              debug.publish("Got Data");
            }
            break;

          default:
            break;
        }
      }
    }
  }
}

bool waterplants(int timeperiod, int relay1, int relay2, int low, int cntoffset, int toffset)
{
  int cnt = 0;
  int total = 0;
  if (timeperiod && (low < 50))
  {
    delay(1000);

    digitalWrite(relay1, !ENABLE_HIGH);
    digitalWrite(relay2, !ENABLE_HIGH);

    for (int count = 0; count < timeperiod; count++)
    {
      watchdogfeed();
      delay(1000);
    }

    digitalWrite(relay1, ENABLE_HIGH);
    digitalWrite(relay2, ENABLE_HIGH);

    cnt = EEPROM.read(cntoffset);
    total = EEPROM.read(toffset);
    cnt++;


    EEPROM.write(cntoffset, cnt);
    if (cnt > total)
    {
      total = cnt;
      EEPROM.write(toffset, total);
    }
    delay(1000);


  } else
  {
    cnt = 0;
    EEPROM.write(cntoffset, cnt);
    return false;
  }

  return true;
}

bool waterpattern(int patterntype, int mhour, bool am, int hourp1, int hourp2)
{
  /* Pattern 1 : How many hours once in hourp1*/
  /* Pattern 2 : AM in hourp1 /PM in hourp2 */

  bool value = 1;
  switch (patterntype)
  {
    case 1:
      {
        value = ((mhour % hourp1) == 0) ? 1 : 0;
      }
      break;

    case 2:
      {
        value = ((mhour % (am ? hourp1 : hourp2)) == 0) ? 1 : 0;
      }
      break;

    case 3:
      {
        if ((mhour >= 6) && (mhour <= 18)) // 6AM to 6PM
          value = ((mhour % hourp2) == 0) ? 1 : 0;
        else
          value = ((mhour % hourp1) == 0) ? 1 : 0;
      }
      break;
  }
  return value;
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
