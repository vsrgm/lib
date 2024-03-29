
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

// Set the LCD address to 0x27/0x3f for a 16 chars and 2 line display
LiquidCrystal_I2C lcd(0x27, 16, 2);
DHT dht;

const char* ssid = WLAN_SSID;
const char* password = WLAN_PASS;

void setup()
{
  ESP.wdtDisable();

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
  ArduinoOTA.setHostname("DisplayModule");

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

  // initialize the LCD
  lcd.begin();

  dht.setup(D3);   /* D1 is used for data communication */
  lcd.clear();
  mqtt.subscribe(&plant);
  watchdogfeed();
}

void watchdogfeed()
{
  ESP.wdtFeed();
  ESP.wdtDisable();
}

void loop()
{
  int ret, type, value1, value2, value3, value4, levelb, levelt;
  char publish[100];
  char instring[100];
  watchdogfeed();
  unsigned int gettemponce = 1;
  ret = MQTT_connect();
  if ((ret == 0) || !(Ping.ping("www.google.com")))
  {
    ESP.restart();
    lcd.noBacklight();

  } else
  {
    // Turn on the blacklight and print a message.
    lcd.backlight();
  }
  debug.publish("Display : reset");

  Adafruit_MQTT_Subscribe *subscription;
  while (1)
  {
    watchdogfeed();
    ArduinoOTA.handle();
    watchdogfeed();

    ret = MQTT_connect();
    if ((ret == 0) || !(Ping.ping("www.google.com")))
    {
      ESP.restart();
      lcd.noBacklight();

    } else
    {
      // Turn on the blacklight and print a message.
      lcd.backlight();
    }
    watchdogfeed();

    delay(dht.getMinimumSamplingPeriod()); /* Delay of amount equal to sampling period */
    watchdogfeed();
    int humidity = dht.getHumidity();/* Get humidity value */
    int temperature = dht.getTemperature();/* Get temperature value */
    watchdogfeed();

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
    watchdogfeed();
    debug.publish("Display : Loop");

    if (gettemponce)
    {
      memset(instring, 0x00, sizeof(instring));
      sprintf(instring, "4");
      status.publish(instring);
      gettemponce = 0;
    }

    while ((subscription = mqtt.readSubscription(5000)))
    {
      watchdogfeed();
      if (subscription == &plant)
      {
        char printformat[17];
        sscanf((char *)plant.lastread, "%d,%d,%d,%d,%d,%d,%d",
               &type, &value1, &value2, &value3, &value4,
               &levelb, &levelt);
        if (type == REPORT)
        {
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

          memset(publish, 0, sizeof(publish));
          sprintf(publish, "%d,H=%d%,T=%dC", AUTOUPDATETEMPHUMID, humidity, temperature);
          status.publish(publish);
        }
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

  uint8_t retries = 15;

  while ((ret = mqtt.connect()) != 0)
  { // connect will return 0 for connected
    Serial.println(mqtt.connectErrorString(ret));
    Serial.println("Retrying MQTT connection in 1 seconds...");
    mqtt.disconnect();
    delay(1000);  // wait 5 seconds
    watchdogfeed();
    retries--;
    if (retries == 0)
    {
      return 0;
    }
  }
  return 1;
}
