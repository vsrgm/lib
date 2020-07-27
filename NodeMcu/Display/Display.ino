
#include <ESP8266WiFi.h>
#include <ESP8266mDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "DHT.h"
#include "key.h"
#include "Adafruit_MQTT.h"
#include "Adafruit_MQTT_Client.h"
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

// Set the LCD address to 0x27/0x3f for a 16 chars and 2 line display
LiquidCrystal_I2C lcd(0x27, 16, 2);
DHT dht;

const char* ssid = STASSID;
const char* password = STAPSK;

void setup()
{
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


  // initialize the LCD
  lcd.begin();

  // Turn on the blacklight and print a message.
  lcd.backlight();

  dht.setup(D3);   /* D1 is used for data communication */
  lcd.clear();
  mqtt.subscribe(&plant);

}


void loop() {
  int oldhumidity, oldtemperature;

  while (1)
  {
    enum feeds
    {
      BPLANTS = 0,
      TPLANTS,
      RESETREPORT,
      REPORT,
      GETREPORT,
      AUTOUPDATETEMPHUMID
    };

    int ret, type, value1, value2, value3, value4, levelb, levelt;
    char publish[100];
    char instring[100];

    Adafruit_MQTT_Subscribe *subscription;

    ArduinoOTA.handle();
    ret = MQTT_connect();
    if (ret == 0)
    {
      ESP.restart();
    }

    delay(dht.getMinimumSamplingPeriod()); /* Delay of amount equal to sampling period */
    int humidity = dht.getHumidity();/* Get humidity value */
    int temperature = dht.getTemperature();/* Get temperature value */

    if (!strncmp(dht.getStatusString(), "OK", 2))
    {
      if ((oldhumidity != humidity) || (oldtemperature != temperature))
      {
        memset(publish, 0, sizeof(publish));
        sprintf(publish, "%d,H=%d%,T=%dC", AUTOUPDATETEMPHUMID, humidity, temperature);
        status.publish(publish);
        oldhumidity = humidity;
        oldtemperature = temperature;
      }

      lcd.setCursor(0, 0); // Cursor0 , Linea0
      lcd.print("H:");
      lcd.print(humidity);
      lcd.print("% ");

      //lcd.setCursor(0, 1); // Cursor0 , Linea0
      lcd.print("T:");
      lcd.print(temperature);
      lcd.print("C");
    }

    while ((subscription = mqtt.readSubscription(5000)))
    {
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
