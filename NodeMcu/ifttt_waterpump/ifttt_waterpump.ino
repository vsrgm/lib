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
#define WATERLEVELTERRACE     D5
#define WATERLEVELBALCONY     D6
// define the number of bytes you want to access
#define EEPROM_SIZE 1

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
	MAGIC,
	BCNT,
	TCNT,
	BTOTAL,
	TTOTAL,
};

enum feeds
{
	BPLANTS = 0,
	TPLANTS,
	RESETREPORT,
	REPORT,
  GETREPORT,
  AUTOPUBLISH,
};
const char* ssid = WLAN_SSID;
const char* password = WLAN_PASS;

void setup()
{
	int Magic;
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
	Magic = EEPROM.read(MAGIC);
	if (Magic == 0xAB)
	{
		bcnt = EEPROM.read(BCNT);
		tcnt = EEPROM.read(TCNT);
		btotal = EEPROM.read(BTOTAL);
		ttotal = EEPROM.read(TTOTAL);
	}else
	{
		resetEEROM(0,0,0,0);
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

void resetEEROM(int bcnt, int tcnt, int btotal, int ttotal)
{
	EEPROM.write(MAGIC, 0xAB);
	EEPROM.write(BCNT, bcnt);
	EEPROM.write(TCNT, tcnt);
	EEPROM.write(BTOTAL, btotal);
	EEPROM.write(TTOTAL, ttotal);
	EEPROM.commit();	
}

void loop()
{
	int count = 0;
	ArduinoOTA.handle();
	MQTT_connect();
  int balwaterlevellow = 0, terracewaterlevellow = 0;
  static int oldbalwaterlevellow = 0, oldterracewaterlevellow = 0;
	char publish[100];
	int type, value1, value2, value3, value4;

	balwaterlevellow = digitalRead(WATERLEVELBALCONY);
	terracewaterlevellow = digitalRead(WATERLEVELTERRACE);
	Adafruit_MQTT_Subscribe *subscription;
	while ((subscription = mqtt.readSubscription(5000)))
	{
		if (subscription == &plant)
		{
			Serial.print(F("GotB: "));
			Serial.println((char *)plant.lastread);
			sscanf((char *)plant.lastread, "%d,%d,%d,%d,%d", &type, &value1, &value2, &value3, &value4);


			if (type == BPLANTS)
			{
				if (value1)// && !balwaterlevellow)
				{
					digitalWrite(RELAY3, !ENABLE_HIGH);
					digitalWrite(RELAY4, !ENABLE_HIGH);

					for(count = 0; count < value1; count++)
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
				}else
				{
					bcnt = 0;
					EEPROM.write(BCNT, bcnt);
				}
			}else if(type == TPLANTS)
			{
				if (value1)// && terracewaterlevellow)
				{
					digitalWrite(RELAY1, !ENABLE_HIGH);
					digitalWrite(RELAY2, !ENABLE_HIGH);

					for(count = 0; count < value1; count++)
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
				}else
				{
					tcnt = 0;
					EEPROM.write(TCNT, tcnt);
				}
			}else if(type == RESETREPORT)
			{
				bcnt = value1;
				btotal = value2;
				tcnt = value3;
				ttotal = value4;
				resetEEROM(value1, value2, value3, value4);
			}

      if ((oldbalwaterlevellow != balwaterlevellow) ||
         (terracewaterlevellow != oldterracewaterlevellow))
      {
        type = AUTOPUBLISH;
        oldterracewaterlevellow = terracewaterlevellow;
        oldbalwaterlevellow = balwaterlevellow;
      }
      
      switch(type)
      {
        case RESETREPORT:
        case TPLANTS:
        case BPLANTS:
                EEPROM.commit();  
                //break;
        case GETREPORT:
        case AUTOPUBLISH:
          memset(publish, 0, sizeof(publish));
          sprintf(publish, "%d,%d,%d,%d,%d,%d,%d,%s,%s", REPORT, bcnt, btotal, tcnt,
              ttotal, balwaterlevellow, terracewaterlevellow,__DATE__,__TIME__);
          status.publish(publish);
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
