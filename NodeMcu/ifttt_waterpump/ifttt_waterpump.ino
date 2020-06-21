//TODO : Implement WatchDog to safe reboot in case if any hang in the board


#include <ESP8266WiFi.h>
#include "Adafruit_MQTT.h"
#include "Adafruit_MQTT_Client.h"

#define Relay1		      D1
#define Relay2		      D2

#define ENABLE_HIGH     1 		
#define WLAN_SSID       "xxxxxxxxx"	// Your SSID
#define WLAN_PASS       "xxxxxxxxx"	// Your password

/************************* Adafruit.io Setup *********************************/

#define AIO_SERVER      "io.adafruit.com"
#define AIO_SERVERPORT  1883			// use 8883 for SSL
#define AIO_USERNAME    "xxxxxxxxx"	// Replace it with your username
#define AIO_KEY         "xxxxxxxxxxxxxxxxxx"	// Replace with your Project Auth Key

/************ Global State (you don't need to change this!) ******************/

// Create an ESP8266 WiFiClient class to connect to the MQTT server.
WiFiClient client;
// or... use WiFiFlientSecure for SSL
//WiFiClientSecure client;

// Setup the MQTT client class by passing in the WiFi client and MQTT server and login details.
Adafruit_MQTT_Client mqtt(&client, AIO_SERVER, AIO_SERVERPORT, AIO_USERNAME, AIO_KEY);

/****************************** Feeds ***************************************/


// Setup a feed called 'onoff' for subscribing to changes.
Adafruit_MQTT_Subscribe bplant = Adafruit_MQTT_Subscribe(&mqtt, AIO_USERNAME"/feeds/balconyplants"); // FeedName
Adafruit_MQTT_Subscribe tplant = Adafruit_MQTT_Subscribe(&mqtt, AIO_USERNAME"/feeds/terraceplants"); // FeedName

void MQTT_connect();

void setup()
{
	pinMode(Relay1, OUTPUT);
	digitalWrite(Relay1, ENABLE_HIGH);
	pinMode(Relay2, OUTPUT);
	digitalWrite(Relay2, ENABLE_HIGH);

	Serial.begin(115200);

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
	mqtt.subscribe(&bplant);
	mqtt.subscribe(&tplant);
}

#define BTIMER_WATER_PLANTS 10000 //millisec
#define TTIMER_WATER_PLANTS 20000 //millisec

void loop()
{
	MQTT_connect();

	Adafruit_MQTT_Subscribe *subscription;
	while ((subscription = mqtt.readSubscription(5000)))
	{
		if (subscription == &bplant)
		{
			Serial.print(F("GotB: "));
			Serial.println((char *)bplant.lastread);
			int bplant_State = atoi((char *)bplant.lastread);
			if (bplant_State)
			{
				digitalWrite(Relay1, !bplant_State);
				delay(BTIMER_WATER_PLANTS);
				digitalWrite(Relay1, bplant_State);
			}
		}
		if (subscription == &tplant)
		{
			Serial.print(F("GotT: "));
			Serial.println((char *)tplant.lastread);
			int tplant_State = atoi((char *)tplant.lastread);
			if (tplant_State)
			{
				digitalWrite(Relay2, !tplant_State);
				delay(TTIMER_WATER_PLANTS);
				digitalWrite(Relay2, tplant_State);
			}
		}
	}
}

void MQTT_connect()
{
	int8_t ret;

	// Stop if already connected.
	if (mqtt.connected())
	{
		return;
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
			// basically die and wait for WDT to reset me
			while (1);
		}
	}
	Serial.println("MQTT Connected!");
}
