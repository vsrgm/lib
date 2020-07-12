//TODO : Implement WatchDog to safe reboot in case if any hang in the board

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
void MQTT_connect();

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
	BPLANTS,
	TPLANTS,
	RESETREPORT,
	REPORT,
};

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
	pinMode(WATERLEVELTERRACE, INPUT);
	pinMode(WATERLEVELBALCONY, INPUT);

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
	MQTT_connect();
	int balwaterlevellow = 0, terracewaterlevellow = 0;
	char publish[100];
	int type, value1, value2, value3, value4;
 
	Adafruit_MQTT_Subscribe *subscription;
	while ((subscription = mqtt.readSubscription(5000)))
	{
		if (subscription == &plant)
		{
			Serial.print(F("GotB: "));
			Serial.println((char *)plant.lastread);
			sscanf((char *)plant.lastread, "%d,%d,%d,%d,%d", &type, &value1, &value2, &value3, &value4);
			balwaterlevellow = digitalRead(WATERLEVELBALCONY);
			terracewaterlevellow = digitalRead(WATERLEVELTERRACE);

			if (type == BPLANTS)
			{
				if (value1)// && !balwaterlevellow)
				{
					digitalWrite(RELAY1, !ENABLE_HIGH);
					digitalWrite(RELAY2, !ENABLE_HIGH);

					for(count = 0; count < value1; count++)
						delay(1000);

					digitalWrite(RELAY1, ENABLE_HIGH);
					digitalWrite(RELAY2, ENABLE_HIGH);
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
				EEPROM.commit();	
				memset(publish, 0, sizeof(publish));
				sprintf(publish, "%d,%d,%d,%d,%d,%d,%d", REPORT, bcnt, btotal, tcnt,
						ttotal, balwaterlevellow, terracewaterlevellow);
				status.publish(publish);

			}else if(type == TPLANTS)
			{
				if (value1)// && !terracewaterlevellow)
				{
					digitalWrite(RELAY3, !ENABLE_HIGH);
					digitalWrite(RELAY4, !ENABLE_HIGH);

					for(count = 0; count < value1; count++)
						delay(1000);

					digitalWrite(RELAY3, ENABLE_HIGH);
					digitalWrite(RELAY4, ENABLE_HIGH);

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
				EEPROM.commit();	
				memset(publish, 0, sizeof(publish));
				sprintf(publish, "%d,%d,%d,%d,%d,%d,%d", REPORT, bcnt, btotal, tcnt,
						ttotal, balwaterlevellow, terracewaterlevellow);
				status.publish(publish);


			}else if(type == RESETREPORT)
			{
				bcnt = value1;
				btotal = value2;
				tcnt = value3;
				ttotal = value4;
				resetEEROM(value1, value2, value3, value4);
				EEPROM.commit();	
				memset(publish, 0, sizeof(publish));
				sprintf(publish, "%d,%d,%d,%d,%d,%d,%d", REPORT, bcnt, btotal, tcnt,
						ttotal, balwaterlevellow, terracewaterlevellow);
				status.publish(publish);

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
