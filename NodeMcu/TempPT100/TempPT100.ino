#include <ArduinoOTA.h>
#include "inc/key.h"
const char* ssid = WLAN_SSID;
const char* password = WLAN_PASS;

uint8_t Addr0 = D0;
uint8_t Addr1 = D1;
uint8_t Addr2 = D2;
uint8_t Addr3 = D4;
uint8_t Addr4 = D10;

uint8_t LedA0 = D5;
uint8_t LedA1 = D6;
uint8_t LedA2 = D7;

uint8_t led[8][3] =
{
  {0, 0, 1}, // A
  {0, 1, 0}, // B
  {0, 1, 1}, // D
  {1, 0, 0}, // C
  {1, 0, 1}, // E
  {1, 1, 0}, // F
  {1, 1, 1}, // G
  {0, 0, 0}  // NP
};

enum segment
{
  A, B, D, C, E, F, G, NL
};

void ledsegment(enum segment pin)
{
  unsigned int gpios = GPO;
  if (led[pin][2])
  {
    gpios |=  1 << LedA0;
  }
  else
  {
    gpios &= ~(1 << LedA0);
  }

  if (led[pin][1])
  {
    gpios |=  1 << LedA1;
  }
  else
  {
    gpios &= ~(1 << LedA1);
  }

  if (led[pin][0])
  {
    gpios |=  1 << LedA2;
  }
  else
  {
    gpios &= ~(1 << LedA2);

  }
  GPO = gpios;
  //digitalWrite(LedA0, led[pin][2]);
  //digitalWrite(LedA1, led[pin][1]);
  //digitalWrite(LedA2, led[pin][0]);

}

enum
{
  CRYB1,
  CRYB2,
  CRYB3,
  CRLED1,
  CYLED1,
  CBLED1,
};

uint8_t chipl[6][3] =
{
  {0, 0, 1}, //CRYB1
  {0, 1, 0}, //CRYB2
  {0, 1, 1}, //CRYB3
  {1, 0, 0}, //CRLED1
  {1, 0, 1}, //CYLED2
  {1, 1, 0}, //CBLED3
};

void chiplselect(uint8_t pin)
{
#if 0
  unsigned int gpios = GPO;
  if (chipl[pin][2])
  {
    gpios |= (1 << Addr0);
  }
  else
  {
    gpios &= ~(1 << Addr0);
  }

  if (chipl[pin][1])
  {
    gpios |= (1 << Addr1);
  }
  else
  {
    gpios &= ~(1 << Addr1);
  }

  if (chipl[pin][0])
  {
    gpios |= (1 << Addr2);
  }
  else
  {
    gpios &= ~(1 << Addr2);
  }
  GPO = gpios;
#else
  ledsegment(NL);
  digitalWrite(Addr0, chipl[pin][2]);
  digitalWrite(Addr1, chipl[pin][1]);
  digitalWrite(Addr2, chipl[pin][0]);
#endif
}

segment number[][7] =
{
  {A, B, C, D, E, F, NL},   // 0
  {B, C, NL},               // 1
  {A, B, G, E, D, NL},      // 2
  {A, B, G, C, D, NL},      // 3
  {F, G, B, C, NL},         // 4
  {A, F, G, C, D, NL},      // 5
  {A, F, E, D , C , G, NL}, // 6
  {A, B, C, NL},            // 7
  {A, B, C, D, E, F, G},    // 8
  {A, B, C, D, F, G, NL},   // 9
};

uint8_t pinarrayval[][7] = {
  { LOW, LOW, LOW, LOW, LOW, LOW, HIGH}, // 0
  {HIGH, LOW, LOW, HIGH, HIGH, HIGH, HIGH}, // 1
  { LOW, LOW, HIGH, LOW, LOW, HIGH, LOW}, // 2
  { LOW, LOW, LOW, LOW, HIGH, HIGH, LOW}, // 3
  {HIGH, LOW, LOW, HIGH, HIGH, LOW, LOW}, // 4
  { LOW, HIGH, LOW, LOW, HIGH, LOW, LOW}, // 5
  { LOW, HIGH, LOW, LOW, LOW, LOW, LOW}, // 6
  { LOW, LOW, LOW, HIGH, HIGH, HIGH, HIGH}, // 7
  { LOW, LOW, LOW, LOW, LOW, LOW, LOW}, // 8
  { LOW, LOW, LOW, LOW, HIGH, LOW, LOW}, // 9
  {HIGH, HIGH, HIGH, HIGH, HIGH, HIGH, HIGH}, // clear
  { LOW,  LOW,  LOW,  LOW,  LOW,  LOW,  LOW}, // clear

};


void setup() {
  pinMode(Addr0, OUTPUT);
  pinMode(Addr1, OUTPUT);
  pinMode(Addr2, OUTPUT);
  pinMode(Addr3, OUTPUT);
  pinMode(Addr4, OUTPUT);

  pinMode(LedA0, OUTPUT);
  pinMode(LedA1, OUTPUT);
  pinMode(LedA2, OUTPUT);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.waitForConnectResult() != WL_CONNECTED) {
    Serial.println("Connection Failed! Rebooting...");
    delay(5000);
    ESP.restart();
  }
  ArduinoOTA.setHostname("TempPT100");
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
  wdt_disable();
}

void displaynumbers(uint8_t pin, uint8_t digit)
{
  switch (pin)
  {
    case CRYB1:
      chiplselect(CRYB1);
      break;

    case CRYB2:
      chiplselect(CRYB2);
      break;

    case CRYB3:
      chiplselect(CRYB3);
      break;
  }

  for (uint8_t idx = 0; idx < 7; idx++)
  {
    ledsegment(number[digit][idx]);
    if (number[digit][idx] == NL)
    {
      break;
    }
  }
}

void watchdogfeed()
{
  ESP.wdtFeed();
  ESP.wdtDisable();
}
#define DELAYCOUNT 1000
void loop()
{
  uint32_t delaycount = DELAYCOUNT;
  uint16_t num = 0;
  while (1)
  {
    ArduinoOTA.handle();
#if 1
    if (delaycount == 0)
    {
      num++;
      delaycount = DELAYCOUNT;
      if (num == 999)
        num = 0;
    } else
    {
      delaycount--;
    }

    {
      uint8_t hund  = num / 100;
      uint8_t tensd = (num - hund * 100) / 10;
      uint8_t onced = (num - hund * 100 - tensd * 10);
      if (hund)
        displaynumbers(CRYB1, hund);

      if (hund || tensd)
        displaynumbers(CRYB2, tensd);

      displaynumbers(CRYB3, onced);
      ledsegment(NL);
    }
#else
    chiplselect(CRYB1);
    ledsegment(A);
    //ledsegment(F);
    //ledsegment(G);
    //ledsegment(B);
    //ledsegment(C);
    ledsegment(D);
    ledsegment(NL);
#endif
    ledsegment(NL);
    delay(0);
    //delayMicroseconds(100);
    watchdogfeed();
  }
}
