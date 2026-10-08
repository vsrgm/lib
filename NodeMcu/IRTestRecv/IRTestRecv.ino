#include <IRrecv.h>
#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoOTA.h>
#include "credentials.h"

const char* ssid = SSID;
const char* password = SSID_PASSWORD;

ESP8266WebServer server(80);

class WebLogSerial : public Print {
public:
  size_t write(uint8_t value) override {
    Serial.write(value);
    buffer[head] = value;
    head = (head + 1) % sizeof(buffer);
    if (count < sizeof(buffer)) count++;
    return 1;
  }

  String snapshot() const {
    String result;
    result.reserve(count);
    size_t start = (head + sizeof(buffer) - count) % sizeof(buffer);
    for (size_t i = 0; i < count; i++) {
      result += buffer[(start + i) % sizeof(buffer)];
    }
    return result;
  }

private:
  char buffer[4096] = {};
  size_t head = 0;
  size_t count = 0;
};

WebLogSerial webLog;
const char LOG_PAGE[] PROGMEM = R"rawliteral(
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IR Receiver Logs</title><style>
body{margin:0;background:#101719;color:#e5eee9;font:15px/1.5 monospace}
header{padding:14px 18px;background:#1c292a;border-bottom:1px solid #405052;position:sticky;top:0}
h1{font:600 18px sans-serif;margin:0}#status{color:#97c7ad;font:12px sans-serif;margin-top:4px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:16px;min-height:70vh}
</style></head><body><header><h1>IR Receiver</h1><div id="status">Connecting to device...</div></header>
<pre id="logs"></pre><script>
const output=document.getElementById('logs'),status=document.getElementById('status');
async function refresh(){try{const response=await fetch('/logs',{cache:'no-store'});if(!response.ok)throw Error();
const wasAtBottom=window.innerHeight+window.scrollY>=document.body.offsetHeight-40;
output.textContent=await response.text();status.textContent='Live serial output';if(wasAtBottom)window.scrollTo(0,document.body.scrollHeight);
}catch(e){status.textContent='Device connection lost; retrying...';}setTimeout(refresh,500)}refresh();
</script></body></html>
)rawliteral";

uint16_t RECV_PIN = D4;
IRrecv irrecv(RECV_PIN);

void setup() {
  Serial.begin(115200);
  webLog.println("Booting");

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  webLog.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    webLog.print(".");
  }
  webLog.println();

  ArduinoOTA.setHostname("IRTestRecv");
  ArduinoOTA.begin();

  server.on("/", []() {
    server.send_P(200, "text/html", LOG_PAGE);
  });
  server.on("/logs", []() {
    server.sendHeader("Cache-Control", "no-store");
    server.send(200, "text/plain; charset=utf-8", webLog.snapshot());
  });
  server.begin();
  webLog.print("Web interface: http://");
  webLog.println(WiFi.localIP());
  webLog.println("OTA ready");

  // put your setup code here, to run once:
  irrecv.enableIRIn();  // Start the receiver

}

void loop() {
  ArduinoOTA.handle();
  server.handleClient();

  decode_results results;
  unsigned long irhash = 0, ircode;
  // put your main code here, to run repeatedly:

  if (irrecv.decode(&results))
  {
    irhash = dump(&results);
    ircode = results.value;
    webLog.println(ircode);
    irrecv.resume();  // Receive the next value
  }
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
  webLog.println(c);
  c++;
  webLog.println("Hash: ");
  unsigned long hash = decodeHash(results);
  webLog.println(hash, HEX);
  webLog.println("For IR Scope/IrScrutinizer: ");
  for (int i = 1; i < count; i++) {

    if ((i % 2) == 1) {
      webLog.print("+");
      webLog.print(results->rawbuf[i]*USECPERTICK, DEC);
    }
    else {
      webLog.print(-(int)results->rawbuf[i]*USECPERTICK, DEC);
    }
    webLog.print(" ");
  }
  webLog.println("-127976");
  webLog.println("For Arduino sketch: ");
  webLog.print("unsigned int raw[");
  webLog.print(count, DEC);
  webLog.print("] = {");
  for (int i = 1; i < count; i++) {

    if ((i % 2) == 1) {
      webLog.print(results->rawbuf[i]*USECPERTICK, DEC);
    }
    else {
      webLog.print((int)results->rawbuf[i]*USECPERTICK, DEC);
    }
    webLog.print(",");
  }
  webLog.print("};");
  webLog.println("");
  webLog.print("irsend.sendRaw(raw,");
  webLog.print(count, DEC);
  webLog.print(",38);");
  webLog.println("");
  webLog.println("");
  return hash;
}
