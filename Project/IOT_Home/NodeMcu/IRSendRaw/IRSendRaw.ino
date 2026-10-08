#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoOTA.h>
#include <IRremoteESP8266.h>
#include <IRsend.h>
#include "credentials.h"

const char* ssid = SSID;
const char* password = SSID_PASSWORD;

constexpr uint16_t IR_TRANSMITTER_PIN = D8; //D5;
constexpr uint16_t IR_FREQUENCY_KHZ = 38;
constexpr uint16_t RAW_SIGNAL_LENGTH = 68;

ESP8266WebServer server(80);
IRsend irsend(IR_TRANSMITTER_PIN);

const uint16_t RAW_FAN_POWER[RAW_SIGNAL_LENGTH] PROGMEM = {9148, 4426, 656, 472, 654, 474, 654, 474, 654, 474, 656, 472, 654, 472, 654, 472, 600, 528, 654, 1596, 654, 1598, 654, 1598, 654, 1596, 600, 1652, 652, 1598, 654, 1596, 654, 1596, 654, 1598, 654, 472, 656, 472, 654, 1598, 656, 472, 654, 474, 656, 472, 656, 472, 654, 474, 656, 1596, 656, 1596, 654, 474, 598, 1652, 654, 1598, 656, 1596, 654, 1598, 654};
const uint16_t RAW_FAN_SPEED1[RAW_SIGNAL_LENGTH] PROGMEM = {9106, 4456, 626, 502, 626, 524, 604, 522, 606, 522, 606, 522, 606, 522, 606, 522, 606, 524, 604, 1648, 604, 1624, 626, 1648, 604, 1646, 606, 1646, 606, 1646, 604, 1646, 606, 1646, 604, 1646, 604, 502, 626, 1646, 606, 1626, 626, 524, 604, 502, 626, 522, 606, 502, 626, 500, 628, 1646, 606, 522, 606, 502, 626, 1648, 604, 1646, 606, 1646, 604, 1646, 606};
const uint16_t RAW_FAN_SPEED2[RAW_SIGNAL_LENGTH] PROGMEM = {9102, 4476, 656, 454, 674, 470, 658, 470, 602, 506, 676, 472, 656, 472, 656, 470, 658, 472, 602, 1648, 602, 1630, 674, 1596, 658, 1594, 656, 1594, 602, 1632, 676, 1574, 678, 1594, 660, 1592, 654, 452, 596, 552, 656, 1594, 660, 1590, 658, 470, 604, 506, 678, 468, 602, 508, 620, 1648, 658, 1574, 676, 472, 656, 470, 602, 1650, 600, 1630, 676, 1576, 676};
const uint16_t RAW_FAN_SPEED3[RAW_SIGNAL_LENGTH] PROGMEM = {9020, 4540, 542, 612, 516, 614, 516, 612, 516, 612, 516, 588, 540, 612, 514, 614, 514, 614, 514, 1738, 516, 1712, 566, 1686, 566, 1686, 540, 1710, 542, 1712, 540, 1712, 540, 1710, 542, 1712, 542, 1736, 522, 582, 540, 1712, 542, 1714, 538, 586, 542, 612, 516, 588, 564, 586, 520, 584, 540, 1710, 540, 588, 540, 612, 542, 1686, 540, 1714, 568, 1684, 540};
const uint16_t RAW_FAN_SPEED4[RAW_SIGNAL_LENGTH] PROGMEM = {9074, 4488, 594, 536, 588, 538, 568, 582, 546, 582, 570, 556, 546, 582, 546, 582, 546, 582, 546, 1706, 570, 1660, 568, 1682, 582, 1694, 546, 1706, 572, 1658, 594, 1680, 546, 1684, 594, 1680, 572, 536, 566, 582, 572, 556, 546, 582, 572, 556, 546, 582, 546, 582, 546, 582, 572, 1660, 594, 1680, 570, 1682, 546, 1684, 568, 1706, 546, 1684, 592, 1660, 568};
const uint16_t RAW_FAN_SPEED5[RAW_SIGNAL_LENGTH] PROGMEM = {9078, 4484, 600, 550, 578, 550, 578, 550, 578, 550, 578, 530, 600, 550, 580, 548, 578, 550, 576, 1674, 580, 1650, 602, 1652, 600, 1674, 578, 1652, 600, 1674, 578, 1674, 578, 1672, 580, 1674, 578, 550, 578, 548, 578, 552, 578, 1674, 578, 550, 578, 550, 578, 550, 578, 550, 578, 1674, 578, 1674, 578, 1654, 600, 550, 578, 1674, 578, 1654, 600, 1674, 578};
const uint16_t RAW_FAN_SPEED6[RAW_SIGNAL_LENGTH] PROGMEM = {9076, 4482, 654, 498, 604, 524, 604, 524, 604, 502, 626, 524, 606, 522, 604, 504, 626, 522, 604, 1628, 626, 1624, 628, 1646, 580, 1650, 626, 1628, 626, 1626, 626, 1648, 580, 1652, 624, 1626, 626, 524, 604, 1648, 604, 524, 606, 1646, 604, 502, 624, 526, 604, 524, 580, 526, 628, 1626, 654, 496, 580, 1672, 604, 502, 626, 1648, 604, 1624, 628, 1626, 626};
const uint16_t RAW_FAN_SPEED7[RAW_SIGNAL_LENGTH] PROGMEM = {9074, 4490, 594, 532, 594, 534, 594, 534, 594, 556, 572, 554, 574, 554, 572, 536, 594, 556, 572, 1658, 596, 1656, 594, 1658, 594, 1678, 574, 1678, 572, 1680, 572, 1678, 574, 1678, 572, 1658, 594, 1678, 574, 1680, 572, 556, 572, 1658, 596, 554, 576, 530, 594, 532, 596, 556, 572, 554, 572, 556, 572, 1658, 594, 554, 574, 1678, 574, 1658, 592, 1680, 574};
const uint16_t RAW_FAN_SPEED8[RAW_SIGNAL_LENGTH] PROGMEM = {9082, 4482, 628, 502, 624, 524, 578, 548, 604, 502, 600, 548, 604, 522, 580, 550, 604, 524, 578, 1674, 604, 1648, 604, 1624, 656, 1618, 634, 1618, 604, 1626, 626, 1648, 604, 1648, 604, 524, 578, 1672, 632, 1620, 580, 548, 604, 524, 578, 550, 604, 524, 580, 526, 602, 1672, 604, 524, 604, 524, 580, 1652, 600, 1674, 580, 1652, 626, 1648, 604, 1626, 626};
const uint16_t RAW_FAN_SPEED9[RAW_SIGNAL_LENGTH] PROGMEM = {9108, 4480, 602, 526, 602, 526, 604, 524, 602, 526, 602, 508, 620, 526, 602, 526, 602, 506, 622, 1648, 602, 1650, 602, 1650, 602, 1648, 602, 1650, 604, 1650, 602, 1650, 602, 1650, 604, 526, 602, 1630, 622, 524, 604, 524, 602, 1648, 604, 526, 602, 526, 602, 506, 622, 1630, 620, 526, 602, 1648, 604, 1648, 604, 526, 602, 1650, 602, 1648, 604, 1648, 604};

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
  char buffer[2048] = {};
  size_t head = 0;
  size_t count = 0;
};

WebLogSerial webLog;

const char LOG_PAGE[] PROGMEM = R"rawliteral(
<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IR Fan Control</title><style>
:root{color-scheme:dark;--bg:#12191a;--panel:#1d2929;--line:#394847;--text:#edf3ed;--muted:#aebbb3;--green:#9fe0b3;--red:#f0a29a}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#192323,#101516 65%);color:var(--text);font:16px/1.45 system-ui,sans-serif;min-height:100vh}
main{width:min(680px,100%);margin:auto;padding:24px 18px}header{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:20px}
h1{font-size:22px;margin:0;font-weight:650}.state{font:13px monospace;color:var(--muted)}h2{font-size:14px;color:var(--muted);font-weight:600;margin:22px 0 10px}
button{font:600 15px system-ui,sans-serif;color:var(--text);background:#293737;border:1px solid #526360;border-radius:6px;min-height:48px;padding:10px 14px;cursor:pointer}
button:hover:not(:disabled){background:#354744;border-color:var(--green)}button:focus-visible{outline:2px solid var(--green);outline-offset:2px}
button:disabled{opacity:.42;cursor:not-allowed}.power{width:100%;background:#30513e;border-color:#5a8e6e}.power.off{background:#492d2a;border-color:#81534d}
.speeds{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}.speed{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px}.speed small{font:11px system-ui,sans-serif;color:var(--muted)}
#message{min-height:22px;color:var(--green);font-size:13px;margin-top:12px}.logs{border:1px solid var(--line);border-radius:6px;background:#0d1213;overflow:hidden;margin-top:24px}
.logs h2{margin:0;padding:10px 12px;border-bottom:1px solid var(--line)}pre{height:190px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:12px;color:#c6d3cb;font:12px/1.5 ui-monospace,monospace}
@media(min-width:560px){main{padding-top:36px}.speeds{grid-template-columns:repeat(5,minmax(0,1fr))}}
</style></head><body><main><header><h1>IR Fan Control</h1><span id="powerState" class="state">Power unknown</span></header>
<button id="power" class="power" onclick="sendCommand('power')">Power</button>
<h2>Fan speed</h2><section class="speeds">
<button class="speed" onclick="sendCommand('speed1')">Speed 1</button><button class="speed" onclick="sendCommand('speed2')">Speed 2</button><button class="speed" onclick="sendCommand('speed3')">Speed 3</button>
<button class="speed" onclick="sendCommand('speed4')">Speed 4</button><button class="speed" onclick="sendCommand('speed5')">Speed 5</button>
<button class="speed" onclick="sendCommand('speed6')">Speed 6</button>
<button class="speed" onclick="sendCommand('speed7')">Speed 7</button>
<button class="speed" onclick="sendCommand('speed8')">Speed 8</button>
<button class="speed" onclick="sendCommand('speed9')">Speed 9</button>
</section><div id="message" role="status"></div>
<section class="logs"><h2>Device log</h2><pre id="logs">Connecting...</pre></section></main>
<script>
const message=document.getElementById('message'),logs=document.getElementById('logs'),powerState=document.getElementById('powerState');
async function sendCommand(cmd){message.textContent='Sending...';try{const response=await fetch('/api/send',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'cmd='+encodeURIComponent(cmd)});const text=await response.text();message.textContent=text;if(response.ok&&cmd==='power')powerState.textContent='Power signal sent';refreshLogs()}catch(error){message.textContent='Device connection failed'}}
async function refreshLogs(){try{const response=await fetch('/logs',{cache:'no-store'});if(response.ok){const atBottom=logs.scrollHeight-logs.scrollTop<=logs.clientHeight+20;logs.textContent=await response.text();if(atBottom)logs.scrollTop=logs.scrollHeight}}catch(error){}setTimeout(refreshLogs,1000)}refreshLogs();
</script></body></html>
)rawliteral";

void sendRawSignal(const uint16_t* signal, uint16_t length) {
  uint16_t raw[RAW_SIGNAL_LENGTH];
  for (uint16_t i = 0; i < length; i++) {
    raw[i] = pgm_read_word(&signal[i]);
  }
  irsend.sendRaw(raw, length, IR_FREQUENCY_KHZ);
}

void setup() {
  Serial.begin(115200);
  webLog.println("Booting IRTestRaw");

  irsend.begin();
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  webLog.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    webLog.print(".");
  }
  webLog.println();

  ArduinoOTA.setHostname("IRTestRaw");
  ArduinoOTA.onStart([]() { webLog.println("OTA update started"); });
  ArduinoOTA.onEnd([]() { webLog.println("OTA update complete"); });
  ArduinoOTA.onError([](ota_error_t error) {
    webLog.print("OTA error: ");
    webLog.println(static_cast<unsigned int>(error));
  });
  ArduinoOTA.begin();

  server.on("/", HTTP_GET, []() {
    server.send_P(200, "text/html", LOG_PAGE);
  });
  server.on("/logs", HTTP_GET, []() {
    server.sendHeader("Cache-Control", "no-store");
    server.send(200, "text/plain; charset=utf-8", webLog.snapshot());
  });
  server.on("/api/send", HTTP_POST, []() {
    if (!server.hasArg("cmd")) {
      server.send(400, "text/plain", "Missing command");
      return;
    }

    String command = server.arg("cmd");
    command.toLowerCase();
    if (command == "power") {
      sendRawSignal(RAW_FAN_POWER, RAW_SIGNAL_LENGTH);
    } else if (command == "speed1") sendRawSignal(RAW_FAN_SPEED1, RAW_SIGNAL_LENGTH);
    else if (command == "speed2") sendRawSignal(RAW_FAN_SPEED2, RAW_SIGNAL_LENGTH);
    else if (command == "speed3") sendRawSignal(RAW_FAN_SPEED3, RAW_SIGNAL_LENGTH);
    else if (command == "speed4") sendRawSignal(RAW_FAN_SPEED4, RAW_SIGNAL_LENGTH);
    else if (command == "speed5") sendRawSignal(RAW_FAN_SPEED5, RAW_SIGNAL_LENGTH);
    else if (command == "speed6") sendRawSignal(RAW_FAN_SPEED6, RAW_SIGNAL_LENGTH);
    else if (command == "speed7") sendRawSignal(RAW_FAN_SPEED7, RAW_SIGNAL_LENGTH);
    else if (command == "speed8") sendRawSignal(RAW_FAN_SPEED8, RAW_SIGNAL_LENGTH);
    else if (command == "speed9") sendRawSignal(RAW_FAN_SPEED9, RAW_SIGNAL_LENGTH);
    else {
      server.send(400, "text/plain", "Unknown command");
      return;
    }

    webLog.print("IR: ");
    webLog.println(command);
    server.send(200, "text/plain", "Sent " + command);
  });
  server.onNotFound([]() {
    server.send(404, "text/plain", "Not found");
  });
  server.begin();

  webLog.print("Web interface: http://");
  webLog.println(WiFi.localIP());
  webLog.println("OTA ready as IRTestRaw");
}

void loop() {
  ArduinoOTA.handle();
  server.handleClient();
}