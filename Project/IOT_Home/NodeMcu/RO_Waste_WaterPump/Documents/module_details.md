# ESP8266 ESP-01 Relay Module - Technical Guide

Based on the image provided, this is an **ESP8266 ESP-01 Relay Module** (not an ESP32). It is a popular, low-cost solution for Wi-Fi-controlled switching.

## 1. Key Components
*   **Microcontroller:** ESP8266 (ESP-01 or ESP-01S module).
    > [!NOTE]
    > The text "AI-Cloud Inside" on the black module confirms it is an AI-Thinker ESP8266 module.
*   **Relay:** Songle **SRD-5VDC-SL-C**.
    *   **Voltage:** 5V DC coil.
    *   **Max Load:** 10A @ 250VAC or 10A @ 30VDC.
*   **Voltage Regulator:** AMS1117-3.3V (converts the 5V input to 3.3V for the ESP8266).

---

## 2. Pinout & Connections

### Power Input (2-pin Green Terminal)
*   **VCC:** Connect to **5V DC** power supply.
*   **GND:** Connect to Ground.

### Relay Output (3-pin Green Terminal)
*   **NO (Normally Open):** Connected to COM only when the relay is active.
*   **COM (Common):** The main line of your load goes here.
*   **NC (Normally Closed):** Connected to COM when the relay is off.

### Programming Header (4-pin Black Header)
Most versions have `RX`, `TX`, `GND`, and `5V` pins. These can be used to monitor the serial output or communicate with the onboard MCU (depending on the version).

---

## 3. Two Common Versions (Crucial for Coding)
There are two main versions of this board. You must determine which one you have to write the correct code.

### Version A: GPIO 0 Control (Direct)
The relay is connected directly to the **GPIO 0** pin of the ESP-01.
*   **How to test:** Write a simple blink sketch for Pin 0.
*   **Logic:** Usually **Active Low** (setting GPIO 0 to `LOW` turns the relay **ON**).

### Version B: Serial Control (LC Technology)
The board has an additional 8-pin microcontroller (like the STC15F104) that sits between the ESP-01 and the relay.
*   **How to test:** You must send Hex commands over Serial at **9600** or **115200** baud.
*   **Commands (Hex):**
    *   **Open Relay:** `A0 01 01 A2`
    *   **Close Relay:** `A0 01 00 A1`

---

## 4. How to Program It
You **cannot** program the ESP-01 while it is plugged into the relay board.
1.  Unplug the black ESP-01 module.
2.  Plug it into a **USB-to-ESP01 Adapter** (or use an Arduino as an ISP).
3.  Upload your code from the Arduino IDE (Select board: "Generic ESP8266 Module").
4.  Plug it back into the relay board and power the board via the 5V terminal.

---

## 6. Memory Specifications
The ESP8266 inside your module has two types of memory:

### Flash Memory (Program Storage)
Based on your image showing a **Black PCB**, you likely have the **ESP-01S** variant.
*   **Capacity:** Typically **1 MB (8 Mbit)**.
*   **Usage:** This is where your code (firmware), Wi-Fi credentials, and small files (using LittleFS or SPIFFS) are stored.
*   **OTA Support:** 1 MB is enough to support Over-The-Air (OTA) updates.
*   *Note: Older Blue PCB versions only had 512 KB.*

### RAM (Working Memory)
All ESP8266 chips have the same internal RAM architecture:
*   **Instruction RAM:** 32 KB (for code execution).
*   **Data RAM:** 80 KB (for variables and stack).
    *   *Real-world availability:* After the system starts and connects to Wi-Fi, you usually have about **35 KB to 45 KB** of "Heap" memory available for your own variables and logic.

### How to verify your specific module:
You can run this snippet in the Arduino IDE to see exactly what your chip reports:
```cpp
void setup() {
  Serial.begin(115200);
}

void loop() {
  uint32_t realSize = ESP.getFlashChipRealSize();
  uint32_t ideSize = ESP.getFlashChipSize();

  Serial.printf("Flash Chip ID: %08X\n", ESP.getFlashChipId());
  Serial.printf("Flash Real Size: %u bytes\n", realSize);
  Serial.printf("Free Heap: %u bytes\n", ESP.getFreeHeap());
  delay(5000);
}
```
