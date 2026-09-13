#include <Wire.h>

#define I2C_SLAVE_ADDR 0x40

// Pin Definitions
const int pin138_A0 = 0; // D0
const int pin138_A1 = 1; // D1
const int pin138_A2 = 2; // D2

const int pin7447_A = 3; // D3
const int pin7447_B = 4; // D4
const int pin7447_C = 5; // D5
const int pin7447_D = 6; // D6

const int pin_DP    = 7; // D7

char displayBuffer[9] = "--------"; // 8 digits + null terminator

void setup() {
  // Initialize digital pins as outputs
  pinMode(pin138_A0, OUTPUT);
  pinMode(pin138_A1, OUTPUT);
  pinMode(pin138_A2, OUTPUT);

  pinMode(pin7447_A, OUTPUT);
  pinMode(pin7447_B, OUTPUT);
  pinMode(pin7447_C, OUTPUT);
  pinMode(pin7447_D, OUTPUT);

  pinMode(pin_DP, OUTPUT);
  digitalWrite(pin_DP, HIGH); // DP active low, keep off initially

  // Initialize Wire as slave
  Wire.begin(I2C_SLAVE_ADDR);
  Wire.onReceive(receiveEvent);
}

void loop() {
  // Multiplex the 8 digits continuously
  for (int digit = 0; digit < 8; digit++) {
    char ch = displayBuffer[digit];

    // Set BCD inputs on SN74LS47N based on the character
    int val = 15; // Blank code or invalid BCD code to keep segments off
    bool dpOn = false;

    if (ch >= '0' && ch <= '9') {
      val = ch - '0';
    } else if (ch == '-') {
      // SN74LS47N doesn't natively do a single dash with standard BCD 0-9.
      // Standard BCD inputs 10-15 produce specific shapes or blanks.
      // Input 14 (binary 1110) displays a dash/minus symbol on SN74LS47N!
      val = 14;
    } else if (ch == ' ') {
      val = 15; // Blank
    } else if (ch == 'C' || ch == 'c') {
      // 74LS47 code for C shape isn't standard, let's blank it or use a custom digit look if direct.
      // Since we use 7447 hardware decoder, we must stick to its fixed decoding logic (0-9, and symbols for 10-15).
      val = 11; // 7447 displays a specific symbol/blank for 11. Let's use it as a custom code if needed.
    } else if (ch == 'H' || ch == 'h') {
      val = 12;
    } else if (ch == 'A' || ch == 'a') {
      val = 10;
    }

    // Write BCD bits to pins D3-D6
    digitalWrite(pin7447_A, (val & 0x01) ? HIGH : LOW);
    digitalWrite(pin7447_B, (val & 0x02) ? HIGH : LOW);
    digitalWrite(pin7447_C, (val & 0x04) ? HIGH : LOW);
    digitalWrite(pin7447_D, (val & 0x08) ? HIGH : LOW);

    // Set decimal point if needed (e.g. customized or always off)
    digitalWrite(pin_DP, dpOn ? LOW : HIGH);

    // Select the digit using 74HC138N (pins D0-D2)
    digitalWrite(pin138_A0, (digit & 0x01) ? HIGH : LOW);
    digitalWrite(pin138_A1, (digit & 0x02) ? HIGH : LOW);
    digitalWrite(pin138_A2, (digit & 0x04) ? HIGH : LOW);

    delay(2); // Short delay for persistence of vision
  }
}

// Function called when data is received from I2C master
void receiveEvent(int howMany) {
  int i = 0;
  while (Wire.available() && i < 8) {
    char c = Wire.read();
    displayBuffer[i++] = c;
  }
  // Fill remaining characters with spaces if message was shorter than 8 chars
  while (i < 8) {
    displayBuffer[i++] = ' ';
  }
  displayBuffer[8] = '\0';
}
