# RO Waste Water Pump Controller Implementation Plan

This project implements a Wi-Fi-controlled relay system to automate the transfer of RO waste water from a storage container to a toilet flush tank using an ESP8266 (ESP-01) and a relay module.

## User Review Required

- **Safety Mechanism:** Since the pump is transfering water to a toilet tank, how do we prevent overflow in the toilet tank?
    - *Option 1:* Add a float switch in the toilet tank connected to the ESP8266.
    - *Option 2:* Simple timer-based pump (e.g., run for 30 seconds when triggered).
- **Trigger Method:** How would you like to trigger the pump?
    - *Option A:* Physical button.
    - *Option B:* Mobile App/Web Interface.
    - *Option C:* Automated when the toilet tank is empty (requires a second sensor).

## Proposed Changes

### [ESP8266 Project](file:///C:/Users/pie5zk/AppData/Local/Google/AndroidStudio2026.1.1/projects/iot_home.6fb8d3d9/ESP8266/RO_Waste_WaterPump/)

#### [NEW] [RO_Waste_WaterPump.ino](file:///C:/Users/pie5zk/AppData/Local/Google/AndroidStudio2026.1.1/projects/iot_home.6fb8d3d9/ESP8266/RO_Waste_WaterPump/RO_Waste_WaterPump.ino)

- Basic firmware to control the relay.
- Implements a simple Web Server to toggle the pump.
- Includes a "Safety Timeout" to prevent the pump from running indefinitely if a sensor fails or the web command gets stuck.

#### [NEW] [README.md](file:///C:/Users/pie5zk/AppData/Local/Google/AndroidStudio2026.1.1/projects/iot_home.6fb8d3d9/ESP8266/RO_Waste_WaterPump/README.md)

- Wiring diagram for the relay and pump.
- Setup instructions for the Arduino IDE.

## Verification Plan

### Manual Verification
- **Relay Test:** Deploy a test script to toggle the relay every 5 seconds to verify the GPIO mapping (GPIO 0 vs Serial).
- **Web Interface Test:** Verify that the pump can be turned ON/OFF via a browser on the same network.
- **Timeout Test:** Verify that the pump automatically shuts off after the safety limit (e.g. 2 minutes).
