# Implementation Status - IOT Home Automation
**Current Version: 1.1.1** (Updated on 18-07-2026)

## Requirement 0: Toilet Assistance (NodeMcu) - [NEW]
- [x] Create `ToiletAssistanceActivity`.
- [x] Implement NodeMcu firmware for Toilet.
    - [x] Sensors: PIR, LDR (A0), DHT11.
    - [x] Actuators: Exhaust Fan (Relay), Emergency Light, Buzzer.
    - [x] Logic: PIR -> Fan (1 min), PIR + Dark -> Light (1 min), No motion + Light -> Buzzer.
    - [x] Internal Logging (.csv via LittleFS).
    - [x] MQTT & Local IP Status/Control.
    - [x] HTTP Firmware Update support.
- [x] Implement Android UI.
    - [x] Status rows for all sensors following "Label - Value" and "Label - Switch - Value" patterns.
    - [x] History log view.
    - [x] Navigation from `SmartHomeActivity`.
    - [x] Settings-based sync (MQTT vs Local IP).

## Requirement 1: Study Room Monitor
- [x] Create `StudyRoomActivity`.
- [x] Create `SmartHomeActivity` as an intermediate menu.
- [x] Implement NodeMcu firmware for Study Room.
    - [x] Sensors: LDR (A0), DHT11 (D3).
    - [x] Actuators: Emergency Light (D2).
    - [x] Logic: LDR detections Low light -> Turn on Emergency Light.
    - [x] Internal Logging (.csv via LittleFS).
    - [x] MQTT & Local IP Status/Control.
    - [x] HTTP Firmware Update support.
- [x] Implement Android UI.
    - [x] Status rows matching "Label - Value" and "Label - Switch - Value".
    - [x] History log view and Multi-port MQTT connection logging.
    - [x] Settings-based sync (MQTT vs Local IP) with default IP 192.168.0.107.
- [x] Navigation from `SmartHomeActivity`.

## Requirement 2: Security Main Door (ESP32-CAM)
- [x] Create `SecurityMainDoorActivity`.
- [x] Implement ESP32-CAM firmware for Main Door.
    - [x] MQTT status (PIR, Door, Bell, LDR, Power, Relay, Buzzer, BMP280).
    - [x] SD Card logging (.csv).
    - [x] SD Card image capture on event.
    - [x] Auto-lighting (LDR -> Relay).
    - [x] Manual controls via MQTT.
    - [x] Local IP Support: HTTP Stream Server (`/stream`), Status Endpoint (`/status`), Control Endpoint (`/control`).
- [x] Implement Android UI.
    - [x] WebView for live MJPEG streaming (Local IP mode).
    - [x] Consistent status view using `item_status_row` (like Study Room).
    - [x] Live status and history view.
    - [x] Manual turn on/off control (Relay, Buzzer, Capture).
    - [x] Image Viewer (Gallery) to see captured .jpg images.
    - [x] Settings-based sync (MQTT vs Local IP).
    - [x] **New**: Support receiving event-based images via MQTT and saving to Gallery.

## Requirement 3: Kitchen Monitor (ESP32-CAM)
- [x] Create `KitchenMonitorActivity`.
- [x] Implement ESP32-CAM firmware for Kitchen.
    - [x] MQTT status (PIR, LDR, Power, Relay, Buzzer, BMP280, MQ135).
    - [x] SD Card logging and image capture.
    - [x] Auto-lighting (PIR + LDR -> Relay).
    - [x] Gas leak alarm (MQ135 -> Buzzer).
    - [x] Manual controls.
    - [x] Local IP Support: HTTP Stream Server, Status and Control endpoints.
- [x] Implement Android UI.
    - [x] WebView for live MJPEG streaming (Local IP mode).
    - [x] Consistent status view using `item_status_row` (like Study Room).
    - [x] Live status and history view.
    - [x] Manual turn on/off control.
    - [x] Capture button for manual image capture.
    - [x] Image Viewer (Gallery) to see captured .jpg images.
    - [x] Settings-based sync (MQTT vs Local IP).
    - [x] **New**: Support receiving event-based images via MQTT and saving to Gallery.

## Requirement 4: RO Waste Water Pump (ESP8266) - [DONE]
- [x] Implement ESP8266 firmware for RO Waste Water.
    - [x] MQTT status (Relay, Water Level Float status).
    - [x] Logic: Relay ON only if Water Float is YES, OFF if LOW.
    - [x] Internal Logging (.csv via LittleFS).
    - [x] MQTT & Local IP Status/Control.
    - [x] HTTP Firmware Update support.
- [x] Implement Android UI.
    - [x] Create `RoWasteWaterActivity`.
    - [x] Status rows matching "Label - Value" and "Label - Switch - Value".
    - [x] History log view and Multi-port MQTT connection logging.
    - [x] Settings-based sync (MQTT vs Local IP).
    - [x] Navigation from `SmartHomeActivity`.

## Requirement 5: Common Features & RC Car Improvements
- [x] Implement `ImageViewerActivity` as a shared gallery for all cameras.
- [x] Add "Gallery" button to `MainActivity` (RC Car).
- [x] Ensure all status views follow the "Label - Value" pattern from Requirement 6.1.

## Requirement 5: MQTT Port Reliability & Logging - [NEW]
- [x] **Settings**: Update `SmartHomeSettingsActivity` to accept comma-separated MQTT ports.
- [x] **Multi-Port Failover**: Android app automatically tries all specified ports (e.g., 1883, 8000, 8883, 8884) until a connection is established.
- [x] **Connection Logging**: Added a scrollable "System Connection Logs" area in all Smart Home activities to monitor connection status in real-time (latest logs first).

## Suggestions / Notes
1. **MJPEG over MQTT**: Implemented event-based single-frame image transfer over MQTT to balance bandwidth and requirements. Full MJPEG stream is reserved for Local IP mode.
2. **NodeMcu Storage**: For Toilet Assistance, LittleFS is used for `.csv` logging as ESP8266 typically lacks an SD card. Storage is limited (~2-3MB).
3. **PIR Timeout Logic**: The 1-minute and 5-minute timers are implemented using non-blocking `millis()` in the firmware.
4. **Suggestions for Requirements Document**:
    * **Define Thresholds**: Specify ADC values for "Low" and "High" light to ensure consistent behavior across different hardware builds.
    * **Power Management**: For battery-operated nodes, sleep modes (Deep Sleep) should be defined to preserve power when no motion is detected.
    * **State Recovery**: Specify if relays should return to their last state after a power failure.
    * **Alerting**: For critical events (Gas leak, Toilet emergency), define if Android notifications should be triggered even if the app is in the background.
    * **SSL/TLS**: For ports 8883 and 8884, implement certificate validation if security is a priority. Currently uses `setInsecure()` or equivalent for ease of setup.
