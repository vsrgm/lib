# Raspberry Pi Camera Control Server

This server allows you to remotely start and stop the UDP video stream on your Raspberry Pi.

## Installation on Raspberry Pi

1. Transfer this folder to your Pi.
2. Install dependencies:
   ```bash
   pip install flask
   ```

## Running the C Server

If you prefer C over Python:

1. Compile the server:
   ```bash
   gcc server.c -o server
   ```
2. Run it:
   ```bash
   ./server
   ```
The server will start on port `5001`.

## API Endpoints

- **Start Stream**: `http://<PI_IP>:5001/start_stream?ip=<PHONE_IP>&port=5000`
- **Stop Stream**: `http://<PI_IP>:5001/stop_stream`
- **Status**: `http://<PI_IP>:5001/status`

## Integration with Android App

When you click "Connect" in the Android app, it will now automatically call this server to trigger the video feed.
