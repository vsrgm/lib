package com.example.sample;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.sample.databinding.ActivitySecurityMainDoorBinding;

import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.MqttCallback;
import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;
import org.json.JSONObject;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkRequest;
import android.net.NetworkCapabilities;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Rect;
import androidx.annotation.NonNull;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.util.concurrent.atomic.AtomicBoolean;

public class SecurityMainDoorActivity extends AppCompatActivity {

    private ActivitySecurityMainDoorBinding binding;
    private MqttClient mqttClient;
    private FirebaseDatabase firebaseDatabase;
    private DatabaseReference outboxRef;
    private ValueEventListener firebaseListener;
    private final ExecutorService executor = Executors.newFixedThreadPool(2);
    private final String statusTopic = "FrmEsp32/main_door/status";
    private final String cmdTopic = "FrmMobile/main_door/command";
    private final String imageTopic = "FrmEsp32/main_door/image";
    private SharedPreferences prefs;
    private int syncMode = 0; // 0: MQTT, 1: IP, 2: Firebase
    private final List<String> discoveredNodes = new ArrayList<>();
    private android.widget.ArrayAdapter<String> nodeAdapter;
    private String selectedNodeId = "";
    private String selectedNodeIp = "";
    private final StringBuilder logBuilder = new StringBuilder();
    private int[] mqttPorts = {};
    private int currentPortIndex = 0;
    private ConnectivityManager connectivityManager;
    private ConnectivityManager.NetworkCallback networkCallback;
    private boolean isLocalAvailable = false;
    private long lastCommandTime = 0;
    
    // MJPEG Native Stats
    private final AtomicBoolean isNativeStreaming = new AtomicBoolean(false);
    private android.view.Surface savedSurface;
    private long totalBytesReceived = 0;
    private long frameCount = 0;
    private long lastStatsTime = 0;
    private long lastByteCount = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivitySecurityMainDoorBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        prefs = getSharedPreferences("SmartHomePrefs", MODE_PRIVATE);
        syncMode = prefs.getInt("sync_mode", 0);

        binding.videoSurface.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(@NonNull SurfaceHolder holder) {
                savedSurface = holder.getSurface();
            }
            @Override public void surfaceChanged(@NonNull SurfaceHolder h, int f, int w, int h1) {}
            @Override public void surfaceDestroyed(@NonNull SurfaceHolder h) { 
                savedSurface = null;
                stopNativeStream(); 
            }
        });

        binding.btnBack.setOnClickListener(v -> finish());
        binding.btnSettings.setOnClickListener(v -> {
            Intent intent = new Intent(this, SmartHomeSettingsActivity.class);
            intent.putExtra("caller_context", "door");
            startActivity(intent);
        });
        
        binding.btnCapture.setOnClickListener(v -> sendCommand("CAPTURE"));
        binding.btnCamConfig.setOnClickListener(v -> showCameraConfigDialog());
        binding.btnReboot.setOnClickListener(v -> {
            new androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("Reboot Device")
                .setMessage("Are you sure you want to reboot the ESP32-CAM?")
                .setPositiveButton("Yes", (dialog, which) -> sendCommand("REBOOT"))
                .setNegativeButton("No", null)
                .show();
        });
        binding.btnGallery.setOnClickListener(v -> {
            startActivity(new android.content.Intent(this, ImageViewerActivity.class));
        });

        // Connected Nodes Setup
        nodeAdapter = new android.widget.ArrayAdapter<>(this, android.R.layout.simple_spinner_item, discoveredNodes);
        nodeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.nodeSelector.setAdapter(nodeAdapter);
        binding.nodeSelector.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(android.widget.AdapterView<?> parent, android.view.View view, int position, long id) {
                String selected = discoveredNodes.get(position);
                if (selected.contains("(") && selected.endsWith(")")) {
                    selectedNodeId = selected.substring(0, selected.indexOf(" (")).trim();
                    selectedNodeIp = selected.substring(selected.lastIndexOf("(") + 1, selected.length() - 1);
                }
            }
            @Override public void onNothingSelected(android.widget.AdapterView<?> parent) {}
        });



        // Initialize labels and switches
        binding.rowPir.label.setText("PIR Motion");
        binding.rowDoor.label.setText("Door Sensor");
        binding.rowBell.label.setText("Calling Bell");
        binding.rowLdr.label.setText("Light Sensor");
        
        binding.rowRelay.label.setText("Door Lamp (Relay)");
        binding.rowRelay.sensorSwitch.setVisibility(View.VISIBLE);
        binding.rowRelay.sensorSwitch.setOnClickListener(v -> {
            boolean isChecked = binding.rowRelay.sensorSwitch.isChecked();
            sendCommand(isChecked ? "RELAY_ON" : "RELAY_OFF");
        });

        binding.rowBuzzer.label.setText("Alarm (Buzzer)");
        binding.rowBuzzer.sensorSwitch.setVisibility(View.VISIBLE);
        binding.rowBuzzer.sensorSwitch.setOnClickListener(v -> {
            boolean isChecked = binding.rowBuzzer.sensorSwitch.isChecked();
            sendCommand(isChecked ? "BUZZER_ON" : "BUZZER_OFF");
        });

        binding.rowFlash.label.setText("Built-in Flash");
        binding.rowFlash.sensorSwitch.setVisibility(View.VISIBLE);
        binding.rowFlash.sensorSwitch.setOnClickListener(v -> {
            boolean isChecked = binding.rowFlash.sensorSwitch.isChecked();
            sendCommand(isChecked ? "FLASH_ON" : "FLASH_OFF");
        });

        binding.rowTemp.label.setText("Temperature");
        binding.rowPres.label.setText("Pressure");

        binding.videoCard.setOnClickListener(v -> toggleLiveFirebase());

        setupWebView();
        
        if (syncMode == 1) {
            startIpSync();
        } else if (syncMode == 2) {
            initFirebase();
        } else {
            initMqtt();
            setupNetworkListener();
        }
    }

    private void initFirebase() {
        String url = prefs.getString("firebase_url", AppDefaults.FIREBASE_URL);
        String doorNode = prefs.getString("firebase_door_node", AppDefaults.NODE_DOOR);
        String room = doorNode;
        if (room.contains("/")) room = room.substring(room.lastIndexOf("/") + 1);
        
        final String finalRoom = room;
        addLog("Authenticating Firebase: " + url);
        String email = prefs.getString("firebase_email", Credentials.FIREBASE_EMAIL);
        String password = prefs.getString("firebase_password", Credentials.FIREBASE_PASSWORD);
        
        FirebaseAuth auth = FirebaseAuth.getInstance();
        if (auth.getCurrentUser() != null) {
            addLog("Firebase: Already Authenticated as " + auth.getCurrentUser().getEmail());
            connectToFirebase(url, finalRoom);
            return;
        }

        auth.signInWithEmailAndPassword(email, password)
            .addOnCompleteListener(task -> {
                if (task.isSuccessful()) {
                    addLog("Firebase Auth: SUCCESS");
                    connectToFirebase(url, finalRoom);
                } else {
                    String error = task.getException() != null ? task.getException().getMessage() : "Unknown Auth Error";
                    addLog("Firebase Auth FAILED: " + error);
                    Toast.makeText(this, "Firebase Auth Failed: " + error, Toast.LENGTH_LONG).show();
                    connectToFirebase(url, finalRoom);
                }
            });
    }

    private boolean isLiveFirebaseActive = false;
    private void toggleLiveFirebase() {
        isLiveFirebaseActive = !isLiveFirebaseActive;
        String doorNode = prefs.getString("firebase_door_node", AppDefaults.NODE_DOOR);
        String room = doorNode;
        if (room.contains("/")) room = room.substring(room.lastIndexOf("/") + 1);
        
        String commandNode = "FrmMobile/" + room; 
        
        String url = prefs.getString("firebase_url", AppDefaults.FIREBASE_URL);
        FirebaseDatabase database = firebaseDatabase != null ? firebaseDatabase : FirebaseDatabase.getInstance(url);
        
        database.getReference(commandNode).child("live_request")
                .setValue(isLiveFirebaseActive)
                .addOnFailureListener(e -> addLog("LiveReq Failed: " + e.getMessage()));
        
        addLog("Firebase Live Mode: " + (isLiveFirebaseActive ? "ON" : "OFF"));
        Toast.makeText(this, "Live Mode " + (isLiveFirebaseActive ? "Active" : "Stopped"), Toast.LENGTH_SHORT).show();
        
        if (isLiveFirebaseActive) {
            stopNativeStream(); // Stop local stream to show Firebase images
        } else if (isLocalAvailable) {
            // Restore local stream if available and Firebase mode turned off
            startNativeStream(selectedNodeIp.isEmpty() ? prefs.getString("local_node_ip", AppDefaults.DEFAULT_NODE_IP) : selectedNodeIp);
        }
    }

    private void connectToFirebase(String url, String room) {
        try {
            firebaseDatabase = FirebaseDatabase.getInstance(url);
            // LISTEN to the OUTBOX under FrmEsp32
            outboxRef = firebaseDatabase.getReference("FrmEsp32").child(room);
            addLog("Listening to Node: FrmEsp32/" + room);
            
            firebaseListener = new ValueEventListener() {
                @Override
                public void onDataChange(DataSnapshot dataSnapshot) {
                    if (dataSnapshot.exists()) {
                        // 1. Handle Status
                        DataSnapshot statusSnap = dataSnapshot.child("status");
                        if (statusSnap.exists()) {
                            Object value = statusSnap.getValue();
                            try {
                                JSONObject json;
                                if (value instanceof java.util.Map) {
                                    json = new JSONObject((java.util.Map) value);
                                } else {
                                    json = new JSONObject(value.toString());
                                }
                                handleStatus(json.toString());
                                if (json.has("ip") && !isLiveFirebaseActive) {
                                    checkLocalConnectivityAndSetupStream(json.getString("ip"));
                                }
                            } catch (Exception ignored) {}
                        }

                        // 2. Handle History
                        DataSnapshot historyNode = dataSnapshot.child("history");
                        if (historyNode.exists()) {
                            for (DataSnapshot child : historyNode.getChildren()) {
                                try {
                                    Object val = child.getValue();
                                    if (val != null) {
                                        JSONObject json = new JSONObject(val.toString());
                                        handleStatus(json.toString());
                                    }
                                } catch (Exception ignored) {}
                            }
                        }

                        // 3. Handle Live Images (PRIORITY: Show in WebView if active)
                        if (isLiveFirebaseActive || !isLocalAvailable) {
                            Object imgValue = dataSnapshot.child("last_image").getValue();
                            if (imgValue != null) {
                                String base64Image = imgValue.toString();
                                runOnUiThread(() -> {
                                    if (binding.videoStream.getVisibility() != View.VISIBLE) {
                                        binding.videoStream.setVisibility(View.VISIBLE);
                                        binding.videoSurface.setVisibility(View.GONE);
                                        binding.layoutStreamStats.setVisibility(View.GONE);
                                    }
                                    String html = "<html><body style='margin:0;padding:0;background:black;display:flex;justify-content:center;align-items:center;'><img src='data:image/jpeg;base64," + base64Image + "' style='width:100%;height:auto;max-height:100%;object-fit:contain;'/></body></html>";
                                    binding.videoStream.loadDataWithBaseURL(null, html, "text/html", "UTF-8", null);
                                });
                            }
                        }
                    } else {
                        addLog("Node not found: " + room);
                    }
                }

                @Override
                public void onCancelled(DatabaseError databaseError) {
                    addLog("Firebase Error: " + databaseError.getMessage());
                }
            };
            outboxRef.addValueEventListener(firebaseListener);
            
            runOnUiThread(() -> {
                binding.syncStatus.setText("Cloud Sync: Active");
                binding.syncStatus.setTextColor(Color.parseColor("#4CAF50"));
            });
        } catch (Exception e) {
            addLog("Firebase Init Failed: " + e.getMessage());
        }
    }

    private void checkLocalConnectivityAndSetupStream(String ipAddress) {
        executor.execute(() -> {
            boolean available = false;
            try {
                URL url = new URL("http://" + ipAddress + "/status");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(1500);
                conn.setReadTimeout(1500);
                int code = conn.getResponseCode();
                if (code == 200) {
                    available = true;
                }
                conn.disconnect();
            } catch (Exception e) {
                available = false;
            }

            isLocalAvailable = available;
            runOnUiThread(() -> {
                if (isLocalAvailable) {
                    addLog("Local connectivity verified. Starting native stream...");
                    startNativeStream(ipAddress);
                    startLocalHeartbeat(ipAddress);
                } else {
                    addLog("Local connectivity unavailable. Checking Firebase image...");
                    stopNativeStream();
                }
            });
        });
    }

    private void startNativeStream(String ip) {
        if (isNativeStreaming.get()) return;
        isNativeStreaming.set(true);
        totalBytesReceived = 0;
        frameCount = 0;
        lastStatsTime = System.currentTimeMillis();
        lastByteCount = 0;

        runOnUiThread(() -> {
            binding.videoStream.setVisibility(View.GONE);
            binding.videoSurface.setVisibility(View.VISIBLE);
            binding.layoutStreamStats.setVisibility(View.VISIBLE);
        });

        executor.execute(() -> {
            HttpURLConnection conn = null;
            try {
                URL url = new URL("http://" + ip + ":81/stream");
                conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(5000);
                conn.setReadTimeout(5000);
                
                InputStream is = conn.getInputStream();
                byte[] buffer = new byte[16384];
                ByteArrayOutputStream frameBuffer = new ByteArrayOutputStream();
                boolean inFrame = false;
                int lastByte = -1;
                
                while (isNativeStreaming.get()) {
                    int bytesRead = is.read(buffer);
                    if (bytesRead <= 0) break;
                    
                    totalBytesReceived += bytesRead;
                    
                    for (int i = 0; i < bytesRead; i++) {
                        int b = buffer[i] & 0xFF;
                        
                        if (!inFrame) {
                            if (lastByte == 0xFF && b == 0xD8) {
                                inFrame = true;
                                frameBuffer.reset();
                                frameBuffer.write(0xFF);
                                frameBuffer.write(0xD8);
                            }
                        } else {
                            frameBuffer.write(b);
                            if (lastByte == 0xFF && b == 0xD9) {
                                renderFrame(frameBuffer.toByteArray());
                                frameCount++;
                                inFrame = false;
                                updateStats();
                            }
                        }
                        lastByte = b;
                    }
                }
            } catch (Exception e) {
                Log.e("MainDoor", "Native stream error", e);
                addLog("Stream Error: " + e.getMessage());
            } finally {
                if (conn != null) conn.disconnect();
                isNativeStreaming.set(false);
            }
        });
    }

    private void stopNativeStream() {
        isNativeStreaming.set(false);
        runOnUiThread(() -> {
            binding.videoSurface.setVisibility(View.GONE);
            binding.layoutStreamStats.setVisibility(View.GONE);
            binding.videoStream.setVisibility(View.VISIBLE);
        });
    }

    private void renderFrame(byte[] data) {
        if (savedSurface == null) return;
        try {
            Bitmap bitmap = BitmapFactory.decodeByteArray(data, 0, data.length);
            if (bitmap == null) return;

            Canvas canvas = binding.videoSurface.getHolder().lockCanvas();
            if (canvas != null) {
                canvas.drawColor(Color.BLACK);
                float scale = Math.min((float)canvas.getWidth() / bitmap.getWidth(), 
                                     (float)canvas.getHeight() / bitmap.getHeight());
                int w = (int)(bitmap.getWidth() * scale);
                int h = (int)(bitmap.getHeight() * scale);
                int left = (canvas.getWidth() - w) / 2;
                int top = (canvas.getHeight() - h) / 2;
                Rect scaledDest = new Rect(left, top, left + w, top + h);
                canvas.drawBitmap(bitmap, null, scaledDest, null);
                binding.videoSurface.getHolder().unlockCanvasAndPost(canvas);
            }
            bitmap.recycle();
        } catch (Exception e) {
            Log.e("MainDoor", "Render error", e);
        }
    }

    private void updateStats() {
        long now = System.currentTimeMillis();
        long delta = now - lastStatsTime;
        if (delta >= 1000) {
            double fps = (double) frameCount * 1000.0 / delta;
            double kbps = (double) (totalBytesReceived - lastByteCount) * 8.0 / delta;
            
            lastStatsTime = now;
            lastByteCount = totalBytesReceived;
            frameCount = 0;

            runOnUiThread(() -> {
                binding.tvStreamThroughput.setText(String.format(Locale.US, "Rate: %.1f kbps", kbps));
                binding.tvStreamFps.setText(String.format(Locale.US, "FPS: %.1f", fps));
            });
        }
    }

    private void startLocalHeartbeat(String ipAddress) {
        executor.execute(() -> {
            addLog("Local Heartbeat Active");
            while (!isFinishing() && isLocalAvailable) {
                try {
                    URL url = new URL("http://" + ipAddress + "/status");
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setConnectTimeout(2000);
                    int code = conn.getResponseCode();
                    conn.disconnect();
                    if (code != 200) {
                        Log.w("MainDoor", "Heartbeat missed: " + code);
                    }
                } catch (Exception e) {
                    Log.e("MainDoor", "Heartbeat error", e);
                }
                try { Thread.sleep(5000); } catch (InterruptedException ignored) {}
            }
        });
    }

    private void setupNetworkListener() {
        connectivityManager = (ConnectivityManager) getSystemService(android.content.Context.CONNECTIVITY_SERVICE);
        networkCallback = new ConnectivityManager.NetworkCallback() {
            @Override
            public void onAvailable(Network network) {
                if (syncMode == 0) {
                    addLog("Network Restored. Re-initializing...");
                    initMqtt();
                }
            }
            @Override
            public void onLost(Network network) {
                if (syncMode == 0) addLog("Network Lost!");
            }
        };
        NetworkRequest request = new NetworkRequest.Builder()
                .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                .build();
        connectivityManager.registerNetworkCallback(request, networkCallback);
    }

    private void unregisterNetworkListener() {
        if (connectivityManager != null && networkCallback != null) {
            try {
                connectivityManager.unregisterNetworkCallback(networkCallback);
            } catch (Exception ignored) {}
        }
    }

    private void setupWebView() {
        WebSettings webSettings = binding.videoStream.getSettings();
        webSettings.setJavaScriptEnabled(true);
        webSettings.setUseWideViewPort(true);
        webSettings.setLoadWithOverviewMode(true);
        
        binding.videoStream.setWebViewClient(new WebViewClient() {
            @Override
            public void onReceivedError(WebView view, android.webkit.WebResourceRequest request, android.webkit.WebResourceError error) {
                super.onReceivedError(view, request, error);
                if (request.isForMainFrame() && request.getUrl().toString().contains("/stream")) {
                    addLog("Stream Error: " + error.getDescription());
                    showRebootDialog("Stream connection failed. Would you like to reboot the camera?");
                }
            }

            @Override
            public void onReceivedHttpError(WebView view, android.webkit.WebResourceRequest request, android.webkit.WebResourceResponse errorResponse) {
                super.onReceivedHttpError(view, request, errorResponse);
                if (request.isForMainFrame() && request.getUrl().toString().contains("/stream")) {
                    addLog("Stream HTTP Error: " + errorResponse.getStatusCode());
                    showRebootDialog("Stream server returned error " + errorResponse.getStatusCode() + ". Reboot camera?");
                }
            }
        });
        
        if (syncMode == 1) {
            String ip = prefs.getString("local_node_ip", AppDefaults.DEFAULT_NODE_IP);
            binding.videoStream.loadUrl("http://" + ip + "/stream");
        } else {
            // MJPEG over MQTT not directly supported in WebView. 
            // In a real app, we'd use a custom view to decode MJPEG from MQTT.
            binding.videoStream.loadData("<html><body style='background:black;color:white;display:flex;justify-content:center;align-items:center;'>MJPEG via MQTT not implemented in WebView</body></html>", "text/html", "UTF-8");
        }
    }

    private void showRebootDialog(String message) {
        runOnUiThread(() -> {
            new androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("Camera Issue Detected")
                .setMessage(message)
                .setPositiveButton("Reboot Now", (dialog, which) -> sendCommand("REBOOT"))
                .setNegativeButton("Ignore", null)
                .show();
        });
    }

    private void startIpSync() {
        executor.execute(() -> {
            while (!isFinishing()) {
                try {
                    String ip = prefs.getString("local_node_ip", AppDefaults.DEFAULT_NODE_IP);
                    URL url = new URL("http://" + ip + "/status");
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setConnectTimeout(2000);
                    if (conn.getResponseCode() == 200) {
                        BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                        StringBuilder sb = new StringBuilder();
                        String line;
                        while ((line = reader.readLine()) != null) sb.append(line);
                        handleStatus(sb.toString());
                    }
                    conn.disconnect();
                } catch (Exception e) {
                    Log.e("MainDoor", "IP sync failed", e);
                }
                try { Thread.sleep(3000); } catch (InterruptedException ignored) {}
            }
        });
    }

    private void initMqtt() {
        closeMqtt();
        String portsStr = prefs.getString("mqtt_ports", getString(R.string.default_mqtt_port));
        String[] portStrings = portsStr.split(",");
        mqttPorts = new int[portStrings.length];
        for (int i = 0; i < portStrings.length; i++) {
            try {
                mqttPorts[i] = Integer.parseInt(portStrings[i].trim());
            } catch (Exception e) {
                mqttPorts[i] = 1883;
            }
        }
        currentPortIndex = 0;
        logBuilder.setLength(0);
        addLog("Initializing MQTT Connection...");
        tryConnectNextPort();
    }

    private void addLog(String message) {
        String time = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
        String logEntry = "[" + time + "] " + message + "\n";
        logBuilder.insert(0, logEntry);
        runOnUiThread(() -> {
            binding.tvConnectionLogs.setText(logBuilder.toString());
            binding.logScrollView.fullScroll(android.view.View.FOCUS_UP);
        });
    }

    private void tryConnectNextPort() {
        if (currentPortIndex >= mqttPorts.length) {
            runOnUiThread(() -> {
                binding.syncStatus.setTextColor(Color.RED);
                binding.syncStatus.setText("Network Connection Failed.");
            });
            addLog("ERROR: All ports failed.");
            return;
        }

        int port = mqttPorts[currentPortIndex];
        String broker = prefs.getString("mqtt_broker", "broker.hivemq.com");
        String clientId = "Android_MainDoor_" + System.currentTimeMillis();

        runOnUiThread(() -> binding.syncStatus.setText("Connecting to Port " + port + "..."));
        addLog("Attempting connection to " + broker + ":" + port);

        executor.execute(() -> {
            try {
                closeMqtt();
                String brokerUri;
                if (port == 8883) brokerUri = "ssl://" + broker + ":" + port;
                else if (port == 8884) brokerUri = "wss://" + broker + ":" + port + "/mqtt";
                else if (port == 8000) brokerUri = "ws://" + broker + ":" + port + "/mqtt";
                else brokerUri = "tcp://" + broker + ":" + port;

                mqttClient = new MqttClient(brokerUri, clientId, new MemoryPersistence());
                MqttConnectOptions options = new MqttConnectOptions();
                options.setAutomaticReconnect(true);
                options.setCleanSession(true);
                options.setConnectionTimeout(10);

                mqttClient.setCallback(new MqttCallback() {
                    @Override public void connectionLost(Throwable cause) {
                        runOnUiThread(() -> binding.syncStatus.setText("Connection Lost. Retrying..."));
                        addLog("Connection Lost: " + (cause != null ? cause.getMessage() : "Unknown"));
                    }
                    @Override public void messageArrived(String topic, MqttMessage message) {
                        if (topic.equals(statusTopic)) {
                            String payload = new String(message.getPayload());
                            handleStatus(payload);
                        } else if (topic.equals(imageTopic)) {
                            saveMqttImage(message.getPayload());
                        }
                    }
                    @Override public void deliveryComplete(IMqttDeliveryToken token) {}
                });

                mqttClient.connect(options);
                mqttClient.subscribe(statusTopic);
                mqttClient.subscribe(imageTopic);

                runOnUiThread(() -> {
                    binding.syncStatus.setText("Connected (Port " + port + ")");
                    binding.syncStatus.setTextColor(Color.parseColor("#4CAF50"));
                });
                addLog("SUCCESS: Connected to port " + port);

            } catch (Exception e) {
                final String errorMsg = e.getMessage() != null ? e.getMessage() : "Timeout";
                addLog("FAILED: Port " + port + " - " + errorMsg);
                currentPortIndex++;
                tryConnectNextPort();
            }
        });
    }

    private void closeMqtt() {
        if (mqttClient != null) {
            try {
                if (mqttClient.isConnected()) mqttClient.disconnect();
                mqttClient.close();
            } catch (Exception ignored) {}
            mqttClient = null;
        }
    }

    private void saveMqttImage(byte[] data) {
        executor.execute(() -> {
            try {
                String subDir = prefs.getString("door_path", getString(R.string.default_door_path));
                File directory = StorageUtils.getWorkDir(subDir);
                String fileName = "IMG_DOOR_" + System.currentTimeMillis() + ".jpg";
                File file = new File(directory, fileName);
                java.io.FileOutputStream fos = new java.io.FileOutputStream(file);
                fos.write(data);
                fos.close();
                runOnUiThread(() -> Toast.makeText(this, "Image received and saved", Toast.LENGTH_SHORT).show());
            } catch (Exception e) {
                Log.e("MainDoor", "Failed to save MQTT image", e);
            }
        });
    }

    private void handleStatus(String payload) {
        runOnUiThread(() -> {
            try {
                JSONObject json = new JSONObject(payload);

                // Update Node Info
                if (json.has("ip")) {
                    String ip = json.getString("ip");
                    String id = json.optString("id", "Node");
                    if (!binding.availableNodesInfo.getText().toString().contains(ip)) {
                        binding.availableNodesInfo.setText(" (" + ip + ")");
                    }
                    String entry = id + " (" + ip + ")";
                    if (!discoveredNodes.contains(entry)) {
                        discoveredNodes.add(entry);
                        nodeAdapter.notifyDataSetChanged();
                    }
                }

                binding.rowPir.value.setText(json.getBoolean("pir") ? "Motion" : "Clear");
                binding.rowDoor.value.setText(json.getBoolean("door") ? "Open" : "Closed");
                binding.rowBell.value.setText(json.getBoolean("bell") ? "Pressed" : "Idle");
                binding.rowLdr.value.setText(json.getBoolean("ldr") ? "Low" : "Good");
                
                // Only update switches AND text values if not recently interacted with (within 5 seconds)
                if (System.currentTimeMillis() - lastCommandTime > 5000) {
                    boolean relay = json.getBoolean("relay");
                    binding.rowRelay.value.setText(relay ? "ON" : "OFF");
                    binding.rowRelay.sensorSwitch.setChecked(relay);

                    boolean buzzer = json.getBoolean("buzzer");
                    binding.rowBuzzer.value.setText(buzzer ? "ON" : "OFF");
                    binding.rowBuzzer.sensorSwitch.setChecked(buzzer);

                    boolean flash = json.getBoolean("flash");
                    binding.rowFlash.value.setText(flash ? "ON" : "OFF");
                    binding.rowFlash.sensorSwitch.setChecked(flash);
                }
                
                binding.rowTemp.value.setText(String.format("%.1f °C", json.getDouble("temp")));
                binding.rowPres.value.setText(String.format("%.1f hPa", json.getDouble("pres")));
                
                String nodeTs = json.optString("ts", "N/A");
                String appTs = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
                binding.syncStatus.setText("Node: " + nodeTs + " | App: " + appTs);
                binding.syncStatus.setTextColor(Color.parseColor("#4CAF50"));
                
                addHistoryRow(json);
            } catch (Exception ignored) {}
        });
    }

    private void addHistoryRow(JSONObject json) {
        try {
            TableRow row = new TableRow(this);
            String time = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
            
            String event = "";
            if (json.getBoolean("pir")) event += "Motion ";
            if (json.getBoolean("door")) event += "DoorOpen ";
            if (json.getBoolean("bell")) event += "Bell ";
            if (event.isEmpty()) event = "Periodic";

            String[] cols = {time, event, json.getString("relay")};
            for (String col : cols) {
                TextView tv = new TextView(this);
                tv.setText(col);
                tv.setPadding(10, 5, 10, 5);
                tv.setTextColor(Color.WHITE);
                row.addView(tv);
            }
            binding.historyTable.addView(row, 1);
        } catch (Exception ignored) {}
    }

    private void sendCommand(String cmd) {
        lastCommandTime = System.currentTimeMillis();
        if (syncMode == 1) {
            executor.execute(() -> {
                try {
                    String ip = selectedNodeIp.isEmpty() ? prefs.getString("local_node_ip", AppDefaults.DEFAULT_NODE_IP) : selectedNodeIp;
                    addLog("Sending IP Cmd [" + cmd + "] to " + ip);
                    URL url = new URL("http://" + ip + "/control?cmd=" + cmd);
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setConnectTimeout(2000);
                    int resp = conn.getResponseCode();
                    conn.disconnect();
                    if (resp == 200) {
                        addLog("IP Cmd Success");
                    } else {
                        addLog("IP Cmd Failed: " + resp);
                    }
                } catch (Exception e) {
                    runOnUiThread(() -> Toast.makeText(this, "Failed to send IP command", Toast.LENGTH_SHORT).show());
                }
            });
        } else if (syncMode == 2) {
            String doorNode = prefs.getString("firebase_door_node", AppDefaults.NODE_DOOR);
            String room = doorNode;
            if (room.contains("/")) room = room.substring(room.lastIndexOf("/") + 1);

            String commandNode = "FrmMobile/" + room;

            if (firebaseDatabase != null) {
                firebaseDatabase.getReference(commandNode).child("command")
                        .setValue(cmd)
                        .addOnCompleteListener(task -> {
                            if (task.isSuccessful()) addLog("Firebase Cmd [" + cmd + "] Sent");
                            else addLog("Firebase Cmd FAILED: " + (task.getException() != null ? task.getException().getMessage() : "Unknown"));
                        });
            } else {
                FirebaseDatabase.getInstance().getReference(commandNode).child("command")
                        .setValue(cmd);
            }
        } else {
            executor.execute(() -> {
                try {
                    if (mqttClient != null && mqttClient.isConnected()) {
                        mqttClient.publish(cmdTopic, new MqttMessage(cmd.getBytes()));
                    }
                } catch (Exception e) {
                    runOnUiThread(() -> Toast.makeText(this, "Failed to send MQTT command", Toast.LENGTH_SHORT).show());
                }
            });
        }
    }

    private void showCameraConfigDialog() {
        android.widget.ScrollView scrollView = new android.widget.ScrollView(this);
        android.widget.LinearLayout layout = new android.widget.LinearLayout(this);
        layout.setOrientation(android.widget.LinearLayout.VERTICAL);
        layout.setPadding(50, 40, 50, 40);
        scrollView.addView(layout);

        // 1. Resolution Picker
        addDialogLabel(layout, "Video Resolution (Frame Size):");
        String[] resolutions = {"CIF (400x296)", "QVGA (320x240)", "VGA (640x480)", "SVGA (800x600)", "XGA (1024x768)", "HD (1280x720)", "UXGA (1600x1200)"};
        int[] resValues = {4, 5, 8, 9, 10, 11, 13};
        android.widget.Spinner resSpinner = new android.widget.Spinner(this);
        resSpinner.setAdapter(new android.widget.ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, resolutions));
        layout.addView(resSpinner);

        // 2. JPEG Quality
        addDialogLabel(layout, "\nJPEG Quality (10=Best, 63=Worst):");
        com.google.android.material.slider.Slider qSlider = createDialogSlider(layout, 10, 63, 1, 12);

        // 3. Brightness & Contrast
        addDialogLabel(layout, "\nBrightness (-2 to 2):");
        com.google.android.material.slider.Slider brSlider = createDialogSlider(layout, -2, 2, 1, 0);

        addDialogLabel(layout, "\nContrast (-2 to 2):");
        com.google.android.material.slider.Slider ctSlider = createDialogSlider(layout, -2, 2, 1, 0);

        // 4. Orientation
        android.widget.Switch swMirror = new android.widget.Switch(this);
        swMirror.setText("Horizontal Mirror");
        swMirror.setPadding(0, 20, 0, 20);
        layout.addView(swMirror);

        android.widget.Switch swFlip = new android.widget.Switch(this);
        swFlip.setText("Vertical Flip");
        swFlip.setPadding(0, 20, 0, 20);
        layout.addView(swFlip);

        // 5. Special Effects
        addDialogLabel(layout, "\nSpecial Effect:");
        String[] effects = {"None", "Negative", "Grayscale", "Red Tint", "Green Tint", "Blue Tint", "Sepia"};
        android.widget.Spinner fxSpinner = new android.widget.Spinner(this);
        fxSpinner.setAdapter(new android.widget.ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, effects));
        layout.addView(fxSpinner);

        new androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Professional Camera Controls")
            .setView(scrollView)
            .setPositiveButton("Apply All", (dialog, which) -> {
                int resIdx = resSpinner.getSelectedItemPosition();
                sendCommand("CAM_SET:framesize:" + resValues[resIdx]);
                sendCommand("CAM_SET:quality:" + (int)qSlider.getValue());
                sendCommand("CAM_SET:brightness:" + (int)brSlider.getValue());
                sendCommand("CAM_SET:contrast:" + (int)ctSlider.getValue());
                sendCommand("CAM_SET:hmirror:" + (swMirror.isChecked() ? 1 : 0));
                sendCommand("CAM_SET:vflip:" + (swFlip.isChecked() ? 1 : 0));
                sendCommand("CAM_SET:special_effect:" + fxSpinner.getSelectedItemPosition());
                
                Toast.makeText(this, "Batch configuration sent to camera", Toast.LENGTH_SHORT).show();
            })
            .setNegativeButton("Cancel", null)
            .show();
    }

    private void addDialogLabel(android.widget.LinearLayout layout, String text) {
        android.widget.TextView label = new android.widget.TextView(this);
        label.setText(text);
        label.setTextColor(Color.DKGRAY);
        label.setTextSize(14);
        layout.addView(label);
    }

    private com.google.android.material.slider.Slider createDialogSlider(android.widget.LinearLayout layout, float min, float max, float step, float val) {
        com.google.android.material.slider.Slider slider = new com.google.android.material.slider.Slider(this);
        slider.setValueFrom(min);
        slider.setValueTo(max);
        slider.setStepSize(step);
        slider.setValue(val);
        layout.addView(slider);
        return slider;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        unregisterNetworkListener();
        if (outboxRef != null && firebaseListener != null) {
            outboxRef.removeEventListener(firebaseListener);
        }
        executor.shutdown();
        closeMqtt();
    }
}
