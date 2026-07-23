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

public class SecurityMainDoorActivity extends AppCompatActivity {

    private ActivitySecurityMainDoorBinding binding;
    private MqttClient mqttClient;
    private DatabaseReference firebaseRef;
    private ValueEventListener firebaseListener;
    private final ExecutorService executor = Executors.newFixedThreadPool(2);
    private final String statusTopic = "FrmEsp32/Securitymaindoor/status";
    private final String cmdTopic = "FrmMobile/esp32cam/Securitymaindoor/command";
    private final String imageTopic = "FrmEsp32/Securitymaindoor/image";
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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivitySecurityMainDoorBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        prefs = getSharedPreferences("SmartHomePrefs", MODE_PRIVATE);
        syncMode = prefs.getInt("sync_mode", 0);

        binding.btnBack.setOnClickListener(v -> finish());
        binding.btnSettings.setOnClickListener(v -> {
            Intent intent = new Intent(this, SmartHomeSettingsActivity.class);
            intent.putExtra("caller_context", "door");
            startActivity(intent);
        });
        
        binding.btnCapture.setOnClickListener(v -> sendCommand("CAPTURE"));
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

        // Quick Controls
        binding.btnQuickRelay.setOnClickListener(v -> {
            boolean currentState = binding.rowRelay.sensorSwitch.isChecked();
            sendCommand(currentState ? "RELAY_OFF" : "RELAY_ON");
        });
        binding.btnQuickBuzzer.setOnClickListener(v -> {
            boolean currentState = binding.rowBuzzer.sensorSwitch.isChecked();
            sendCommand(currentState ? "BUZZER_OFF" : "BUZZER_ON");
        });
        binding.btnQuickFlash.setOnClickListener(v -> {
            boolean currentState = binding.rowFlash.sensorSwitch.isChecked();
            sendCommand(currentState ? "FLASH_OFF" : "FLASH_ON");
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
        String node = prefs.getString("firebase_door_node", AppDefaults.NODE_DOOR);
        
        addLog("Authenticating Firebase...");
        String email = prefs.getString("firebase_email", Credentials.FIREBASE_EMAIL);
        String password = prefs.getString("firebase_password", Credentials.FIREBASE_PASSWORD);
        
        if (email.isEmpty() || password.isEmpty()) {
            connectToFirebase(url, node);
            return;
        }

        FirebaseAuth.getInstance().signInWithEmailAndPassword(email, password)
            .addOnCompleteListener(task -> {
                if (task.isSuccessful()) {
                    addLog("Auth Success. Connecting to Database...");
                    connectToFirebase(url, node);
                } else {
                    addLog("Auth Failed: " + (task.getException() != null ? task.getException().getMessage() : "Unknown"));
                }
            });
    }

    private void connectToFirebase(String url, String node) {
        try {
            FirebaseDatabase database = FirebaseDatabase.getInstance(url);
            firebaseRef = database.getReference(node);
            
            firebaseListener = new ValueEventListener() {
                @Override
                public void onDataChange(DataSnapshot dataSnapshot) {
                    if (dataSnapshot.exists()) {
                        Object value = dataSnapshot.child("status").getValue();
                        if (value != null) {
                            handleStatus(value.toString());
                            
                            // Dynamically extract ESP32 Cam IP from Firebase status
                            try {
                                JSONObject json = new JSONObject(value.toString());
                                if (json.has("ip")) {
                                    String espIp = json.getString("ip");
                                    checkLocalConnectivityAndSetupStream(espIp);
                                }
                            } catch (Exception ignored) {}
                        }
                    }
                }

                @Override
                public void onCancelled(DatabaseError databaseError) {
                    addLog("Firebase Error: " + databaseError.getMessage());
                }
            };
            firebaseRef.addValueEventListener(firebaseListener);
            
            runOnUiThread(() -> {
                binding.syncStatus.setText("Firebase Connected");
                binding.syncStatus.setTextColor(Color.parseColor("#4CAF50"));
            });
        } catch (Exception e) {
            addLog("Firebase Init Failed: " + e.getMessage());
        }
    }

    private void checkLocalConnectivityAndSetupStream(String ipAddress) {
        executor.execute(() -> {
            boolean isLocalAvailable = false;
            try {
                URL url = new URL("http://" + ipAddress + "/status");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(1500);
                conn.setReadTimeout(1500);
                int code = conn.getResponseCode();
                if (code == 200) {
                    isLocalAvailable = true;
                }
                conn.disconnect();
            } catch (Exception e) {
                isLocalAvailable = false;
            }

            final boolean useLocal = isLocalAvailable;
            runOnUiThread(() -> {
                if (useLocal) {
                    addLog("Local connectivity verified. Starting local stream...");
                    binding.videoStream.loadUrl("http://" + ipAddress + "/stream");
                } else {
                    addLog("Local connectivity unavailable. Showing Firebase placeholder stream.");
                    binding.videoStream.loadData("<html><body style='background:black;color:white;display:flex;justify-content:center;align-items:center;'>Streaming over cloud via Firebase node status active.</body></html>", "text/html", "UTF-8");
                }
            });
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
        binding.videoStream.setWebViewClient(new WebViewClient());
        
        if (syncMode == 1) {
            String ip = prefs.getString("local_node_ip", AppDefaults.DEFAULT_NODE_IP);
            binding.videoStream.loadUrl("http://" + ip + "/stream");
        } else {
            // MJPEG over MQTT not directly supported in WebView. 
            // In a real app, we'd use a custom view to decode MJPEG from MQTT.
            binding.videoStream.loadData("<html><body style='background:black;color:white;display:flex;justify-content:center;align-items:center;'>MJPEG via MQTT not implemented in WebView</body></html>", "text/html", "UTF-8");
        }
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
                
                boolean relay = json.getBoolean("relay");
                binding.rowRelay.value.setText(relay ? "ON" : "OFF");
                binding.rowRelay.sensorSwitch.setChecked(relay);

                boolean buzzer = json.getBoolean("buzzer");
                binding.rowBuzzer.value.setText(buzzer ? "ON" : "OFF");
                binding.rowBuzzer.sensorSwitch.setChecked(buzzer);

                boolean flash = json.getBoolean("flash");
                binding.rowFlash.value.setText(flash ? "ON" : "OFF");
                binding.rowFlash.sensorSwitch.setChecked(flash);
                
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
        if (syncMode == 1) {
            executor.execute(() -> {
                try {
                    String ip = prefs.getString("local_node_ip", AppDefaults.DEFAULT_NODE_IP);
                    URL url = new URL("http://" + ip + "/control?cmd=" + cmd);
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.getResponseCode();
                    conn.disconnect();
                } catch (Exception e) {
                    runOnUiThread(() -> Toast.makeText(this, "Failed to send IP command", Toast.LENGTH_SHORT).show());
                }
            });
        } else if (syncMode == 2) {
            String commandNode = "FrmMobile/esp32cam/Securitymaindoor";
            FirebaseDatabase.getInstance().getReference(commandNode).child("command").setValue(cmd);
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

    @Override
    protected void onDestroy() {
        super.onDestroy();
        unregisterNetworkListener();
        if (firebaseRef != null && firebaseListener != null) {
            firebaseRef.removeEventListener(firebaseListener);
        }
        executor.shutdown();
        closeMqtt();
    }
}
