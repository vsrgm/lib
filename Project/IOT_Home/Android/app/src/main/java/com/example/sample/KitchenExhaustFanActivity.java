package com.example.sample;

import android.content.Intent;
import android.graphics.Color;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.NetworkRequest;
import android.os.Bundle;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.TableRow;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.sample.databinding.ActivityKitchenExhaustFanBinding;

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
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class KitchenExhaustFanActivity extends AppCompatActivity {

    private static final String TAG = "KitchenExhaustFanActivity";
    private ActivityKitchenExhaustFanBinding binding;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private DatabaseReference firebaseRef;
    private FirebaseDatabase firebaseDatabase;
    private ValueEventListener firebaseListener;
    private boolean isHistoryExpanded = false;
    private android.content.SharedPreferences prefs;
    private MqttClient mqttClient;
    private int syncMode = 0; // 0: MQTT, 1: IP, 2: Firebase
    private final List<String> discoveredNodes = new ArrayList<>();
    private ArrayAdapter<String> nodeAdapter;
    private final List<String> mqttHistoryBuffer = new ArrayList<>();
    private String selectedNodeId = "";
    private String selectedNodeIp = "";
    private final StringBuilder logBuilder = new StringBuilder();
    private ConnectivityManager connectivityManager;
    private ConnectivityManager.NetworkCallback networkCallback;
    private boolean isSyncing = false;
    private long lastInteractionTime = 0;
    private long connectionAttemptId = 0;

    private int[] mqttPorts = {};
    private int currentPortIndex = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        android.content.SharedPreferences themePrefs = getSharedPreferences("ThemePrefs", MODE_PRIVATE);
        int themeMode = themePrefs.getInt("theme_mode", 0);
        switch (themeMode) {
            case 1: androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_NO); break;
            case 2: androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_YES); break;
            default: androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM); break;
        }

        super.onCreate(savedInstanceState);
        binding = ActivityKitchenExhaustFanBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        prefs = getSharedPreferences("SmartHomePrefs", MODE_PRIVATE);
        syncMode = prefs.getInt("sync_mode", 0);

        nodeAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, discoveredNodes);
        nodeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        binding.nodeSelector.setAdapter(nodeAdapter);
        
        loadSettings();
        binding.nodeSelector.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(android.widget.AdapterView<?> parent, android.view.View view, int position, long id) {
                String selected = discoveredNodes.get(position);
                if (selected.contains("(") && selected.endsWith(")")) {
                    selectedNodeId = selected.substring(0, selected.indexOf(" (")).trim();
                    selectedNodeIp = selected.substring(selected.lastIndexOf("(") + 1, selected.length() - 1);
                    if (selectedNodeId.startsWith("Local")) selectedNodeId = "";
                }
            }
            @Override
            public void onNothingSelected(android.widget.AdapterView<?> parent) {}
        });

        if (syncMode == 2) {
            initFirebase();
        } else {
            initMqtt();
            setupNetworkListener();
        }

        // Setup Rows
        binding.rowFanStatus.label.setText("Exhaust Fan");
        binding.rowFanStatus.sensorSwitch.setVisibility(android.view.View.VISIBLE);
        binding.rowFanStatus.sensorSwitch.setEnabled(false);
        
        binding.rowBuzzerStatus.label.setText("Buzzer");
        binding.rowBuzzerStatus.sensorSwitch.setVisibility(android.view.View.VISIBLE);
        binding.rowBuzzerStatus.sensorSwitch.setEnabled(false);

        binding.rowMq2A.label.setText("Gas Level (Analog)");
        binding.rowMq2D.label.setText("Smoke Detection");
        binding.rowLdrStatus.label.setText("LDR Light Sensor");
        
        binding.rowUserSwitch.label.setText("Manual Toggle Switch");
        
        binding.rowManual.label.setText("Manual Override");
        binding.rowManual.sensorSwitch.setVisibility(android.view.View.VISIBLE);

        binding.btnBack.setOnClickListener(v -> finish());
        binding.btnSync.setOnClickListener(v -> syncData());
        binding.btnSyncHistory.setOnClickListener(v -> syncHistory());
        binding.btnClearHistory.setOnClickListener(v -> clearHistory());
        binding.btnSettings.setOnClickListener(v -> {
            Intent intent = new Intent(this, SmartHomeSettingsActivity.class);
            intent.putExtra("caller_context", "kitchen_fan");
            startActivity(intent);
        });

        binding.historyHeader.setOnClickListener(v -> toggleHistoryExpansion());

        binding.rowFanStatus.sensorSwitch.setOnClickListener(v -> updateSensorConfig());
        binding.rowBuzzerStatus.sensorSwitch.setOnClickListener(v -> updateSensorConfig());
        
        binding.rowManual.sensorSwitch.setOnClickListener(v -> {
            boolean isChecked = binding.rowManual.sensorSwitch.isChecked();
            binding.rowFanStatus.sensorSwitch.setEnabled(isChecked);
            binding.rowBuzzerStatus.sensorSwitch.setEnabled(isChecked);
            updateSensorConfig();
        });
    }

    private void initFirebase() {
        String url = prefs.getString("firebase_url", AppDefaults.FIREBASE_URL);
        String roomSetting = prefs.getString("firebase_kitchen_fan_node", AppDefaults.NODE_KITCHEN_FAN);
        if (roomSetting.startsWith("FrmMobile/")) roomSetting = roomSetting.substring(10);
        
        final String room = roomSetting;
        addLog("Authenticating Firebase...");
        String email = prefs.getString("firebase_email", Credentials.FIREBASE_EMAIL);
        String password = prefs.getString("firebase_password", Credentials.FIREBASE_PASSWORD);
        
        if (email.isEmpty() || password.isEmpty()) {
            connectToFirebase(url, room);
            return;
        }

        FirebaseAuth.getInstance().signInWithEmailAndPassword(email, password)
            .addOnCompleteListener(task -> {
                if (task.isSuccessful()) {
                    addLog("Auth Success.");
                    connectToFirebase(url, room);
                } else {
                    addLog("Auth Failed: " + (task.getException() != null ? task.getException().getMessage() : "Unknown"));
                }
            });
    }

    private void connectToFirebase(String url, String room) {
        try {
            firebaseDatabase = FirebaseDatabase.getInstance(url);
            DatabaseReference outboxRef = firebaseDatabase.getReference("FrmNodeMcu").child(room);
            
            addLog("Listening: FrmNodeMcu/" + room);
            
            firebaseListener = new ValueEventListener() {
                @Override
                public void onDataChange(DataSnapshot dataSnapshot) {
                    if (dataSnapshot.exists()) {
                        DataSnapshot statusNode = dataSnapshot.child("status");
                        if (statusNode.exists()) {
                            // If it's a map, handle it directly. If it's a string, parse it.
                            Object val = statusNode.getValue();
                            if (val instanceof Map) {
                                handleStatusMap((Map<String, Object>) val);
                            } else if (val instanceof String) {
                                handleMqttStatus("firebase/status", (String) val);
                            }
                        }

                        DataSnapshot historyNode = dataSnapshot.child("history");
                        if (historyNode.exists()) {
                            mqttHistoryBuffer.clear();
                            for (DataSnapshot child : historyNode.getChildren()) {
                                Object entry = child.getValue();
                                if (entry != null) mqttHistoryBuffer.add(entry.toString());
                            }
                            updateHistoryTable(mqttHistoryBuffer);
                        }
                    }
                }

                @Override
                public void onCancelled(DatabaseError databaseError) {
                    addLog("Firebase Error: " + databaseError.getMessage());
                }
            };
            
            firebaseRef = outboxRef;
            outboxRef.addValueEventListener(firebaseListener);
            
            runOnUiThread(() -> {
                binding.syncStatus.setText("Cloud Sync: Active");
                binding.syncStatus.setTextColor(Color.parseColor("#4CAF50"));
            });
        } catch (Exception e) {
            addLog("Firebase Init Failed: " + e.getMessage());
        }
    }

    private void handleStatusMap(Map<String, Object> map) {
        if (System.currentTimeMillis() - lastInteractionTime < 5000) return;
        
        runOnUiThread(() -> {
            try {
                isSyncing = true;

                // IP and ID
                if (map.containsKey("ip")) {
                    String ip = String.valueOf(map.get("ip"));
                    String id = String.valueOf(map.getOrDefault("id", "Node"));
                    String currentInfo = binding.availableNodesInfo.getText().toString();
                    if (!currentInfo.contains(ip)) {
                        binding.availableNodesInfo.setText(" (" + ip + ")");
                    }
                    String entry = id + " (" + ip + ")";
                    if (!discoveredNodes.contains(entry)) {
                        discoveredNodes.add(entry);
                        nodeAdapter.notifyDataSetChanged();
                    }
                }

                // Fan
                if (map.containsKey("fan")) {
                    String fanState = String.valueOf(map.get("fan"));
                    boolean isOn = "ON".equalsIgnoreCase(fanState);
                    if (!binding.rowFanStatus.value.getText().toString().equals(fanState)) {
                        binding.rowFanStatus.value.setText(fanState);
                    }
                    if (binding.rowFanStatus.sensorSwitch.isChecked() != isOn) {
                        binding.rowFanStatus.sensorSwitch.setChecked(isOn);
                    }
                }

                // Buzzer
                if (map.containsKey("buzzer")) {
                    String buzzerState = String.valueOf(map.get("buzzer"));
                    boolean isOn = "ON".equalsIgnoreCase(buzzerState);
                    if (!binding.rowBuzzerStatus.value.getText().toString().equals(buzzerState)) {
                        binding.rowBuzzerStatus.value.setText(buzzerState);
                    }
                    if (binding.rowBuzzerStatus.sensorSwitch.isChecked() != isOn) {
                        binding.rowBuzzerStatus.sensorSwitch.setChecked(isOn);
                    }
                }

                // Sensors
                if (map.containsKey("mq2_a")) {
                    binding.rowMq2A.value.setText(String.valueOf(map.get("mq2_a")));
                }
                if (map.containsKey("mq2_d")) {
                    String smoke = String.valueOf(map.get("mq2_d"));
                    binding.rowMq2D.value.setText(smoke);
                    binding.rowMq2D.value.setTextColor("SMOKE".equalsIgnoreCase(smoke) ? Color.RED : Color.parseColor("#4CAF50"));
                }
                if (map.containsKey("ldr")) {
                    String ldrState = String.valueOf(map.get("ldr"));
                    binding.rowLdrStatus.value.setText(ldrState);
                    binding.rowLdrStatus.value.setTextColor("DARK".equalsIgnoreCase(ldrState) ? Color.RED : Color.parseColor("#4CAF50"));
                }

                if (map.containsKey("user_sw")) {
                    binding.rowUserSwitch.value.setText(String.valueOf(map.get("user_sw")));
                }
                
                // Manual
                if (map.containsKey("manual")) {
                    String manualState = String.valueOf(map.get("manual"));
                    boolean isManual = "ON".equalsIgnoreCase(manualState) || "true".equalsIgnoreCase(manualState);
                    
                    if (!binding.rowManual.value.getText().toString().equals(manualState)) {
                        binding.rowManual.value.setText(manualState);
                    }
                    if (binding.rowManual.sensorSwitch.isChecked() != isManual) {
                        binding.rowManual.sensorSwitch.setChecked(isManual);
                        binding.rowFanStatus.sensorSwitch.setEnabled(isManual);
                        binding.rowBuzzerStatus.sensorSwitch.setEnabled(isManual);
                    }
                }

                binding.syncStatus.setText("Last Update: " + new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date()));
                isSyncing = false;
            } catch (Exception e) {
                Log.e(TAG, "Map Parse Error: " + e.getMessage());
                isSyncing = false;
            }
        });
    }

    private void loadSettings() {
        syncMode = prefs.getInt("sync_mode", 0);
        String savedIp = prefs.getString("local_node_ip", AppDefaults.DEFAULT_NODE_IP);
        String localEntryPrefix = "Local IP (";

        boolean changed = false;
        for (int i = discoveredNodes.size() - 1; i >= 0; i--) {
            if (discoveredNodes.get(i).startsWith(localEntryPrefix)) {
                discoveredNodes.remove(i);
                changed = true;
            }
        }

        if (syncMode == 1 && !savedIp.isEmpty()) {
            String entry = localEntryPrefix + savedIp + ")";
            if (!discoveredNodes.contains(entry)) {
                discoveredNodes.add(0, entry);
                changed = true;
                selectedNodeIp = savedIp;
                selectedNodeId = "";
                runOnUiThread(() -> binding.nodeSelector.setSelection(0));
            }
        }

        if (changed) {
            nodeAdapter.notifyDataSetChanged();
        }
    }

    private void setupNetworkListener() {
        if (syncMode == 2) return;
        connectivityManager = (ConnectivityManager) getSystemService(android.content.Context.CONNECTIVITY_SERVICE);
        networkCallback = new ConnectivityManager.NetworkCallback() {
            @Override
            public void onAvailable(Network network) {
                addLog("Network Restored. Re-initializing...");
                initMqtt();
            }

            @Override
            public void onLost(Network network) {
                addLog("Network Lost!");
                runOnUiThread(() -> {
                    binding.syncStatus.setTextColor(Color.RED);
                    binding.syncStatus.setText("Network connection lost.");
                });
            }
        };

        NetworkRequest request = new NetworkRequest.Builder()
                .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                .build();
        connectivityManager.registerNetworkCallback(request, networkCallback);
    }

    private void updateSensorConfig() {
        if (isSyncing) return;
        lastInteractionTime = System.currentTimeMillis();
        try {
            JSONObject config = new JSONObject();
            config.put("fan", binding.rowFanStatus.sensorSwitch.isChecked());
            config.put("buzzer", binding.rowBuzzerStatus.sensorSwitch.isChecked());
            config.put("manual", binding.rowManual.sensorSwitch.isChecked());
            
            String payload = "CONFIG:" + config.toString();

            if (syncMode == 1 && !selectedNodeIp.isEmpty()) {
                executor.execute(() -> {
                    try {
                        URL url = new URL("http://" + selectedNodeIp + "/control?fan=" + (binding.rowFanStatus.sensorSwitch.isChecked() ? "ON" : "OFF") + "&buzzer=" + (binding.rowBuzzerStatus.sensorSwitch.isChecked() ? "ON" : "OFF") + "&manual=" + (binding.rowManual.sensorSwitch.isChecked() ? "true" : "false"));
                        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                        conn.setRequestMethod("GET");
                        conn.getResponseCode();
                        conn.disconnect();
                    } catch (Exception ignored) {}
                });
            } else if (syncMode == 2) {
                if (firebaseDatabase != null) {
                    String fbNode = prefs.getString("firebase_kitchen_fan_node", AppDefaults.NODE_KITCHEN_FAN);
                    firebaseDatabase.getReference(fbNode).child("command")
                        .setValue(payload);
                }
            } else if (isMqttAvailable()) {
                executor.execute(() -> {
                    try {
                        String targetTopic = selectedNodeId.isEmpty() ? "smart_home/all/commands" : "smart_home/" + selectedNodeId + "/commands";
                        mqttClient.publish(targetTopic, new MqttMessage(payload.getBytes()));
                    } catch (Exception ignored) {}
                });
            }
        } catch (Exception ignored) {}
    }

    private boolean isMqttAvailable() { return mqttClient != null && mqttClient.isConnected(); }

    private void toggleHistoryExpansion() {
        isHistoryExpanded = !isHistoryExpanded;
        binding.historyContainer.setVisibility(isHistoryExpanded ? android.view.View.VISIBLE : android.view.View.GONE);
        binding.historyHeader.setText(isHistoryExpanded ? R.string.history_collapse : R.string.history_expand);
    }

    private void syncHistory() {
        mqttHistoryBuffer.clear();
        if (syncMode == 1 && !selectedNodeIp.isEmpty()) {
            executor.execute(() -> {
                try {
                    URL url = new URL("http://" + selectedNodeIp + "/sync");
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    if (conn.getResponseCode() == 200) {
                        BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                        String line;
                        while ((line = reader.readLine()) != null) {
                            if (!line.trim().isEmpty()) mqttHistoryBuffer.add(line);
                        }
                        runOnUiThread(() -> updateHistoryTable(mqttHistoryBuffer));
                    }
                    conn.disconnect();
                } catch (Exception ignored) {}
            });
        } else if (syncMode == 2) {
            if (firebaseDatabase != null) {
                String fbNode = prefs.getString("firebase_kitchen_fan_node", "");
                firebaseDatabase.getReference(fbNode).child("command")
                    .setValue("HISTORY");
            }
        } else if (isMqttAvailable()) {
            executor.execute(() -> {
                try {
                    String targetTopic = selectedNodeId.isEmpty() ? "smart_home/all/commands" : "smart_home/" + selectedNodeId + "/commands";
                    mqttClient.publish(targetTopic, new MqttMessage("HISTORY".getBytes()));
                } catch (Exception ignored) {}
            });
        }
    }

    private void clearHistory() {
        if (syncMode == 1 && !selectedNodeIp.isEmpty()) {
            executor.execute(() -> {
                try {
                    URL url = new URL("http://" + selectedNodeIp + "/clear");
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.getResponseCode();
                    conn.disconnect();
                } catch (Exception ignored) {}
            });
        } else if (syncMode == 2) {
            if (firebaseDatabase != null) {
                String fbNode = prefs.getString("firebase_kitchen_fan_node", "");
                firebaseDatabase.getReference(fbNode).child("command")
                    .setValue("CLEAR");
            }
        } else if (isMqttAvailable()) {
            executor.execute(() -> {
                try {
                    String targetTopic = selectedNodeId.isEmpty() ? "smart_home/all/commands" : "smart_home/" + selectedNodeId + "/commands";
                    mqttClient.publish(targetTopic, new MqttMessage("CLEAR".getBytes()));
                } catch (Exception ignored) {}
            });
        }
    }

    private void initMqtt() {
        closeMqtt();
        connectionAttemptId++;
        final long currentId = connectionAttemptId;
        
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
        tryConnectNextPort(currentId);
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

    private void tryConnectNextPort(final long attemptId) {
        if (attemptId != connectionAttemptId) return;

        if (currentPortIndex >= mqttPorts.length) {
            runOnUiThread(() -> {
                binding.syncStatus.setTextColor(Color.RED);
                binding.syncStatus.setText("Connection Failed.");
            });
            return;
        }

        int port = mqttPorts[currentPortIndex];
        String broker = prefs.getString("mqtt_broker", "broker.hivemq.com");
        String clientId = "Android_Fan_" + System.currentTimeMillis() % 100000;

        runOnUiThread(() -> binding.syncStatus.setText("Connecting to Port " + port + "..."));

        executor.execute(() -> {
            try {
                if (attemptId != connectionAttemptId) return;

                closeMqtt();
                String brokerUri;
                if (port == 8883) brokerUri = "ssl://" + broker + ":" + port;
                else if (port == 8884) brokerUri = "wss://" + broker + ":" + port + "/mqtt";
                else if (port == 8000) brokerUri = "ws://" + broker + ":" + port + "/mqtt";
                else brokerUri = "tcp://" + broker + ":" + port;
                
                MqttClient client = new MqttClient(brokerUri, clientId, new MemoryPersistence());
                MqttConnectOptions options = new MqttConnectOptions();
                options.setAutomaticReconnect(true);
                options.setCleanSession(true);
                options.setConnectionTimeout(10);

                client.setCallback(new MqttCallback() {
                    @Override public void connectionLost(Throwable cause) {
                        addLog("Connection Lost: " + (cause != null ? cause.getMessage() : "Unknown"));
                    }
                    @Override public void messageArrived(String topic, MqttMessage message) {
                        String payload = new String(message.getPayload());
                        if (topic.endsWith("/status")) handleMqttStatus(topic, payload);
                        else if (topic.endsWith("/history")) handleMqttHistory(payload);
                        else if (topic.endsWith("/discovery")) handleDiscovery(payload);
                    }
                    @Override public void deliveryComplete(IMqttDeliveryToken token) {}
                });

                client.connect(options);
                
                if (attemptId != connectionAttemptId) {
                    try { client.disconnect(); client.close(); } catch (Exception ignored) {}
                    return;
                }

                mqttClient = client;
                mqttClient.subscribe("smart_home/+/status");
                mqttClient.subscribe("smart_home/+/history");
                mqttClient.subscribe("smart_home/nodes/discovery");
                mqttClient.publish("smart_home/all/commands", new MqttMessage("DISCOVER".getBytes()));

                runOnUiThread(() -> {
                    binding.syncStatus.setText("Connected (Port " + port + ")");
                    binding.syncStatus.setTextColor(Color.parseColor("#4CAF50"));
                });
                addLog("SUCCESS: Connected to port " + port);

            } catch (Exception e) {
                if (attemptId != connectionAttemptId) return;
                currentPortIndex++;
                tryConnectNextPort(attemptId);
            }
        });
    }

    private void handleMqttStatus(String topic, String payload) {
        if (!"firebase/status".equals(topic) && !selectedNodeId.isEmpty() && !topic.contains(selectedNodeId)) return;
        if (System.currentTimeMillis() - lastInteractionTime < 5000) return;

        runOnUiThread(() -> {
            try {
                String jsonStr = payload;
                if (jsonStr.contains("=") && !jsonStr.contains("\":")) {
                    jsonStr = jsonStr.replace("=", "\":\"")
                                   .replace("{", "{\"")
                                   .replace(", ", "\", \"")
                                   .replace("}", "\"}");
                }

                JSONObject json = new JSONObject(jsonStr);
                if (json.has("fan")) {
                    isSyncing = true;

                    if (json.has("ip")) {
                        String ip = json.getString("ip");
                        String id = json.optString("id", "Node");
                        String currentInfo = binding.availableNodesInfo.getText().toString();
                        if (!currentInfo.contains(ip)) {
                            binding.availableNodesInfo.setText(" (" + ip + ")");
                        }
                        String entry = id + " (" + ip + ")";
                        if (!discoveredNodes.contains(entry)) {
                            discoveredNodes.add(entry);
                            nodeAdapter.notifyDataSetChanged();
                        }
                    }

                    String fanState = json.getString("fan");
                    boolean isFanOn = "ON".equalsIgnoreCase(fanState);
                    if (!binding.rowFanStatus.value.getText().toString().equals(fanState)) {
                        binding.rowFanStatus.value.setText(fanState);
                    }
                    if (binding.rowFanStatus.sensorSwitch.isChecked() != isFanOn) {
                        binding.rowFanStatus.sensorSwitch.setChecked(isFanOn);
                    }

                    if (json.has("buzzer")) {
                        String buzzerState = json.getString("buzzer");
                        boolean isBuzzerOn = "ON".equalsIgnoreCase(buzzerState);
                        if (!binding.rowBuzzerStatus.value.getText().toString().equals(buzzerState)) {
                            binding.rowBuzzerStatus.value.setText(buzzerState);
                        }
                        if (binding.rowBuzzerStatus.sensorSwitch.isChecked() != isBuzzerOn) {
                            binding.rowBuzzerStatus.sensorSwitch.setChecked(isBuzzerOn);
                        }
                    }

                    if (json.has("mq2_a")) {
                        binding.rowMq2A.value.setText(String.valueOf(json.get("mq2_a")));
                    }
                    if (json.has("mq2_d")) {
                        String smoke = json.getString("mq2_d");
                        binding.rowMq2D.value.setText(smoke);
                        binding.rowMq2D.value.setTextColor("SMOKE".equalsIgnoreCase(smoke) ? Color.RED : Color.parseColor("#4CAF50"));
                    }
                    if (json.has("ldr")) {
                        String ldrState = String.valueOf(json.get("ldr"));
                        binding.rowLdrStatus.value.setText(ldrState);
                        binding.rowLdrStatus.value.setTextColor("DARK".equalsIgnoreCase(ldrState) ? Color.RED : Color.parseColor("#4CAF50"));
                    }

                    if (json.has("user_sw")) {
                        binding.rowUserSwitch.value.setText(json.getString("user_sw"));
                    }
                    
                    String manualState = json.getString("manual");
                    boolean isManual = "ON".equalsIgnoreCase(manualState) || "true".equalsIgnoreCase(manualState);
                    
                    if (!binding.rowManual.value.getText().toString().equals(manualState)) {
                        binding.rowManual.value.setText(manualState);
                    }
                    if (binding.rowManual.sensorSwitch.isChecked() != isManual) {
                        binding.rowManual.sensorSwitch.setChecked(isManual);
                        binding.rowFanStatus.sensorSwitch.setEnabled(isManual);
                        binding.rowBuzzerStatus.sensorSwitch.setEnabled(isManual);
                    }
                    
                    binding.syncStatus.setText("Last Update: " + new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date()));
                    isSyncing = false;
                }
            } catch (Exception e) {
                Log.e(TAG, "Status Parse Error: " + e.getMessage());
                isSyncing = false;
            }
        });
    }

    private void handleDiscovery(String payload) {
        try {
            JSONObject json = new JSONObject(payload);
            if (json.getString("id").startsWith("Kitchen_Fan")) {
                String ip = json.getString("ip");
                String id = json.getString("id");
                String entry = id + " (" + ip + ")";
                runOnUiThread(() -> {
                    if (!discoveredNodes.contains(entry)) {
                        discoveredNodes.add(entry);
                        nodeAdapter.notifyDataSetChanged();
                    }
                });
            }
        } catch (Exception ignored) {}
    }

    private void handleMqttHistory(String payload) {
        if ("EOF".equals(payload)) {
            runOnUiThread(() -> updateHistoryTable(mqttHistoryBuffer));
        } else {
            mqttHistoryBuffer.add(payload);
        }
    }

    private void updateHistoryTable(List<String> lines) {
        if (binding.historyTable.getChildCount() > 1) {
            binding.historyTable.removeViews(1, binding.historyTable.getChildCount() - 1);
        }
        
        for (int i = lines.size() - 1; i >= 0; i--) {
            String[] cols = lines.get(i).split(",");
            TableRow row = new TableRow(this);
            for (String col : cols) {
                TextView tv = new TextView(this);
                tv.setText(col);
                tv.setTextColor(Color.parseColor("#4CAF50"));
                tv.setPadding(20, 8, 20, 8);
                row.addView(tv);
            }
            binding.historyTable.addView(row);
        }
    }

    private void syncData() {
        lastInteractionTime = 0;
        if (syncMode == 1 && !selectedNodeIp.isEmpty()) {
            executor.execute(() -> {
                try {
                    URL url = new URL("http://" + selectedNodeIp + "/status");
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    if (conn.getResponseCode() == 200) {
                        BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                        StringBuilder sb = new StringBuilder();
                        String line;
                        while ((line = reader.readLine()) != null) sb.append(line);
                        handleMqttStatus("local/ip/status", sb.toString());
                    }
                    conn.disconnect();
                } catch (Exception ignored) {}
            });
        } else if (syncMode == 2) {
            if (firebaseDatabase != null) {
                String fbNode = prefs.getString("firebase_kitchen_fan_node", "");
                firebaseDatabase.getReference(fbNode).child("command")
                    .setValue("SYNC");
            }
        } else if (isMqttAvailable()) {
            executor.execute(() -> {
                try {
                    mqttClient.publish("smart_home/all/commands", new MqttMessage("SYNC".getBytes()));
                } catch (Exception ignored) {}
            });
        }
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

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (connectivityManager != null && networkCallback != null) {
            try { connectivityManager.unregisterNetworkCallback(networkCallback); } catch (Exception ignored) {}
        }
        if (firebaseRef != null && firebaseListener != null) {
            firebaseRef.removeEventListener(firebaseListener);
        }
        executor.shutdown();
        closeMqtt();
    }
}
