package com.example.sample;

import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Toast;

import android.widget.ArrayAdapter;
import android.widget.TableRow;
import android.widget.TextView;
import android.graphics.Color;

import androidx.appcompat.app.AppCompatActivity;

import com.example.sample.databinding.ActivityStudyRoomBinding;

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
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkRequest;
import android.net.NetworkCapabilities;

public class StudyRoomActivity extends AppCompatActivity {

    private static final String TAG = "StudyRoomActivity";
    private ActivityStudyRoomBinding binding;
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
    private boolean manualOverrideEnabled = false;
    private boolean isSyncing = false;
    private long lastInteractionTime = 0;
    private long connectionAttemptId = 0;

    // MQTT Configuration for HiveMQ Broker
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
        binding = ActivityStudyRoomBinding.inflate(getLayoutInflater());
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

        binding.rowDhtTemp.label.setText(R.string.label_dht_temp);
        binding.rowDhtHum.label.setText(R.string.label_dht_hum);
        binding.rowAmbient.label.setText(R.string.label_ambient_light);
        binding.rowEmergencyLight.label.setText(R.string.label_emergency_light);
        binding.rowEmergencyLight.sensorSwitch.setVisibility(android.view.View.VISIBLE);
        binding.rowManualOverride.label.setText("Manual Override");
        binding.rowManualOverride.sensorSwitch.setVisibility(android.view.View.VISIBLE);
        binding.rowHeapFree.label.setText("Free Memory");

        binding.btnBack.setOnClickListener(v -> finish());
        binding.btnSync.setOnClickListener(v -> syncData());
        binding.btnSyncHistory.setOnClickListener(v -> syncHistory());
        binding.btnClearHistory.setOnClickListener(v -> clearHistory());
        binding.btnSettings.setOnClickListener(v -> {
            Intent intent = new Intent(this, SmartHomeSettingsActivity.class);
            intent.putExtra("caller_context", "study");
            startActivity(intent);
        });

        binding.historyHeader.setOnClickListener(v -> toggleHistoryExpansion());

        // Interaction listeners
        binding.rowEmergencyLight.sensorSwitch.setOnClickListener(v -> {
            binding.rowEmergencyLight.value.setText(binding.rowEmergencyLight.sensorSwitch.isChecked() ? "ON" : "OFF");
            updateSensorConfig();
        });
        
        binding.rowManualOverride.sensorSwitch.setOnClickListener(v -> {
            boolean isChecked = binding.rowManualOverride.sensorSwitch.isChecked();
            binding.rowManualOverride.value.setText(isChecked ? "ON" : "OFF");
            binding.rowEmergencyLight.sensorSwitch.setEnabled(isChecked);
            updateSensorConfig();
        });

        // Initialize state based on manual override
        binding.rowEmergencyLight.sensorSwitch.setEnabled(binding.rowManualOverride.sensorSwitch.isChecked());
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadSettings();
        if (isMqttAvailable()) {
            executor.execute(() -> {
                try {
                    mqttClient.publish("smart_home/all/commands", new MqttMessage("DISCOVER".getBytes()));
                } catch (Exception ignored) {}
            });
        }
    }

    private void initFirebase() {
        String url = prefs.getString("firebase_url", AppDefaults.FIREBASE_URL);
        String room = AppDefaults.NODE_STUDY; 
        
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
            // LISTEN to the OUTBOX
            DatabaseReference outboxRef = firebaseDatabase.getReference("FrmNodeMcu").child(room);
            
            addLog("Listening to Outbox: FrmNodeMcu/" + room);
            
            firebaseListener = new ValueEventListener() {
                @Override
                public void onDataChange(DataSnapshot dataSnapshot) {
                    if (dataSnapshot.exists()) {
                        addLog("Data received from Outbox");
                        Object value = dataSnapshot.child("status").getValue();
                        if (value != null) {
                            handleMqttStatus("firebase/status", value.toString());
                        }
                        
                        Object otaStatus = dataSnapshot.child("ota_status").getValue();
                        if (otaStatus != null) {
                            addLog("OTA Status: " + otaStatus.toString());
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
                    } else {
                        addLog("Outbox is empty. Waiting for device...");
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

    private void loadSettings() {
        syncMode = prefs.getInt("sync_mode", 0); // 0: MQTT, 1: IP, 2: Firebase
        String savedIp = prefs.getString("local_node_ip", AppDefaults.DEFAULT_NODE_IP);
        String localEntryPrefix = "Local IP (";

        boolean changed = false;
        // Clean up any old Local IP entries to refresh
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
                
                // Auto-select Local IP
                selectedNodeIp = savedIp;
                selectedNodeId = "";
                runOnUiThread(() -> binding.nodeSelector.setSelection(0));
            }
        }

        if (changed) {
            nodeAdapter.notifyDataSetChanged();
        }
        updateDiscoveryStatus();
    }

    private void setupNetworkListener() {
        if (syncMode == 2) return;
        connectivityManager = (ConnectivityManager) getSystemService(android.content.Context.CONNECTIVITY_SERVICE);
        networkCallback = new ConnectivityManager.NetworkCallback() {
            @Override
            public void onAvailable(Network network) {
                addLog("Network Restored. Re-initializing...");
                runOnUiThread(() -> {
                    binding.syncStatus.setTextColor(Color.GRAY);
                    binding.syncStatus.setText("Network restored. Reconnecting...");
                });
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

    private void unregisterNetworkListener() {
        if (connectivityManager != null && networkCallback != null) {
            try {
                connectivityManager.unregisterNetworkCallback(networkCallback);
            } catch (Exception ignored) {}
        }
    }

    private void updateDiscoveryStatus() {
        runOnUiThread(() -> {
            int count = discoveredNodes.size();
            if (count == 0) {
                binding.availableNodesInfo.setText(" (Searching...)");
            } else {
                binding.availableNodesInfo.setText(" (" + count + " Available)");
            }
        });
    }

    private void updateSensorConfig() {
        if (isSyncing) return;
        lastInteractionTime = System.currentTimeMillis();
        try {
            JSONObject config = new JSONObject();
            config.put("en_emer", binding.rowEmergencyLight.sensorSwitch.isChecked());
            config.put("manual_override", binding.rowManualOverride.sensorSwitch.isChecked());
            
            String payload = "CONFIG:" + config.toString();

            if (syncMode == 1 && !selectedNodeIp.isEmpty()) {
                executor.execute(() -> {
                    try {
                        URL url = new URL("http://" + selectedNodeIp + "/config");
                        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                        conn.setRequestMethod("POST");
                        conn.setConnectTimeout(5000);
                        conn.setDoOutput(true);
                        conn.getOutputStream().write(payload.getBytes());
                        int code = conn.getResponseCode();
                        conn.disconnect();
                        if (code != 200) runOnUiThread(() -> Toast.makeText(this, "Config sync failed: " + code, Toast.LENGTH_SHORT).show());
                    } catch (Exception e) {
                        runOnUiThread(() -> Toast.makeText(this, "Node unreachable: " + e.getMessage(), Toast.LENGTH_SHORT).show());
                    }
                });
            } else if (syncMode == 2) {
                addLog("Sending Config to Firebase...");
                if (firebaseDatabase != null) {
                    firebaseDatabase.getReference("FrmMobile").child("study").child("command")
                        .setValue(payload)
                        .addOnCompleteListener(task -> {
                            if (task.isSuccessful()) addLog("Config Sent successfully");
                            else addLog("Config Send Failed: " + (task.getException() != null ? task.getException().getMessage() : "Unknown"));
                        });
                } else {
                    addLog("Firebase Database not initialized");
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
                firebaseDatabase.getReference("FrmMobile").child("study").child("command")
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

    private void toggleHistoryExpansion() {
        isHistoryExpanded = !isHistoryExpanded;
        binding.cardIpSync.setVisibility(android.view.View.GONE == binding.cardIpSync.getVisibility() ? android.view.View.VISIBLE : android.view.View.GONE);
        binding.statusScrollView.setVisibility(android.view.View.GONE == binding.statusScrollView.getVisibility() ? android.view.View.VISIBLE : android.view.View.GONE);
        binding.historyContainer.setVisibility(isHistoryExpanded ? android.view.View.VISIBLE : android.view.View.GONE);
        binding.historyHeader.setText(isHistoryExpanded ? R.string.history_collapse : R.string.history_expand);
    }

    private void syncHistory() {
        if (syncMode == 1 && !selectedNodeIp.isEmpty()) {
            mqttHistoryBuffer.clear();
            executor.execute(() -> {
                try {
                    URL url = new URL("http://" + selectedNodeIp + "/sync");
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setConnectTimeout(5000);
                    if (conn.getResponseCode() == 200) {
                        BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                        String line;
                        while ((line = reader.readLine()) != null) {
                            if (!line.trim().isEmpty()) mqttHistoryBuffer.add(line);
                        }
                        runOnUiThread(() -> {
                            updateHistoryTable(mqttHistoryBuffer);
                            Toast.makeText(this, "History synced via IP", Toast.LENGTH_SHORT).show();
                        });
                    } else {
                        final int code = conn.getResponseCode();
                        runOnUiThread(() -> Toast.makeText(this, "Sync failed: " + code, Toast.LENGTH_SHORT).show());
                    }
                    conn.disconnect();
                } catch (Exception e) {
                    runOnUiThread(() -> Toast.makeText(this, "Node unreachable: " + e.getMessage(), Toast.LENGTH_SHORT).show());
                }
            });
        } else if (syncMode == 2) {
            if (firebaseDatabase != null) {
                firebaseDatabase.getReference("FrmMobile").child("study").child("command")
                    .setValue("HISTORY");
            }
        } else if (isMqttAvailable()) {
            mqttHistoryBuffer.clear();
            executor.execute(() -> {
                try {
                    String targetTopic = selectedNodeId.isEmpty() ? "smart_home/all/commands" : "smart_home/" + selectedNodeId + "/commands";
                    mqttClient.publish(targetTopic, new MqttMessage("HISTORY".getBytes()));
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
        if (attemptId != connectionAttemptId) return; // Stale attempt

        if (currentPortIndex >= mqttPorts.length) {
            runOnUiThread(() -> {
                binding.syncStatus.setTextColor(Color.RED);
                binding.syncStatus.setText("Network Connection Failed.");
            });
            addLog("ERROR: All ports failed. Please check settings or network.");
            return;
        }

        int port = mqttPorts[currentPortIndex];
        String broker = prefs.getString("mqtt_broker", "broker.hivemq.com");
        String clientId = "Android_" + System.currentTimeMillis() % 100000;

        runOnUiThread(() -> binding.syncStatus.setText("Connecting to Port " + port + "..."));
        addLog("Attempting connection to " + broker + ":" + port);

        executor.execute(() -> {
            try {
                if (attemptId != connectionAttemptId) return;

                closeMqtt(); // Clean up any previous attempt
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
                options.setKeepAliveInterval(60);

                client.setCallback(new MqttCallback() {
                    @Override public void connectionLost(Throwable cause) {
                        runOnUiThread(() -> binding.syncStatus.setText("Connection Lost. Retrying..."));
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

                Log.w(TAG, "Port " + port + " failed: " + e.getMessage());
                final String errorMsg = e.getMessage() != null ? e.getMessage() : "Timeout";
                addLog("FAILED: Port " + port + " - " + errorMsg);
                currentPortIndex++;
                tryConnectNextPort(attemptId);
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

    private void handleMqttStatus(String topic, String payload) {
        if (!"firebase/status".equals(topic) && !selectedNodeId.isEmpty() && !topic.contains(selectedNodeId)) return;
        if (System.currentTimeMillis() - lastInteractionTime < 4000) return;

        runOnUiThread(() -> {
            try {
                JSONObject json = new JSONObject(payload);
                if (json.has("dht_temp")) {
                    isSyncing = true;

                    // Update IP and ID info if available (crucial for Firebase mode)
                    if (json.has("ip")) {
                        String ip = json.getString("ip");
                        String id = json.optString("id", "Node");
                        if (syncMode == 2) {
                            binding.availableNodesInfo.setText(" (" + ip + ")");
                            String entry = id + " (" + ip + ")";
                            if (!discoveredNodes.contains(entry)) {
                                discoveredNodes.add(entry);
                                nodeAdapter.notifyDataSetChanged();
                            }
                        }
                    }

                    binding.rowDhtTemp.value.setText(json.getString("dht_temp") + " °C");
                    binding.rowDhtHum.value.setText(json.getString("dht_hum") + " %");
                    binding.rowAmbient.value.setText(json.getString("light_raw"));
                    
                    String emerState = json.getString("emer");
                    binding.rowEmergencyLight.value.setText(emerState);
                    
                    String overrideState = json.optString("manual_override", "OFF");
                    manualOverrideEnabled = "ON".equals(overrideState);
                    
                    // Display the long reason string but use ON/OFF for logic
                    binding.rowManualOverride.value.setText(json.optString("reason", overrideState));
                    binding.rowManualOverride.sensorSwitch.setChecked(manualOverrideEnabled);

                    // Sync the Emergency Light switch position and enabled state
                    binding.rowEmergencyLight.sensorSwitch.setChecked("ON".equals(emerState));
                    binding.rowEmergencyLight.sensorSwitch.setEnabled(manualOverrideEnabled);
                    
                    long heap = json.optLong("heap", 0);
                    long fsFree = json.optLong("fs_free", 0);
                    String memoryInfo = (heap / 1024) + " KB Heap";
                    if (fsFree > 0) memoryInfo += " | " + (fsFree / 1024) + " KB FS";
                    binding.rowHeapFree.value.setText(memoryInfo);

                    binding.syncStatus.setText("Last Update: " + new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date()));
                    isSyncing = false;
                }
            } catch (Exception ignored) {
                isSyncing = false;
            }
        });
    }

    private void handleDiscovery(String payload) {
        try {
            JSONObject json = new JSONObject(payload);
            String ip = json.getString("ip");
            String id = json.getString("id");
            String entry = id + " (" + ip + ")";
            runOnUiThread(() -> {
                if (!discoveredNodes.contains(entry)) {
                    discoveredNodes.add(entry);
                    nodeAdapter.notifyDataSetChanged();
                    updateDiscoveryStatus();
                }
            });
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
        
        saveToLocalCsv(lines);

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

    private void saveToLocalCsv(List<String> newLines) {
        String path = prefs.getString("csv_path", getString(R.string.default_csv_path));
        File file;
        if (path.startsWith("/")) {
            file = new File(path);
        } else {
            file = new File(StorageUtils.getDownloadsDir(), path);
        }
        
        try {
            // Ensure parent directories exist
            File parent = file.getParentFile();
            if (parent != null && !parent.exists()) parent.mkdirs();

            List<String> existingLines = new ArrayList<>();
            if (file.exists()) {
                BufferedReader br = new BufferedReader(new java.io.FileReader(file));
                String line;
                while ((line = br.readLine()) != null) {
                    existingLines.add(line.trim());
                }
                br.close();
            }

            java.io.FileWriter fw = new java.io.FileWriter(file, true);
            int addedCount = 0;
            
            for (String newLine : newLines) {
                String trimmed = newLine.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("Date,Time")) continue; // Skip header
                
                if (!existingLines.contains(trimmed)) {
                    fw.write(trimmed + "\n");
                    existingLines.add(trimmed);
                    addedCount++;
                }
            }
            fw.close();
            
            if (addedCount > 0) {
                final int finalAdded = addedCount;
                runOnUiThread(() -> Toast.makeText(this, "Saved " + finalAdded + " new records to local CSV", Toast.LENGTH_SHORT).show());
            }
        } catch (Exception e) {
            Log.e(TAG, "Error saving local CSV: " + e.getMessage());
        }
    }

    private void syncData() {
        lastInteractionTime = 0;
        if (syncMode == 1 && !selectedNodeIp.isEmpty()) {
            executor.execute(() -> {
                try {
                    URL url = new URL("http://" + selectedNodeIp + "/status");
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setConnectTimeout(5000);
                    if (conn.getResponseCode() == 200) {
                        BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                        StringBuilder sb = new StringBuilder();
                        String line;
                        while ((line = reader.readLine()) != null) sb.append(line);
                        handleMqttStatus("local/ip/status", sb.toString());
                        runOnUiThread(() -> Toast.makeText(this, "Data synced via IP", Toast.LENGTH_SHORT).show());
                    } else {
                        final int code = conn.getResponseCode();
                        runOnUiThread(() -> Toast.makeText(this, "Sync failed: " + code, Toast.LENGTH_SHORT).show());
                    }
                    conn.disconnect();
                } catch (Exception e) {
                    runOnUiThread(() -> Toast.makeText(this, "Node unreachable: " + e.getMessage(), Toast.LENGTH_SHORT).show());
                }
            });
        } else if (syncMode == 2) {
            if (firebaseDatabase != null) {
                firebaseDatabase.getReference("FrmMobile").child("study").child("command")
                    .setValue("SYNC");
            }
        } else if (isMqttAvailable()) {
            executor.execute(() -> {
                try {
                    String targetTopic = "smart_home/all/commands";
                    mqttClient.publish(targetTopic, new MqttMessage("SYNC".getBytes()));
                } catch (Exception ignored) {}
            });
        } else {
            addLog("MQTT not available. Re-initializing...");
            initMqtt();
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
