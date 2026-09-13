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
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.sample.databinding.ActivityBedRoomMonitorBinding;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.MqttCallback;
import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;
import org.json.JSONObject;

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

public class BedRoomMonitorActivity extends AppCompatActivity {

    private static final String TAG = "BedRoomMonitorActivity";
    private ActivityBedRoomMonitorBinding binding;
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

    private final android.os.Handler heartbeatHandler = new android.os.Handler(android.os.Looper.getMainLooper());
    private final Runnable heartbeatRunnable = new Runnable() {
        @Override
        public void run() {
            sendNodeCommand("HEARTBEAT");
            heartbeatHandler.postDelayed(this, 30000); // Send every 30 seconds
        }
    };

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
        binding = ActivityBedRoomMonitorBinding.inflate(getLayoutInflater());
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
        } else if (AppDefaults.ENABLE_MQTT) {
            initMqtt();
            setupNetworkListener();
        } else {
            setupNetworkListener();
        }

        binding.rowMq135Analog.label.setText("MQ135 Gas Analog");
        binding.rowMq135Digital.label.setText("MQ135 Gas Digital");
        binding.rowDhtTemp.label.setText("DHT11 Temperature");
        binding.rowDhtHum.label.setText("DHT11 Humidity");
        binding.rowAmbient.label.setText("LDR Light");
        binding.rowPir.label.setText("PIR Motion");
        binding.rowBuzzer.label.setText("Buzzer Alert");
        binding.rowBuzzer.sensorSwitch.setVisibility(android.view.View.VISIBLE);
        binding.rowManualOverride.label.setText("Manual Override");
        binding.rowManualOverride.sensorSwitch.setVisibility(android.view.View.VISIBLE);
        binding.rowIrTransmitterMode.label.setText("IR Transmitter Mode");
        binding.rowIrTransmitterMode.sensorSwitch.setVisibility(android.view.View.VISIBLE);
        binding.rowHeapFree.label.setText("Free Memory");

        binding.btnBack.setOnClickListener(v -> finish());
        binding.btnSync.setOnClickListener(v -> syncData());
        binding.btnSyncHistory.setOnClickListener(v -> syncHistory());
        binding.btnClearHistory.setOnClickListener(v -> clearHistory());
        binding.btnSettings.setOnClickListener(v -> {
            Intent intent = new Intent(this, SmartHomeSettingsActivity.class);
            intent.putExtra("caller_context", "bedroom");
            startActivity(intent);
        });

        binding.historyHeader.setOnClickListener(v -> toggleHistoryExpansion());

        binding.rowManualOverride.sensorSwitch.setOnClickListener(v -> {
            boolean isChecked = binding.rowManualOverride.sensorSwitch.isChecked();
            binding.rowManualOverride.value.setText(isChecked ? "ON" : "OFF");
            binding.rowBuzzer.sensorSwitch.setEnabled(isChecked);
            updateSensorConfig();
        });

        binding.sbBuzzerFreq.setOnSeekBarChangeListener(new android.widget.SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(android.widget.SeekBar seekBar, int progress, boolean fromUser) {
                binding.tvBuzzerFreqVal.setText(progress + " Hz");
            }
            @Override public void onStartTrackingTouch(android.widget.SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(android.widget.SeekBar seekBar) {
                updateSensorConfig();
            }
        });

        binding.rowBuzzer.sensorSwitch.setOnClickListener(v -> {
            binding.rowBuzzer.value.setText(binding.rowBuzzer.sensorSwitch.isChecked() ? "ON" : "OFF");
            updateSensorConfig();
        });

        binding.rowIrTransmitterMode.sensorSwitch.setOnClickListener(v -> {
            boolean isChecked = binding.rowIrTransmitterMode.sensorSwitch.isChecked();
            binding.rowIrTransmitterMode.value.setText(isChecked ? "ON" : "OFF");
            binding.remotesHeader.setVisibility(isChecked ? android.view.View.VISIBLE : android.view.View.GONE);
            if (!isChecked) {
                binding.remotesContainer.setVisibility(android.view.View.GONE);
                binding.remotesHeader.setText("Remote Controls ▼ (Click to Expand)");
            }
            updateSensorConfig();
        });

        binding.remotesHeader.setOnClickListener(v -> {
            if (binding.remotesContainer.getVisibility() == android.view.View.VISIBLE) {
                binding.remotesContainer.setVisibility(android.view.View.GONE);
                binding.remotesHeader.setText("Remote Controls ▼ (Click to Expand)");
            } else {
                binding.remotesContainer.setVisibility(android.view.View.VISIBLE);
                binding.remotesHeader.setText("Remote Controls ▲ (Click to Collapse)");
            }
        });

        setupRemoteButtons();

        binding.rowBuzzer.sensorSwitch.setEnabled(binding.rowManualOverride.sensorSwitch.isChecked());
    }

    private void setupRemoteButtons() {
        // Sony TV (Sony protocol = 4 in IRremoteESP8266)
        setupIrButton(binding.btnTvPower, "4,A90,12");
        setupIrButton(binding.btnTvInput, "4,A50,12");
        setupIrButton(binding.btnTvExit, "4,C70,12");
        setupIrButton(binding.btnTvVolUp, "4,490,12");
        setupIrButton(binding.btnTvVolDown, "4,C90,12");
        setupIrButton(binding.btnTvChUp, "4,090,12");
        setupIrButton(binding.btnTvChDown, "4,890,12");
        setupIrButton(binding.btnTvMute, "4,290,12");
        setupIrButton(binding.btnTvHome, "4,070,12");

        // Sony Soundbar
        setupIrButton(binding.btnSbPower, "4,540A,15");
        setupIrButton(binding.btnSbInput, "4,0C0A,15");
        setupIrButton(binding.btnSbMute, "4,140A,15");
        setupIrButton(binding.btnSbVolUp, "4,240A,15");
        setupIrButton(binding.btnSbVolDown, "4,640A,15");
        setupIrButton(binding.btnSbSwUp, "4,3A0A,15");
        setupIrButton(binding.btnSbSwDown, "4,7A0A,15");
        setupIrButton(binding.btnSbNight, "4,1E0A,15");
        setupIrButton(binding.btnSbMusic, "4,120A,15");

        // Sarru Automation (NEC protocol = 3 in IRremoteESP8266)
        setupIrButton(binding.btnSaPower, "3,00FF906F,32");
        setupIrButton(binding.btnSa1, "3,00FF6897,32");
        setupIrButton(binding.btnSa2, "3,00FF9867,32");
        setupIrButton(binding.btnSa3, "3,00FFB04F,32");
        setupIrButton(binding.btnSa4, "3,00FF30CF,32");
        setupIrButton(binding.btnSa5, "3,00FF18E7,32");
        setupIrButton(binding.btnSa6, "3,00FF7A85,32");
        setupIrButton(binding.btnSa7, "3,00FF10EF,32");
        setupIrButton(binding.btnSa8, "3,00FF38C7,32");
        setupIrButton(binding.btnSa9, "3,00FF5AA5,32");

        // AC Remote
        setupIrButton(binding.btnAcPowerOn, "PRESET:AC_ON");
        setupIrButton(binding.btnAcPowerOff, "PRESET:AC_OFF");
        setupIrButton(binding.btnAc26c, "PRESET:AC_26C");

        // Fan Remote
        setupIrButton(binding.btnFanPower, "PRESET:FAN_OFF");
        setupIrButton(binding.btnFanSpeedUp, "PRESET:FAN_SPEED3");
        setupIrButton(binding.btnFanSpeedDown, "PRESET:FAN_SPEED1");
        setupIrButton(binding.btnFanSpeed1, "PRESET:FAN_SPEED1");
        setupIrButton(binding.btnFanSpeed2, "PRESET:FAN_SPEED2");
        setupIrButton(binding.btnFanSpeed3, "PRESET:FAN_SPEED3");
        setupIrButton(binding.btnFanSpeed4, "PRESET:FAN_SPEED4");
        setupIrButton(binding.btnFanSpeed5, "PRESET:FAN_SPEED5");
        setupIrButton(binding.btnFanOn, "PRESET:FAN_ON");
    }

    private void setupIrButton(android.view.View btn, String irData) {
        btn.setOnClickListener(v -> {
            // Glow effect
            v.animate().scaleX(1.1f).scaleY(1.1f).setDuration(100).withEndAction(() -> v.animate().scaleX(1.0f).scaleY(1.0f).setDuration(100).start()).start();
            
            sendIrCommand(irData);
        });
    }

    private void sendIrCommand(String irData) {
        String payload = "IR_SEND:" + irData;
        sendNodeCommand(payload);
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadSettings();
        heartbeatHandler.removeCallbacks(heartbeatRunnable);
        heartbeatHandler.post(heartbeatRunnable);
        if (syncMode == 1) {
            LocalDiscoveryManager.discoverDevices(this, (key, deviceName, ipAddress) -> {
                if ("ip_bedroom".equals(key)) {
                    runOnUiThread(this::loadSettings);
                }
            });
        } else if (AppDefaults.ENABLE_MQTT && isMqttAvailable()) {
            executor.execute(() -> {
                try {
                    mqttClient.publish("smart_home/all/commands", new MqttMessage("DISCOVER".getBytes()));
                } catch (Exception ignored) {}
            });
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        heartbeatHandler.removeCallbacks(heartbeatRunnable);
    }

    private void sendNodeCommand(String payload) {
        if (syncMode == 1 && !selectedNodeIp.isEmpty()) {
            executor.execute(() -> {
                try {
                    String urlStr = "http://" + selectedNodeIp + (payload.startsWith("CONFIG:") ? "/config" : "/control?cmd=" + payload);
                    URL url = new URL(urlStr);
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    if (payload.startsWith("CONFIG:")) {
                        conn.setRequestMethod("POST");
                        conn.setDoOutput(true);
                        conn.getOutputStream().write(payload.getBytes());
                    }
                    conn.getResponseCode();
                    conn.disconnect();
                } catch (Exception ignored) {}
            });
        } else if (syncMode == 2) {
            if (firebaseDatabase != null) {
                firebaseDatabase.getReference("FrmMobile").child("bedroom").child("command")
                        .setValue(payload);
            }
        } else if (AppDefaults.ENABLE_MQTT && isMqttAvailable()) {
            executor.execute(() -> {
                try {
                    String targetTopic = selectedNodeId.isEmpty() ? "smart_home/all/commands" : "smart_home/" + selectedNodeId + "/commands";
                    mqttClient.publish(targetTopic, new MqttMessage(payload.getBytes()));
                } catch (Exception ignored) {}
            });
        }
    }

    private void initFirebase() {
        String url = prefs.getString("firebase_url", AppDefaults.FIREBASE_URL);
        String room = AppDefaults.NODE_BEDROOM;
        
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
            
            addLog("Listening to Outbox: FrmNodeMcu/" + room);
            
            firebaseListener = new ValueEventListener() {
                @Override
                public void onDataChange(DataSnapshot dataSnapshot) {
                    if (dataSnapshot.exists()) {
                        Object value = dataSnapshot.child("status").getValue();
                        if (value != null) {
                            handleMqttStatus("firebase/status", value.toString());
                        }
                        
                        DataSnapshot historyNode = dataSnapshot.child("history");
                        if (historyNode.exists()) {
                            final List<String> newHistory = new ArrayList<>();
                            for (DataSnapshot child : historyNode.getChildren()) {
                                Object entry = child.getValue();
                                if (entry != null) newHistory.add(entry.toString());
                            }
                            
                            executor.execute(() -> {
                                mqttHistoryBuffer.clear();
                                mqttHistoryBuffer.addAll(newHistory);
                                saveToLocalCsv(newHistory);
                                runOnUiThread(() -> updateHistoryTable(mqttHistoryBuffer));
                            });
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
        syncMode = prefs.getInt("sync_mode", 0);
        String savedIp = LocalDiscoveryManager.getDeviceIp(prefs, "bedroom");
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
            config.put("manual_override", binding.rowManualOverride.sensorSwitch.isChecked());
            config.put("buzzer", binding.rowBuzzer.sensorSwitch.isChecked());
            config.put("buzzer_freq", binding.sbBuzzerFreq.getProgress());
            config.put("ir_transmitter_mode", binding.rowIrTransmitterMode.sensorSwitch.isChecked());
            
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
                    firebaseDatabase.getReference("FrmMobile").child("bedroom").child("command")
                        .setValue(payload)
                        .addOnCompleteListener(task -> {
                            if (task.isSuccessful()) addLog("Config Sent successfully");
                            else addLog("Config Send Failed: " + (task.getException() != null ? task.getException().getMessage() : "Unknown"));
                        });
                }
            } else if (AppDefaults.ENABLE_MQTT && isMqttAvailable()) {
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
                firebaseDatabase.getReference("FrmMobile").child("bedroom").child("command")
                    .setValue("CLEAR");
            }
        } else if (AppDefaults.ENABLE_MQTT && isMqttAvailable()) {
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
                    }
                    conn.disconnect();
                } catch (Exception e) {
                    runOnUiThread(() -> Toast.makeText(this, "Node unreachable: " + e.getMessage(), Toast.LENGTH_SHORT).show());
                }
            });
        } else if (syncMode == 2) {
            if (firebaseDatabase != null) {
                firebaseDatabase.getReference("FrmMobile").child("bedroom").child("command")
                    .setValue("HISTORY");
            }
        } else if (AppDefaults.ENABLE_MQTT && isMqttAvailable()) {
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
        if (!AppDefaults.ENABLE_MQTT) return;
        
        executor.execute(this::closeMqtt);
        
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
        
        if (logBuilder.length() > 1000) {
            logBuilder.setLength(1000);
        }

        runOnUiThread(() -> {
            binding.tvConnectionLogs.setText(logBuilder.toString());
            binding.logScrollView.post(() -> binding.logScrollView.fullScroll(android.view.View.FOCUS_UP));
        });
    }

    private void tryConnectNextPort(final long attemptId) {
        if (attemptId != connectionAttemptId) return;

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
                final String errorMsg = e.getMessage() != null ? e.getMessage() : "Timeout";
                addLog("FAILED: Port " + port + " - " + errorMsg);
                currentPortIndex++;
                tryConnectNextPort(attemptId);
            }
        });
    }

    private void closeMqtt() {
        final MqttClient clientToClose = mqttClient;
        if (clientToClose != null) {
            mqttClient = null;
            try {
                if (clientToClose.isConnected()) clientToClose.disconnect(500);
                clientToClose.close();
            } catch (Exception ignored) {}
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

                    binding.rowMq135Analog.value.setText(json.optString("mq135_analog", "0"));
                    binding.rowMq135Digital.value.setText("1".equals(json.optString("mq135_digital", "0")) ? "ALERT" : "NORMAL");
                    binding.rowDhtTemp.value.setText(json.getString("dht_temp") + " °C");
                    binding.rowDhtHum.value.setText(json.getString("dht_hum") + " %");
                    binding.rowAmbient.value.setText(json.getString("light_raw"));
                    binding.rowPir.value.setText("1".equals(json.optString("pir", "0")) ? "MOTION DETECTED" : "NO MOTION");
                    
                    String buzzerState = json.optString("buzzer", "OFF");
                    binding.rowBuzzer.value.setText(buzzerState);
                    
                    int bFreq = json.optInt("buzzer_freq", 2000);
                    
                    String overrideState = json.optString("manual_override", "OFF");
                    manualOverrideEnabled = "ON".equals(overrideState);
                    
                    binding.rowManualOverride.value.setText(json.optString("reason", overrideState));
                    binding.rowManualOverride.sensorSwitch.setChecked(manualOverrideEnabled);

                    String irTransState = json.optString("ir_transmitter_mode", "OFF");
                    binding.rowIrTransmitterMode.value.setText(irTransState);
                    boolean isTransmitting = "ON".equals(irTransState);
                    binding.remotesHeader.setVisibility(isTransmitting ? android.view.View.VISIBLE : android.view.View.GONE);
                    if (!isTransmitting) {
                        binding.remotesContainer.setVisibility(android.view.View.GONE);
                    }

                    if (System.currentTimeMillis() - lastInteractionTime > 5000) {
                        binding.rowBuzzer.sensorSwitch.setChecked("ON".equals(buzzerState));
                        binding.rowIrTransmitterMode.sensorSwitch.setChecked(isTransmitting);
                        binding.sbBuzzerFreq.setProgress(bFreq);
                        binding.tvBuzzerFreqVal.setText(bFreq + " Hz");
                    }
                    
                    binding.rowBuzzer.sensorSwitch.setEnabled(manualOverrideEnabled);
                    
                    long heap = json.optLong("heap", 0);
                    binding.rowHeapFree.value.setText((heap / 1024) + " KB Heap");

                    String nodeTs = json.optString("ts", "N/A");
                    String appTs = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
                    binding.syncStatus.setText("Node: " + nodeTs + " | App: " + appTs);
                    binding.syncStatus.setTextColor(Color.parseColor("#4CAF50"));
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
        
        int count = 0;
        int maxRows = 20;

        for (int i = lines.size() - 1; i >= 0 && count < maxRows; i--) {
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
            count++;
        }
    }

    private void saveToLocalCsv(List<String> newLines) {
        File file = new File(StorageUtils.getDownloadsDir(), "IOT_HOME/BedRoom/BedRoomMonitor.csv");
        try {
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
                if (trimmed.isEmpty() || trimmed.startsWith("Date,Time")) continue;
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
                    }
                    conn.disconnect();
                } catch (Exception e) {
                    runOnUiThread(() -> Toast.makeText(this, "Node unreachable: " + e.getMessage(), Toast.LENGTH_SHORT).show());
                }
            });
        } else if (syncMode == 2) {
            if (firebaseDatabase != null) {
                firebaseDatabase.getReference("FrmMobile").child("bedroom").child("command")
                    .setValue("SYNC");
            }
        } else if (AppDefaults.ENABLE_MQTT && isMqttAvailable()) {
            executor.execute(() -> {
                try {
                    mqttClient.publish("smart_home/all/commands", new MqttMessage("SYNC".getBytes()));
                } catch (Exception ignored) {}
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
