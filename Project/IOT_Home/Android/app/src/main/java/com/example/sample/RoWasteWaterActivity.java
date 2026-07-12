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

import com.example.sample.databinding.ActivityRoWaterBinding;

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

public class RoWasteWaterActivity extends AppCompatActivity {

    private static final String TAG = "RoWasteWaterActivity";
    private ActivityRoWaterBinding binding;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private boolean isHistoryExpanded = false;
    private android.content.SharedPreferences prefs;
    private MqttClient mqttClient;
    private int syncMode = 0; // 0: MQTT, 1: IP
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
        binding = ActivityRoWaterBinding.inflate(getLayoutInflater());
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

        initMqtt();
        setupNetworkListener();

        binding.rowPumpStatus.label.setText(R.string.label_pump_status);
        binding.rowPumpStatus.sensorSwitch.setVisibility(android.view.View.VISIBLE);
        
        binding.rowWaterLevel.label.setText(R.string.label_water_level);
        
        binding.rowManual.label.setText(R.string.label_manual_control);
        binding.rowManual.sensorSwitch.setVisibility(android.view.View.VISIBLE);

        binding.btnBack.setOnClickListener(v -> finish());
        binding.btnSync.setOnClickListener(v -> syncData());
        binding.btnSyncHistory.setOnClickListener(v -> syncHistory());
        binding.btnClearHistory.setOnClickListener(v -> clearHistory());
        binding.btnSettings.setOnClickListener(v -> {
            Intent intent = new Intent(this, SmartHomeSettingsActivity.class);
            intent.putExtra("caller_context", "ro_pump");
            startActivity(intent);
        });

        binding.historyHeader.setOnClickListener(v -> toggleHistoryExpansion());

        binding.rowPumpStatus.sensorSwitch.setOnClickListener(v -> {
            updateSensorConfig();
        });
        
        binding.rowManual.sensorSwitch.setOnClickListener(v -> {
            updateSensorConfig();
        });
    }

    private void loadSettings() {
        syncMode = prefs.getInt("sync_mode", 0);
        String savedIp = prefs.getString("local_node_ip", getString(R.string.default_node_ip));
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
            config.put("pump", binding.rowPumpStatus.sensorSwitch.isChecked());
            config.put("manual", binding.rowManual.sensorSwitch.isChecked());
            
            String payload = "CONFIG:" + config.toString();

            if (syncMode == 1 && !selectedNodeIp.isEmpty()) {
                executor.execute(() -> {
                    try {
                        URL url = new URL("http://" + selectedNodeIp + "/config");
                        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                        conn.setRequestMethod("POST");
                        conn.setDoOutput(true);
                        conn.getOutputStream().write(payload.getBytes());
                        conn.getResponseCode();
                        conn.disconnect();
                    } catch (Exception ignored) {}
                });
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
        String clientId = "Android_RO_" + System.currentTimeMillis() % 100000;

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
        if (!selectedNodeId.isEmpty() && !topic.contains(selectedNodeId)) return;
        if (System.currentTimeMillis() - lastInteractionTime < 4000) return;

        runOnUiThread(() -> {
            try {
                JSONObject json = new JSONObject(payload);
                if (json.has("pump")) {
                    isSyncing = true;
                    String pumpState = json.getString("pump");
                    binding.rowPumpStatus.value.setText(pumpState);
                    binding.rowPumpStatus.sensorSwitch.setChecked("ON".equals(pumpState));
                    
                    String level = json.getString("level");
                    binding.rowWaterLevel.value.setText(level);
                    
                    String manualState = json.getString("manual");
                    binding.rowManual.value.setText(manualState);
                    binding.rowManual.sensorSwitch.setChecked("ON".equals(manualState));
                    
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
            if (json.getString("id").startsWith("RO_Pump")) {
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
        executor.shutdown();
        closeMqtt();
    }
}
