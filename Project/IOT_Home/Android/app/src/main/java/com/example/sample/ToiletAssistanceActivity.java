package com.example.sample;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TableRow;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.sample.databinding.ActivityToiletAssistanceBinding;

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
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkRequest;
import android.net.NetworkCapabilities;

public class ToiletAssistanceActivity extends AppCompatActivity {

    private ActivityToiletAssistanceBinding binding;
    private MqttClient mqttClient;
    private DatabaseReference firebaseRef;
    private ValueEventListener firebaseListener;
    private final ExecutorService executor = Executors.newFixedThreadPool(2);
    private final String baseTopic = "smart_home/toilet/";
    private SharedPreferences prefs;
    private int syncMode = 0; // 0: MQTT, 1: IP, 2: Firebase
    private final StringBuilder logBuilder = new StringBuilder();
    private int[] mqttPorts = {};
    private int currentPortIndex = 0;
    private ConnectivityManager connectivityManager;
    private ConnectivityManager.NetworkCallback networkCallback;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityToiletAssistanceBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        prefs = getSharedPreferences("SmartHomePrefs", MODE_PRIVATE);
        syncMode = prefs.getInt("sync_mode", 0);

        binding.btnBack.setOnClickListener(v -> finish());
        binding.btnSettings.setOnClickListener(v -> {
            Intent intent = new Intent(this, SmartHomeSettingsActivity.class);
            intent.putExtra("caller_context", "toilet");
            startActivity(intent);
        });

        // Initialize labels
        binding.rowPir.label.setText("PIR Motion");
        binding.rowLdr.label.setText("Light Sensor");
        
        binding.rowExhaust.label.setText(getString(R.string.label_exhaust_fan));
        binding.rowExhaust.sensorSwitch.setVisibility(View.VISIBLE);
        binding.rowExhaust.sensorSwitch.setOnClickListener(v -> {
            boolean isChecked = binding.rowExhaust.sensorSwitch.isChecked();
            sendCommand(isChecked ? "FAN_ON" : "FAN_OFF");
        });

        binding.rowEmergencyLight.label.setText(getString(R.string.label_emergency_light));
        binding.rowEmergencyLight.sensorSwitch.setVisibility(View.VISIBLE);
        binding.rowEmergencyLight.sensorSwitch.setOnClickListener(v -> {
            boolean isChecked = binding.rowEmergencyLight.sensorSwitch.isChecked();
            sendCommand(isChecked ? "LIGHT_ON" : "LIGHT_OFF");
        });

        binding.rowBuzzer.label.setText(getString(R.string.label_buzzer));
        binding.rowBuzzer.sensorSwitch.setVisibility(View.VISIBLE);
        binding.rowBuzzer.sensorSwitch.setOnClickListener(v -> {
            boolean isChecked = binding.rowBuzzer.sensorSwitch.isChecked();
            sendCommand(isChecked ? "BUZZER_ON" : "BUZZER_OFF");
        });

        binding.rowTemp.label.setText("Temperature");
        binding.rowHum.label.setText("Humidity");

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
        String node = prefs.getString("firebase_toilet_node", AppDefaults.NODE_TOILET);
        
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
                    Log.e("Toilet", "IP sync failed", e);
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
        String clientId = "Android_Toilet_" + System.currentTimeMillis();

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
                        String payload = new String(message.getPayload());
                        if (topic.equals(baseTopic + "status")) handleStatus(payload);
                    }
                    @Override public void deliveryComplete(IMqttDeliveryToken token) {}
                });

                mqttClient.connect(options);
                mqttClient.subscribe(baseTopic + "status");

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

    private void handleStatus(String payload) {
        runOnUiThread(() -> {
            try {
                JSONObject json = new JSONObject(payload);
                binding.rowPir.value.setText(json.getBoolean("pir") ? "Motion" : "Clear");
                binding.rowLdr.value.setText(json.getBoolean("ldr") ? "Low" : "High");
                
                boolean fan = json.getBoolean("fan");
                binding.rowExhaust.value.setText(fan ? "ON" : "OFF");
                binding.rowExhaust.sensorSwitch.setChecked(fan);

                boolean light = json.getBoolean("light");
                binding.rowEmergencyLight.value.setText(light ? "ON" : "OFF");
                binding.rowEmergencyLight.sensorSwitch.setChecked(light);

                boolean buzzer = json.getBoolean("buzzer");
                binding.rowBuzzer.value.setText(buzzer ? "ON" : "OFF");
                binding.rowBuzzer.sensorSwitch.setChecked(buzzer);

                binding.rowTemp.value.setText(String.format("%.1f °C", json.getDouble("temp")));
                binding.rowHum.value.setText(String.format("%.1f %%", json.getDouble("hum")));
                
                String time = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
                binding.syncStatus.setText("Last Update: " + time);
                
                addHistoryRow(json);
            } catch (Exception ignored) {}
        });
    }

    private void addHistoryRow(JSONObject json) {
        try {
            TableRow row = new TableRow(this);
            String time = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
            String event = json.getBoolean("pir") ? "Motion" : "Periodic";

            String[] cols = {time, event, json.getBoolean("fan") ? "Fan ON" : "Fan OFF"};
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
                } catch (Exception e) {}
            });
        } else if (syncMode == 2) {
            if (firebaseRef != null) {
                firebaseRef.child("command").setValue(cmd);
            }
        } else {
            executor.execute(() -> {
                try {
                    if (mqttClient != null && mqttClient.isConnected()) {
                        mqttClient.publish(baseTopic + "commands", new MqttMessage(cmd.getBytes()));
                    }
                } catch (Exception e) {}
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
