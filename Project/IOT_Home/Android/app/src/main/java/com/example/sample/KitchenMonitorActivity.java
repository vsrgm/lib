package com.example.sample;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.webkit.WebSettings;
import android.webkit.WebViewClient;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.sample.databinding.ActivityKitchenMonitorBinding;

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
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkRequest;
import android.net.NetworkCapabilities;

public class KitchenMonitorActivity extends AppCompatActivity {

    private ActivityKitchenMonitorBinding binding;
    private MqttClient mqttClient;
    private DatabaseReference firebaseRef;
    private ValueEventListener firebaseListener;
    private final ExecutorService executor = Executors.newFixedThreadPool(2);
    private final String baseTopic = "smart_home/kitchen/";
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
        binding = ActivityKitchenMonitorBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        prefs = getSharedPreferences("SmartHomePrefs", MODE_PRIVATE);
        syncMode = prefs.getInt("sync_mode", 0);

        binding.btnBack.setOnClickListener(v -> finish());
        binding.btnSettings.setOnClickListener(v -> {
            Intent intent = new Intent(this, SmartHomeSettingsActivity.class);
            intent.putExtra("caller_context", "kitchen");
            startActivity(intent);
        });
        
        binding.btnCapture.setOnClickListener(v -> sendCommand("CAPTURE"));
        binding.btnGallery.setOnClickListener(v -> {
            startActivity(new android.content.Intent(this, ImageViewerActivity.class));
        });

        // Initialize labels and switches
        binding.rowPir.label.setText("PIR Motion");
        binding.rowLdr.label.setText("Light Sensor (LDR)");
        binding.rowPower.label.setText("External Power");
        
        binding.rowRelay.label.setText("Kitchen Lamp (Relay)");
        binding.rowRelay.sensorSwitch.setVisibility(View.VISIBLE);
        binding.rowRelay.sensorSwitch.setEnabled(false); // Locked until manual override
        binding.rowRelay.sensorSwitch.setOnClickListener(v -> {
            sendCommand(binding.rowRelay.sensorSwitch.isChecked() ? "RELAY_ON" : "RELAY_OFF");
        });

        binding.rowBuzzer.label.setText("Alarm (Buzzer)");
        binding.rowBuzzer.sensorSwitch.setVisibility(View.VISIBLE);
        binding.rowBuzzer.sensorSwitch.setEnabled(false); // Locked until manual override
        binding.rowBuzzer.sensorSwitch.setOnClickListener(v -> {
            sendCommand(binding.rowBuzzer.sensorSwitch.isChecked() ? "BUZZER_ON" : "BUZZER_OFF");
        });

        binding.rowManualOverride.label.setText("Manual Override");
        binding.rowManualOverride.sensorSwitch.setVisibility(View.VISIBLE);
        binding.rowManualOverride.sensorSwitch.setOnClickListener(v -> {
            boolean isChecked = binding.rowManualOverride.sensorSwitch.isChecked();
            binding.rowRelay.sensorSwitch.setEnabled(isChecked);
            binding.rowBuzzer.sensorSwitch.setEnabled(isChecked);
            sendCommand(isChecked ? "OVERRIDE_ON" : "OVERRIDE_OFF");
        });

        binding.rowGasDigital.label.setText("Gas Leak (Digital)");
        binding.rowGasAnalog.label.setText("Gas Level (Analog)");
        binding.rowTempLm358.label.setText("Temp (LM358)");
        binding.rowTempBmp.label.setText("Temp (BMP280)");
        binding.rowPres.label.setText("Pressure (BMP280)");

        setupWebView();
        syncResolution();

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
        String node = prefs.getString("firebase_node", AppDefaults.NODE_KITCHEN);
        
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

    @Override
    protected void onResume() {
        super.onResume();
        syncMode = prefs.getInt("sync_mode", 0);
        setupWebView();
        syncResolution();
    }

    private void syncResolution() {
        String width = prefs.getString("video_width", "640");
        String height = prefs.getString("video_height", "480");
        sendCommand("RES:" + width + "," + height);
    }

    private void setupWebView() {
        WebSettings webSettings = binding.videoStream.getSettings();
        webSettings.setJavaScriptEnabled(true);
        webSettings.setUseWideViewPort(true);
        webSettings.setLoadWithOverviewMode(true);
        binding.videoStream.setWebViewClient(new WebViewClient());
        
        if (syncMode == 1) {
            // Local IP: Show MJPEG Stream
            binding.videoStream.setVisibility(View.VISIBLE);
            binding.ivCapturedImage.setVisibility(View.GONE);
            String ip = prefs.getString("local_node_ip", AppDefaults.DEFAULT_NODE_IP);
            binding.videoStream.loadUrl("http://" + ip + "/stream");
        } else {
            // MQTT/Firebase: Show static images
            binding.videoStream.setVisibility(View.GONE);
            binding.ivCapturedImage.setVisibility(View.VISIBLE);
            binding.ivCapturedImage.setImageResource(android.R.drawable.ic_menu_camera);
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
                    Log.e("Kitchen", "IP sync failed", e);
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
        String clientId = "Android_Kitchen_" + System.currentTimeMillis();

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
                        if (topic.equals(baseTopic + "status")) {
                            String payload = new String(message.getPayload());
                            handleStatus(payload);
                        } else if (topic.equals(baseTopic + "image")) {
                            saveMqttImage(message.getPayload());
                        }
                    }
                    @Override public void deliveryComplete(IMqttDeliveryToken token) {}
                });

                mqttClient.connect(options);
                mqttClient.subscribe(baseTopic + "status");
                mqttClient.subscribe(baseTopic + "image");

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
                // Update UI first
                android.graphics.Bitmap bitmap = android.graphics.BitmapFactory.decodeByteArray(data, 0, data.length);
                if (bitmap != null) {
                    runOnUiThread(() -> binding.ivCapturedImage.setImageBitmap(bitmap));
                }

                String subDir = prefs.getString("kitchen_path", getString(R.string.default_kitchen_path));
                File directory = StorageUtils.getWorkDir(subDir);
                String fileName = "IMG_KITCHEN_" + System.currentTimeMillis() + ".jpg";
                File file = new File(directory, fileName);
                java.io.FileOutputStream fos = new java.io.FileOutputStream(file);
                fos.write(data);
                fos.close();
                runOnUiThread(() -> Toast.makeText(this, "Image received and saved", Toast.LENGTH_SHORT).show());
            } catch (Exception e) {
                Log.e("Kitchen", "Failed to save MQTT image", e);
            }
        });
    }

    private void handleStatus(String payload) {
        runOnUiThread(() -> {
            try {
                JSONObject json = new JSONObject(payload);
                binding.rowPir.value.setText(json.optBoolean("pir", false) ? "Motion" : "Clear");
                binding.rowLdr.value.setText(String.valueOf(json.optInt("ldr", 0)));
                binding.rowPower.value.setText(json.optBoolean("pwr", false) ? "ON" : "OFF");
                
                boolean manual = json.optBoolean("manual_override", false);
                binding.rowManualOverride.value.setText(manual ? "ON" : "OFF");
                binding.rowManualOverride.sensorSwitch.setChecked(manual);
                
                binding.rowRelay.sensorSwitch.setEnabled(manual);
                binding.rowBuzzer.sensorSwitch.setEnabled(manual);

                boolean relay = json.getBoolean("relay");
                binding.rowRelay.value.setText(relay ? "ON" : "OFF");
                binding.rowRelay.sensorSwitch.setChecked(relay);

                boolean buzzer = json.optBoolean("buzzer", false);
                binding.rowBuzzer.value.setText(buzzer ? "ON" : "OFF");
                binding.rowBuzzer.sensorSwitch.setChecked(buzzer);

                binding.rowGasDigital.value.setText(json.optBoolean("gas_d", false) ? "LEAK!" : "Normal");
                int gasA = json.optInt("gas_a", 0);
                binding.rowGasAnalog.value.setText(String.valueOf(gasA));
                
                binding.rowTempLm358.value.setText(String.format("%.1f °C", json.optDouble("temp_lm358", 0.0)));
                binding.rowTempBmp.value.setText(String.format("%.1f °C", json.optDouble("temp_bmp", 0.0)));
                binding.rowPres.value.setText(String.format("%.1f hPa", json.optDouble("pres", 0.0)));
                
                String time = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
                binding.syncStatus.setText("Last Update: " + time);

                if (gasA > 20000 || json.optBoolean("gas_d", false)) {
                    binding.rowGasDigital.value.setTextColor(Color.RED);
                    binding.rowGasAnalog.value.setTextColor(Color.RED);
                    Toast.makeText(this, "GAS LEAK DETECTED!", Toast.LENGTH_SHORT).show();
                } else {
                    binding.rowGasDigital.value.setTextColor(Color.GRAY);
                    binding.rowGasAnalog.value.setTextColor(Color.GRAY);
                }
                
                addHistoryRow(json);
            } catch (Exception ignored) {}
        });
    }

    private void addHistoryRow(JSONObject json) {
        try {
            TableRow row = new TableRow(this);
            String time = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
            String event = json.getBoolean("pir") ? "Motion" : "Periodic";
            if (json.optInt("gas_a", 0) > 20000 || json.optBoolean("gas_d", false)) event = "GAS ALERT";

            String[] cols = {time, event, String.valueOf(json.optInt("gas_a", 0))};
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
