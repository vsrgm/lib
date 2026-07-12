package com.example.sample;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.sample.databinding.ActivitySmartHomeSettingsBinding;

import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import org.json.JSONObject;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SmartHomeSettingsActivity extends AppCompatActivity {

    private ActivitySmartHomeSettingsBinding binding;
    private SharedPreferences prefs;
    private String callerContext = "";
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Apply saved theme
        android.content.SharedPreferences themePrefs = getSharedPreferences("ThemePrefs", MODE_PRIVATE);
        int themeMode = themePrefs.getInt("theme_mode", 0);
        switch (themeMode) {
            case 1: androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_NO); break;
            case 2: androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_YES); break;
            default: androidx.appcompat.app.AppCompatDelegate.setDefaultNightMode(androidx.appcompat.app.AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM); break;
        }

        super.onCreate(savedInstanceState);
        binding = ActivitySmartHomeSettingsBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        prefs = getSharedPreferences("SmartHomePrefs", MODE_PRIVATE);
        callerContext = getIntent().getStringExtra("caller_context");
        if (callerContext == null) callerContext = "";

        loadSettings();

        binding.csvPath.addTextChangedListener(new android.text.TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) {
                binding.tvCsvPathInfo.setText("Current Path: " + s.toString());
            }
            @Override public void afterTextChanged(android.text.Editable s) {}
        });

        binding.btnBack.setOnClickListener(v -> finish());
        binding.btnSave.setOnClickListener(v -> {
            saveSettings();
            syncSettingsToNode();
            Toast.makeText(this, "Settings Saved Locally", Toast.LENGTH_SHORT).show();
        });

        binding.btnTriggerOta.setOnClickListener(v -> triggerOta());
    }

    private void triggerOta() {
        String url = binding.otaUrl.getText().toString().trim();
        if (url.isEmpty()) {
            Toast.makeText(this, "Please enter Firmware URL", Toast.LENGTH_SHORT).show();
            return;
        }

        // Auto-fix Dropbox links
        if (url.contains("dropbox.com") && url.endsWith("dl=0")) {
            url = url.replace("dl=0", "dl=1");
            binding.otaUrl.setText(url);
            Toast.makeText(this, "Dropbox link auto-formatted for direct download", Toast.LENGTH_SHORT).show();
        }

        // Auto-fix GitHub links to raw content
        if (url.contains("github.com") && !url.contains("raw.githubusercontent.com")) {
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/");
            binding.otaUrl.setText(url);
            Toast.makeText(this, "GitHub link converted to Direct Raw URL", Toast.LENGTH_SHORT).show();
        }

        String dbUrl = binding.firebaseUrl.getText().toString().trim();
        String targetNode = "";

        // Automatically decide node based on caller context
        switch (callerContext) {
            case "study":
                targetNode = binding.firebaseStudyNode.getText().toString().trim();
                if (targetNode.isEmpty() || "smart_home/study".equals(targetNode)) targetNode = "FrmMobile/study";
                break;
            case "kitchen":
                targetNode = binding.firebaseNode.getText().toString().trim();
                if (targetNode.isEmpty()) targetNode = "smart_home/kitchen";
                break;
            case "door":
                targetNode = binding.firebaseDoorNode.getText().toString().trim();
                if (targetNode.isEmpty()) targetNode = "smart_home/main_door";
                break;
            case "toilet":
                targetNode = binding.firebaseToiletNode.getText().toString().trim();
                if (targetNode.isEmpty()) targetNode = "smart_home/toilet";
                break;
            default:
                targetNode = binding.firebaseStudyNode.getText().toString().trim();
                if (targetNode.isEmpty() || "smart_home/study".equals(targetNode)) targetNode = "FrmMobile/study";
                break;
        }

        final String finalTargetNode = targetNode;
        try {
            FirebaseDatabase database = dbUrl.isEmpty() ? FirebaseDatabase.getInstance() : FirebaseDatabase.getInstance(dbUrl);
            DatabaseReference ref = database.getReference(finalTargetNode).child("command");
            
            ref.setValue("OTA:" + url).addOnCompleteListener(task -> {
                if (task.isSuccessful()) {
                    Toast.makeText(this, "OTA Command Sent to " + finalTargetNode, Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(this, "Failed to send OTA command", Toast.LENGTH_SHORT).show();
                }
            });
        } catch (Exception e) {
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void syncSettingsToNode() {
        String ip = prefs.getString("local_node_ip", "192.168.0.107");
        String brokerInput = binding.mqttBroker.getText().toString().trim();
        final String mBroker = brokerInput.isEmpty() ? getString(R.string.default_mqtt_broker) : brokerInput;

        String portInput = binding.mqttPort.getText().toString().trim();
        final String mPorts = portInput.isEmpty() ? getString(R.string.default_mqtt_port) : portInput;

        executor.execute(() -> {
            try {
                URL url = new URL("http://" + ip + "/config");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setDoOutput(true);

                JSONObject json = new JSONObject();
                json.put("mqtt_broker", mBroker);
                // For simplicity, we send the first port to the node if it expects an int,
                // or we can adjust the node firmware to handle multiple ports if needed.
                // Based on Automation requirement.txt 1.1.3.3, it's the Mobile app that tries multiple ports.
                String firstPort = mPorts.split(",")[0].trim();
                json.put("mqtt_port", Integer.parseInt(firstPort));

                try (OutputStream os = conn.getOutputStream()) {
                    os.write(json.toString().getBytes());
                }

                int code = conn.getResponseCode();
                runOnUiThread(() -> {
                    if (code == 200) {
                        Toast.makeText(this, "Settings Synced to NodeMCU", Toast.LENGTH_SHORT).show();
                        finish();
                    } else {
                        Toast.makeText(this, "Sync to Node Failed: " + code, Toast.LENGTH_SHORT).show();
                    }
                });
                conn.disconnect();
            } catch (Exception e) {
                Log.e("Settings", "Sync failed", e);
                runOnUiThread(() -> Toast.makeText(this, "NodeMCU not reachable", Toast.LENGTH_SHORT).show());
            }
        });
    }

    private void loadSettings() {
        binding.mqttBroker.setText(prefs.getString("mqtt_broker", "broker.hivemq.com"));
        binding.mqttPort.setText(prefs.getString("mqtt_ports", "1883,8000"));
        binding.localIp.setText(prefs.getString("local_node_ip", "192.168.0.107"));
        
        binding.csvPath.setText(prefs.getString("csv_path", "IOT_HOME/StudyRoom/StudyRoomMonitor.csv"));
        binding.kitchenPath.setText(prefs.getString("kitchen_path", "IOT_HOME/Kitchen"));
        binding.doorPath.setText(prefs.getString("door_path", "IOT_HOME/MainDoor"));
        
        binding.firebaseUrl.setText(prefs.getString("firebase_url", "https://gapsmarthome-default-rtdb.asia-southeast1.firebasedatabase.app/"));
        binding.firebaseNode.setText(prefs.getString("firebase_node", "smart_home/kitchen"));
        binding.firebaseStudyNode.setText(prefs.getString("firebase_study_node", "FrmMobile/study"));
        binding.firebaseDoorNode.setText(prefs.getString("firebase_door_node", "smart_home/main_door"));
        binding.firebaseToiletNode.setText(prefs.getString("firebase_toilet_node", "smart_home/toilet"));

        // Context-aware Storage Settings visibility
        binding.containerCsvPath.setVisibility(android.view.View.GONE);
        binding.containerKitchenPath.setVisibility(android.view.View.GONE);
        binding.containerDoorPath.setVisibility(android.view.View.GONE);
        binding.tvCsvPathInfo.setVisibility(android.view.View.GONE);

        if ("study".equals(callerContext)) {
            binding.containerCsvPath.setVisibility(android.view.View.VISIBLE);
            binding.tvCsvPathInfo.setVisibility(android.view.View.VISIBLE);
        } else if ("kitchen".equals(callerContext)) {
            binding.containerKitchenPath.setVisibility(android.view.View.VISIBLE);
        } else if ("door".equals(callerContext)) {
            binding.containerDoorPath.setVisibility(android.view.View.VISIBLE);
        }

        binding.tvCsvPathInfo.setText("Path: [Downloads]/" + binding.csvPath.getText().toString());
        
        int mode = prefs.getInt("sync_mode", 0); // 0: MQTT, 1: IP, 2: Firebase
        if (mode == 0) {
            binding.rbModeMqtt.setChecked(true);
        } else if (mode == 1) {
            binding.rbModeIp.setChecked(true);
        } else {
            binding.rbModeFirebase.setChecked(true);
            binding.cardFirebaseSettings.setVisibility(android.view.View.VISIBLE);
        }

        binding.rgSyncMode.setOnCheckedChangeListener((group, checkedId) -> {
            binding.cardFirebaseSettings.setVisibility(checkedId == R.id.rb_mode_firebase ? android.view.View.VISIBLE : android.view.View.GONE);
        });
    }

    private void saveSettings() {
        String mBroker = binding.mqttBroker.getText().toString().trim();
        String portStr = binding.mqttPort.getText().toString().trim();
        String localIp = binding.localIp.getText().toString().trim();
        String csvPath = binding.csvPath.getText().toString().trim();
        String kitchenPath = binding.kitchenPath.getText().toString().trim();
        String doorPath = binding.doorPath.getText().toString().trim();
        String fbUrl = binding.firebaseUrl.getText().toString().trim();
        String fbNode = binding.firebaseNode.getText().toString().trim();
        String fbStudyNode = binding.firebaseStudyNode.getText().toString().trim();
        String fbDoorNode = binding.firebaseDoorNode.getText().toString().trim();
        String fbToiletNode = binding.firebaseToiletNode.getText().toString().trim();

        int mode = 0;
        if (binding.rbModeIp.isChecked()) mode = 1;
        else if (binding.rbModeFirebase.isChecked()) mode = 2;

        prefs.edit()
            .putString("mqtt_broker", mBroker)
            .putString("mqtt_ports", portStr)
            .putString("local_node_ip", localIp)
            .putString("csv_path", csvPath)
            .putString("kitchen_path", kitchenPath)
            .putString("door_path", doorPath)
            .putString("firebase_url", fbUrl)
            .putString("firebase_node", fbNode)
            .putString("firebase_study_node", fbStudyNode)
            .putString("firebase_door_node", fbDoorNode)
            .putString("firebase_toilet_node", fbToiletNode)
            .putInt("sync_mode", mode)
            .apply();
    }
}
