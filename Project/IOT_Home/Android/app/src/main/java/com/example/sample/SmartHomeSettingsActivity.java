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
        String fullUrl = binding.otaUrl.getText().toString().trim();
        String minimalUrl = binding.minimalOtaUrl.getText().toString().trim();
        
        if (fullUrl.isEmpty()) {
            Toast.makeText(this, "Please enter Full Firmware URL", Toast.LENGTH_SHORT).show();
            return;
        }

        fullUrl = fixUrl(fullUrl);
        binding.otaUrl.setText(fullUrl);
        
        if (!minimalUrl.isEmpty()) {
            minimalUrl = fixUrl(minimalUrl);
            binding.minimalOtaUrl.setText(minimalUrl);
        }

        String dbUrl = binding.firebaseUrl.getText().toString().trim();
        String targetNode = getTargetNode();

        try {
            FirebaseDatabase database = dbUrl.isEmpty() ? FirebaseDatabase.getInstance() : FirebaseDatabase.getInstance(dbUrl);
            DatabaseReference ref = database.getReference(targetNode).child("command");
            
            if (!minimalUrl.isEmpty()) {
                // Two-step update
                String finalFullUrl = fullUrl;
                String finalMinimalUrl = minimalUrl;
                ref.setValue("OTA_FULL:" + finalFullUrl).addOnCompleteListener(task -> {
                    if (task.isSuccessful()) {
                        Toast.makeText(this, "Step 1: Full URL saved. Sending Bridge...", Toast.LENGTH_SHORT).show();
                        binding.getRoot().postDelayed(() -> {
                            ref.setValue("OTA:" + finalMinimalUrl);
                        }, 2000);
                    }
                });
            } else {
                // Standard single-step update
                ref.setValue("OTA:" + fullUrl).addOnCompleteListener(task -> {
                    if (task.isSuccessful()) {
                        Toast.makeText(this, "OTA Command Sent to " + targetNode, Toast.LENGTH_SHORT).show();
                    }
                });
            }
        } catch (Exception e) {
            Toast.makeText(this, "Error: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private String fixUrl(String url) {
        if (url.contains("dropbox.com") && url.endsWith("dl=0")) {
            return url.replace("dl=0", "dl=1");
        }
        if (url.contains("github.com") && !url.contains("raw.githubusercontent.com")) {
            return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/");
        }
        return url;
    }

    private String getTargetNode() {
        switch (callerContext) {
            case "study": return binding.firebaseStudyNode.getText().toString().trim().isEmpty() ? prefs.getString("firebase_study_node", "") : binding.firebaseStudyNode.getText().toString().trim();
            case "kitchen": return binding.firebaseNode.getText().toString().trim().isEmpty() ? prefs.getString("firebase_node", "") : binding.firebaseNode.getText().toString().trim();
            case "door": return binding.firebaseDoorNode.getText().toString().trim().isEmpty() ? prefs.getString("firebase_door_node", "") : binding.firebaseDoorNode.getText().toString().trim();
            case "toilet": return binding.firebaseToiletNode.getText().toString().trim().isEmpty() ? prefs.getString("firebase_toilet_node", "") : binding.firebaseToiletNode.getText().toString().trim();
            case "ro_pump": return binding.firebaseRoPumpNode.getText().toString().trim().isEmpty() ? prefs.getString("firebase_ro_pump_node", "") : binding.firebaseRoPumpNode.getText().toString().trim();
            case "kitchen_fan": return binding.firebaseKitchenFanNode.getText().toString().trim().isEmpty() ? prefs.getString("firebase_kitchen_fan_node", "") : binding.firebaseKitchenFanNode.getText().toString().trim();
            case "pi_kitchen": return binding.firebasePiKitchenNode.getText().toString().trim().isEmpty() ? prefs.getString("firebase_pi_kitchen_node", "") : binding.firebasePiKitchenNode.getText().toString().trim();
            default: return prefs.getString("firebase_study_node", "");
        }
    }

    private void syncSettingsToNode() {
        boolean isPi = "pi_kitchen".equals(callerContext);
        String ip = isPi ? prefs.getString("pi_kitchen_ip", "") : prefs.getString("local_node_ip", "");
        if (ip.isEmpty()) {
            Toast.makeText(this, "Device IP not set", Toast.LENGTH_SHORT).show();
            return;
        }
        
        final String finalIp = ip;
        String brokerInput = binding.mqttBroker.getText().toString().trim();
        final String mBroker = brokerInput.isEmpty() ? prefs.getString("mqtt_broker", "") : brokerInput;

        String portInput = binding.mqttPort.getText().toString().trim();
        final String mPorts = portInput.isEmpty() ? prefs.getString("mqtt_ports", "") : portInput;

        executor.execute(() -> {
            try {
                int syncMode = prefs.getInt("sync_mode", 0);
                String fbUrl = binding.firebaseUrl.getText().toString().trim();

                URL url = new URL("http://" + finalIp + (isPi ? ":5001" : "") + "/config");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json");
                conn.setDoOutput(true);

                JSONObject json = new JSONObject();
                json.put("mqtt_broker", mBroker);
                String firstPort = mPorts.split(",")[0].trim();
                json.put("mqtt_port", Integer.parseInt(firstPort));
                
                if (isPi) {
                    json.put("fb_url", fbUrl);
                    json.put("fb_node", binding.firebasePiKitchenNode.getText().toString().trim());
                    json.put("fb_secret", binding.firebaseSecret.getText().toString().trim());
                    
                    if (syncMode == 2) { // Firebase mode
                        FirebaseDatabase database = fbUrl.isEmpty() ? FirebaseDatabase.getInstance() : FirebaseDatabase.getInstance(fbUrl);
                        String node = binding.firebasePiKitchenNode.getText().toString().trim();
                        
                        java.util.Map<String, Object> configMap = new java.util.HashMap<>();
                        configMap.put("mqtt_broker", mBroker);
                        configMap.put("fb_url", fbUrl);
                        configMap.put("fb_node", node);
                        configMap.put("fb_secret", binding.firebaseSecret.getText().toString().trim());
                        
                        database.getReference("FrmMobile").child(node).child("config").setValue(configMap);
                    }
                }

                try (OutputStream os = conn.getOutputStream()) {
                    os.write(json.toString().getBytes());
                }

                int code = conn.getResponseCode();
                runOnUiThread(() -> {
                    if (code == 200) {
                        Toast.makeText(this, "Settings Synced to " + (isPi ? "Pi" : "NodeMCU"), Toast.LENGTH_SHORT).show();
                        finish();
                    } else {
                        Toast.makeText(this, "Sync Failed: " + code, Toast.LENGTH_SHORT).show();
                    }
                });
                conn.disconnect();
            } catch (Exception e) {
                Log.e("Settings", "Sync failed", e);
                runOnUiThread(() -> Toast.makeText(this, "Device not reachable at " + finalIp, Toast.LENGTH_SHORT).show());
            }
        });
    }

    private void loadSettings() {
        binding.mqttBroker.setText(prefs.getString("mqtt_broker", AppDefaults.MQTT_BROKER));
        binding.mqttPort.setText(prefs.getString("mqtt_ports", AppDefaults.MQTT_PORTS));
        
        boolean isPi = "pi_kitchen".equals(callerContext);
        binding.localIp.setText(prefs.getString(isPi ? "pi_kitchen_ip" : "local_node_ip", isPi ? AppDefaults.DEFAULT_PI_IP : AppDefaults.DEFAULT_NODE_IP));
        
        binding.csvPath.setText(prefs.getString("csv_path", "IOT_HOME/StudyRoom/StudyRoomMonitor.csv"));
        binding.kitchenPath.setText(prefs.getString("kitchen_path", "IOT_HOME/Kitchen"));
        binding.doorPath.setText(prefs.getString("door_path", "IOT_HOME/MainDoor"));
        
        binding.firebaseUrl.setText(prefs.getString("firebase_url", AppDefaults.FIREBASE_URL));
        binding.firebaseSecret.setText(prefs.getString("firebase_secret", Credentials.FIREBASE_SECRET));
        binding.firebaseEmail.setText(prefs.getString("firebase_email", Credentials.FIREBASE_EMAIL));
        binding.firebasePassword.setText(prefs.getString("firebase_password", Credentials.FIREBASE_PASSWORD));
        binding.firebaseNode.setText(prefs.getString("firebase_node", AppDefaults.NODE_KITCHEN));
        binding.firebaseStudyNode.setText(prefs.getString("firebase_study_node", AppDefaults.NODE_STUDY));
        binding.firebaseDoorNode.setText(prefs.getString("firebase_door_node", AppDefaults.NODE_DOOR));
        binding.firebaseToiletNode.setText(prefs.getString("firebase_toilet_node", AppDefaults.NODE_TOILET));
        binding.firebaseRoPumpNode.setText(prefs.getString("firebase_ro_pump_node", AppDefaults.NODE_RO_PUMP));
        binding.firebaseKitchenFanNode.setText(prefs.getString("firebase_kitchen_fan_node", AppDefaults.NODE_KITCHEN_FAN));
        binding.firebasePiKitchenNode.setText(prefs.getString("firebase_pi_kitchen_node", AppDefaults.NODE_PI_KITCHEN));
        binding.videoWidth.setText(prefs.getString("video_width", AppDefaults.DEFAULT_WIDTH));
        binding.videoHeight.setText(prefs.getString("video_height", AppDefaults.DEFAULT_HEIGHT));
        binding.videoFps.setText(prefs.getString("video_fps", AppDefaults.DEFAULT_FPS));
        binding.streamTargetIpEdit.setText(prefs.getString("stream_target_ip", ""));

        // Context-aware Firebase Node visibility
        binding.containerFirebaseNode.setVisibility(android.view.View.GONE);
        binding.containerFirebaseStudyNode.setVisibility(android.view.View.GONE);
        binding.containerFirebaseDoorNode.setVisibility(android.view.View.GONE);
        binding.containerFirebaseToiletNode.setVisibility(android.view.View.GONE);
        binding.containerFirebaseRoPumpNode.setVisibility(android.view.View.GONE);
        binding.containerFirebaseKitchenFanNode.setVisibility(android.view.View.GONE);
        binding.containerFirebasePiKitchenNode.setVisibility(android.view.View.GONE);

        switch (callerContext) {
            case "kitchen":
                binding.containerFirebaseNode.setVisibility(android.view.View.VISIBLE);
                break;
            case "study":
                binding.containerFirebaseStudyNode.setVisibility(android.view.View.VISIBLE);
                break;
            case "door":
                binding.containerFirebaseDoorNode.setVisibility(android.view.View.VISIBLE);
                break;
            case "toilet":
                binding.containerFirebaseToiletNode.setVisibility(android.view.View.VISIBLE);
                break;
            case "ro_pump":
                binding.containerFirebaseRoPumpNode.setVisibility(android.view.View.VISIBLE);
                break;
            case "kitchen_fan":
                binding.containerFirebaseKitchenFanNode.setVisibility(android.view.View.VISIBLE);
                break;
            case "pi_kitchen":
                binding.containerFirebasePiKitchenNode.setVisibility(android.view.View.VISIBLE);
                break;
        }

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
            binding.containerVideoRes.setVisibility(android.view.View.VISIBLE);
        } else if ("pi_kitchen".equals(callerContext)) {
            binding.containerKitchenPath.setVisibility(android.view.View.VISIBLE);
            binding.containerVideoRes.setVisibility(android.view.View.GONE);
            binding.cardOtaSettings.setVisibility(android.view.View.GONE);
        } else if ("door".equals(callerContext)) {
            binding.containerDoorPath.setVisibility(android.view.View.VISIBLE);
            binding.containerVideoRes.setVisibility(android.view.View.VISIBLE);
        }

        if (!"pi_kitchen".equals(callerContext)) {
            binding.containerMinimalOta.setVisibility(android.view.View.VISIBLE);
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
        binding.minimalOtaUrl.setText(prefs.getString("minimal_ota_url", ""));
    }

    private void saveSettings() {
        String mBroker = binding.mqttBroker.getText().toString().trim();
        String portStr = binding.mqttPort.getText().toString().trim();
        String localIp = binding.localIp.getText().toString().trim();
        String csvPath = binding.csvPath.getText().toString().trim();
        String kitchenPath = binding.kitchenPath.getText().toString().trim();
        String doorPath = binding.doorPath.getText().toString().trim();
        String fbUrl = binding.firebaseUrl.getText().toString().trim();
        if (fbUrl.endsWith("/")) fbUrl = fbUrl.substring(0, fbUrl.length() - 1);
        
        String fbSecret = binding.firebaseSecret.getText().toString().trim();
        String fbEmail = binding.firebaseEmail.getText().toString().trim();
        String fbPassword = binding.firebasePassword.getText().toString().trim();
        String fbNode = binding.firebaseNode.getText().toString().trim();
        String fbStudyNode = binding.firebaseStudyNode.getText().toString().trim();
        String fbDoorNode = binding.firebaseDoorNode.getText().toString().trim();
        String fbToiletNode = binding.firebaseToiletNode.getText().toString().trim();
        String fbRoPumpNode = binding.firebaseRoPumpNode.getText().toString().trim();
        String fbKitchenFanNode = binding.firebaseKitchenFanNode.getText().toString().trim();
        String fbPiKitchenNode = binding.firebasePiKitchenNode.getText().toString().trim();
        
        // Auto-fix Pi Kitchen node if it has the old prefix
        if (fbPiKitchenNode.contains("/")) {
            fbPiKitchenNode = fbPiKitchenNode.substring(fbPiKitchenNode.lastIndexOf("/") + 1);
        }
        String minOtaUrl = binding.minimalOtaUrl.getText().toString().trim();
        String vWidth = binding.videoWidth.getText().toString().trim();
        String vHeight = binding.videoHeight.getText().toString().trim();
        String vFps = binding.videoFps.getText().toString().trim();
        String targetIp = binding.streamTargetIpEdit.getText().toString().trim();

        int mode = 0;
        if (binding.rbModeIp.isChecked()) mode = 1;
        else if (binding.rbModeFirebase.isChecked()) mode = 2;

        SharedPreferences.Editor editor = prefs.edit()
            .putString("mqtt_broker", mBroker)
            .putString("mqtt_ports", portStr)
            .putString("csv_path", csvPath)
            .putString("kitchen_path", kitchenPath)
            .putString("door_path", doorPath)
            .putString("firebase_url", fbUrl)
            .putString("firebase_secret", fbSecret)
            .putString("firebase_email", fbEmail)
            .putString("firebase_password", fbPassword)
            .putString("firebase_node", fbNode)
            .putString("firebase_study_node", fbStudyNode)
            .putString("firebase_door_node", fbDoorNode)
            .putString("firebase_toilet_node", fbToiletNode)
            .putString("firebase_ro_pump_node", fbRoPumpNode)
            .putString("firebase_kitchen_fan_node", fbKitchenFanNode)
            .putString("firebase_pi_kitchen_node", fbPiKitchenNode)
            .putString("minimal_ota_url", minOtaUrl)
            .putString("video_width", vWidth)
            .putString("video_height", vHeight)
            .putString("video_fps", vFps)
            .putString("stream_target_ip", targetIp)
            .putInt("sync_mode", mode);

        if ("pi_kitchen".equals(callerContext)) {
            editor.putString("pi_kitchen_ip", localIp);
        } else {
            editor.putString("local_node_ip", localIp);
        }
        
        editor.apply();
    }
}
