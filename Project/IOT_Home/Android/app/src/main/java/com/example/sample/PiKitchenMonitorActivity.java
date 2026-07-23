package com.example.sample;

import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Rect;
import android.media.MediaCodec;
import android.media.MediaFormat;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.example.sample.databinding.ActivityPiKitchenMonitorBinding;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.auth.FirebaseAuth;

import org.eclipse.paho.client.mqttv3.MqttCallback;
import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;

public class PiKitchenMonitorActivity extends AppCompatActivity {

    private ActivityPiKitchenMonitorBinding binding;
    private SharedPreferences prefs;
    private final ExecutorService executor = Executors.newFixedThreadPool(8);
    private DatagramSocket udpSocket;
    private DatagramSocket statsSocket;
    private final AtomicBoolean isStreaming = new AtomicBoolean(false);
    private final AtomicBoolean isFirebaseStreaming = new AtomicBoolean(false);

    private final android.os.Handler heartbeatHandler = new android.os.Handler(android.os.Looper.getMainLooper());
    private final Runnable heartbeatRunnable = new Runnable() {
        @Override
        public void run() {
            if (isStreaming.get()) {
                updatePresence(true);
                heartbeatHandler.postDelayed(this, 10000);
            }
        }
    };
    private String piIp = "";
    private int piPort = 5001;
    private int localUdpPort = 5000;
    private MqttClient mqttClient;
    private DatabaseReference firebaseRef;
    private DatabaseReference cmdRef;
    
    private List<String> imageList = new ArrayList<>();
    private int currentImageIndex = -1;
    private GestureDetector gestureDetector;

    private MediaCodec decoder;
    private android.view.Surface savedSurface;
    private final ByteArrayOutputStream mjpegBuffer = new ByteArrayOutputStream();
    
    private long totalBytesReceived = 0;
    private volatile long serverTotalBytesSent = 0;
    private long frameCount = 0;
    private long lastStatsTime = 0;
    private long lastByteCount = 0;
    private final StringBuilder errorBuilder = new StringBuilder();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityPiKitchenMonitorBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.videoSurface.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(@NonNull SurfaceHolder holder) {
                savedSurface = holder.getSurface();
            }
            @Override public void surfaceChanged(@NonNull SurfaceHolder h, int f, int w, int h1) {}
            @Override public void surfaceDestroyed(@NonNull SurfaceHolder h) { 
                savedSurface = null;
                releaseDecoder(); 
            }
        });

        prefs = getSharedPreferences("SmartHomePrefs", MODE_PRIVATE);
        piIp = prefs.getString("pi_kitchen_ip", AppDefaults.DEFAULT_PI_IP);
        
        binding.etStreamW.setText(prefs.getString("video_width", AppDefaults.DEFAULT_WIDTH));
        binding.etStreamH.setText(prefs.getString("video_height", AppDefaults.DEFAULT_HEIGHT));
        binding.etStreamFps.setText(prefs.getString("video_fps", AppDefaults.DEFAULT_FPS));

        setupControls();
        binding.tvConnectionDetails.setText("Pi IP: " + piIp + "\nControl Port: " + piPort + "\nUDP Port: " + localUdpPort);

        binding.btnBack.setOnClickListener(v -> finish());
        binding.btnSettings.setOnClickListener(v -> {
            android.content.Intent intent = new android.content.Intent(this, SmartHomeSettingsActivity.class);
            intent.putExtra("caller_context", "pi_kitchen");
            startActivity(intent);
        });
        binding.btnStartStream.setOnClickListener(v -> {
            saveStreamConfig();
            startStream();
        });
        binding.btnStopStream.setOnClickListener(v -> stopStream());
        binding.btnRefreshCaptures.setOnClickListener(v -> refreshImageList());
        binding.btnCloseOverlay.setOnClickListener(v -> {
            binding.ivOverlayCapture.setVisibility(View.GONE);
            binding.btnCloseOverlay.setVisibility(View.GONE);
        });
        binding.btnFullscreen.setOnClickListener(v -> toggleFullscreen());
        binding.btnCloseFull.setOnClickListener(v -> binding.fullImageOverlay.setVisibility(View.GONE));
        binding.btnSendShell.setOnClickListener(v -> sendShellCommand());
        binding.btnStopShell.setOnClickListener(v -> stopShellCommand());
        binding.btnCreateFile.setOnClickListener(v -> showCreateFileDialog());
        binding.btnResetControls.setOnClickListener(v -> resetControls());
        binding.cardErrorInfo.setOnClickListener(v -> {
            errorBuilder.setLength(0);
            binding.cardErrorInfo.setVisibility(View.GONE);
        });
        binding.etShellInput.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEND) {
                sendShellCommand();
                return true;
            }
            return false;
        });

        setupSwipeNavigation();
        initMqtt();
        authenticateFirebase();
    }

    private void addError(String message) {
        String time = new java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault()).format(new java.util.Date());
        String errorEntry = "[" + time + "] " + message + "\n";
        errorBuilder.insert(0, errorEntry);
        Log.e("PiMonitor", message);
        runOnUiThread(() -> {
            binding.cardErrorInfo.setVisibility(View.VISIBLE);
            binding.tvErrorDetails.setText(errorBuilder.toString());
        });
    }

    private void setupControls() {
        // We will now populate controls dynamically from fetchControls()
    }

    private void applyControlRemote(String id, int value) {
        executor.execute(() -> {
            boolean success = false;
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/set_control?id=" + id + "&val=" + value);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(2000);
                if (conn.getResponseCode() == 200) success = true;
                conn.disconnect();
            } catch (Exception ignored) {}
            if (!success) sendShellCommandRemote("set_control:" + id + ":" + value);
        });
    }

    private void addDynamicControl(JSONObject ctrl) {
        try {
            int id = ctrl.getInt("id");
            String name = ctrl.getString("name");
            int min = ctrl.getInt("min");
            int max = ctrl.getInt("max");
            int val = ctrl.getInt("val");
            int type = ctrl.getInt("type");

            runOnUiThread(() -> {
                android.widget.LinearLayout container = new android.widget.LinearLayout(this);
                container.setOrientation(android.widget.LinearLayout.VERTICAL);
                container.setPadding(0, 8, 0, 8);

                android.widget.TextView label = new android.widget.TextView(this);
                label.setText(String.format(Locale.US, "%s: %d", name, val));
                label.setTextColor(Color.WHITE);
                label.setTextSize(12);
                container.addView(label);

                if (min < max) {
                    com.google.android.material.slider.Slider slider = new com.google.android.material.slider.Slider(this);
                    slider.setValueFrom(min);
                    slider.setValueTo(max);
                    slider.setValue(Math.max(min, Math.min(max, (float)val)));
                    slider.addOnChangeListener((s, value, fromUser) -> {
                        label.setText(String.format(Locale.US, "%s: %d", name, (int)value));
                        if (fromUser) applyControlRemote(String.valueOf(id), (int)value);
                    });
                    container.addView(slider);
                }

                binding.layoutDynamicControls.addView(container);
            });
        } catch (Exception e) {
            Log.e("PiMonitor", "Error adding dynamic control", e);
        }
    }

    private void fetchControls() {
        int syncMode = prefs.getInt("sync_mode", 0);
        if (syncMode == 2) {
            triggerFirebaseControlsRefresh();
            return;
        }

        executor.execute(() -> {
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/query_controls");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(2500);
                if (conn.getResponseCode() == 200) {
                    InputStream is = conn.getInputStream();
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    byte[] buffer = new byte[8192];
                    int len;
                    while ((len = is.read(buffer)) != -1) baos.write(buffer, 0, len);
                    JSONObject json = new JSONObject(baos.toString());
                    JSONArray controls = json.getJSONArray("controls");

                    runOnUiThread(() -> binding.layoutDynamicControls.removeAllViews());
                    for (int i = 0; i < controls.length(); i++) {
                        addDynamicControl(controls.getJSONObject(i));
                    }
                } else {
                    triggerFirebaseControlsRefresh();
                }
                conn.disconnect();
            } catch (Exception e) {
                Log.w("PiMonitor", "Local controls fetch failed.");
                triggerFirebaseControlsRefresh();
            }
        });
    }

    private void triggerFirebaseControlsRefresh() {
        sendShellCommandRemote("query_controls");
    }

    private void resetControls() {
        executor.execute(() -> {
            boolean success = false;
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/reset_controls");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                if (conn.getResponseCode() == 200) {
                    success = true;
                    runOnUiThread(() -> {
                        Toast.makeText(this, "Controls Reset to Defaults", Toast.LENGTH_SHORT).show();
                        fetchControls(); // Refresh the UI sliders
                    });
                }
                conn.disconnect();
            } catch (Exception e) {
                addError("Reset failed: " + e.getMessage());
            }
            if (!success) sendShellCommandRemote("reset_controls");
        });
    }

    private void authenticateFirebase() {
        String email = prefs.getString("firebase_email", Credentials.FIREBASE_EMAIL);
        String password = prefs.getString("firebase_password", Credentials.FIREBASE_PASSWORD);
        
        if (email.isEmpty() || password.isEmpty()) {
            initFirebase();
            initShellListener();
            refreshImageList();
            return;
        }

        FirebaseAuth.getInstance().signInWithEmailAndPassword(email, password)
            .addOnCompleteListener(task -> {
                if (task.isSuccessful()) {
                    initFirebase();
                    initShellListener();
                    refreshImageList();
                } else {
                    String error = task.getException() != null ? task.getException().getMessage() : "Unknown";
                    addError("Firebase Auth Failed: " + error);
                    runOnUiThread(() -> Toast.makeText(this, "Firebase Auth Failed: " + error, Toast.LENGTH_SHORT).show());
                }
            });
    }

    private final Map<String, ImageView> pendingThumbnails = new ConcurrentHashMap<>();

    private void initShellListener() {
        try {
            firebaseRef.child("shell").addValueEventListener(new ValueEventListener() {
                @Override
                public void onDataChange(DataSnapshot snapshot) {
                    if (snapshot.exists() && snapshot.hasChild("response")) {
                        String resId = snapshot.child("res_id").getValue(String.class);
                        String lastSentId = prefs.getString("last_shell_id", "");
                        if (resId != null && resId.equals(lastSentId)) {
                            String response = snapshot.child("response").getValue(String.class);
                            if (response != null && response.startsWith("IMG_LIST:")) {
                                handleShellImageList(response.substring(9));
                            } else if (response != null && response.startsWith("IMG_DATA:")) {
                                handleShellImageData(response.substring(9));
                            } else {
                                runOnUiThread(() -> {
                                    binding.tvShellOutput.append("\n> Remote Command executed.");
                                    binding.shellOutputScroll.post(() -> binding.shellOutputScroll.fullScroll(View.FOCUS_DOWN));
                                });
                            }
                        }
                    }
                }
                @Override public void onCancelled(DatabaseError error) {}
            });
        } catch (Exception ignored) {}
    }

    private void handleShellImageData(String base64) {
        try {
            byte[] decodedString = android.util.Base64.decode(base64, android.util.Base64.DEFAULT);
            Bitmap bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
            if (bitmap != null) {
                runOnUiThread(() -> {
                    // Update full image if visible
                    if (binding.ivOverlayCapture.getVisibility() == View.VISIBLE) {
                        binding.ivOverlayCapture.setImageBitmap(bitmap);
                    }
                    // Also check if any thumbnail was waiting for this
                    // (Actually we don't have the filename here unless we include it in response)
                    // For now, just show in the main preview if visible
                });
            }
        } catch (Exception ignored) {}
    }

    private void handleShellImageList(String data) {
        try {
            JSONArray images = new JSONArray(data);
            List<String> sortedList = new ArrayList<>();
            for (int i = 0; i < images.length(); i++) sortedList.add(images.getString(i));
            Collections.sort(sortedList, Collections.reverseOrder());
            this.imageList = sortedList;
            runOnUiThread(() -> {
                binding.layoutCaptures.removeAllViews();
                for (String name : sortedList) addCaptureIcon(name);
            });
        } catch (Exception ignored) {}
    }

    private boolean isProcessRunning = false;

    private void sendShellCommand() {
        String cmd = binding.etShellInput.getText().toString().trim();
        if (cmd.isEmpty()) return;

        if (isProcessRunning) {
            binding.tvShellOutput.append("\n> " + cmd);
            sendShellStdin(cmd);
            binding.etShellInput.setText("");
            return;
        }

        binding.tvShellOutput.append("\n$ " + cmd);
        binding.etShellInput.setText("");
        binding.shellOutputScroll.post(() -> binding.shellOutputScroll.fullScroll(View.FOCUS_DOWN));

        isProcessRunning = true; // Mark as busy
        executor.execute(() -> {
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/shell?cmd=" + cmd.replace(" ", "+"));
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestProperty("Connection", "close");
                conn.setConnectTimeout(5000); 
                conn.setReadTimeout(10000);
                if (conn.getResponseCode() == 200) {
                    InputStream is = conn.getInputStream();
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    byte[] buffer = new byte[4096];
                    int len;
                    while ((len = is.read(buffer)) != -1) baos.write(buffer, 0, len);
                    
                    JSONObject json = new JSONObject(baos.toString());
                    String output = json.optString("output", "");
                    runOnUiThread(() -> {
                        if (!output.isEmpty()) binding.tvShellOutput.append("\n" + output.replace("\\n", "\n"));
                        binding.shellOutputScroll.post(() -> binding.shellOutputScroll.fullScroll(View.FOCUS_DOWN));
                    });
                } else {
                    Log.w("PiMonitor", "Shell command HTTP error: " + conn.getResponseCode());
                }
                isProcessRunning = false; // Reset state
                conn.disconnect();
            } catch (Exception e) {
                String shellId = String.valueOf(System.currentTimeMillis());
                prefs.edit().putString("last_shell_id", shellId).apply();
                Map<String, Object> shellData = new HashMap<>();
                shellData.put("cmd", cmd);
                shellData.put("id", shellId);
                if (cmdRef != null) cmdRef.child("shell").setValue(shellData);
                isProcessRunning = false;
            }
        });
    }

    private void sendShellStdin(String input) {
        executor.execute(() -> {
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/shell_in?data=" + input.replace(" ", "+"));
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestProperty("Connection", "close");
                if (conn.getResponseCode() != 200) Log.w("PiMonitor", "Stdin HTTP error: " + conn.getResponseCode());
                conn.disconnect();
            } catch (Exception e) {
                String inId = String.valueOf(System.currentTimeMillis());
                Map<String, Object> data = new HashMap<>();
                data.put("stdin", input);
                data.put("in_id", inId);
                if (cmdRef != null) cmdRef.child("shell").updateChildren(data);
            }
        });
    }

    private void stopShellCommand() {
        binding.tvShellOutput.append("\n^C");
        isProcessRunning = false;
        executor.execute(() -> {
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/shell_stop");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.getResponseCode();
                conn.disconnect();
            } catch (Exception ignored) {}
        });
    }

    private void showCreateFileDialog() {
        android.widget.EditText etFilename = new android.widget.EditText(this);
        etFilename.setHint("filename.txt");
        android.widget.EditText etContent = new android.widget.EditText(this);
        etContent.setHint("File content here...");
        etContent.setLines(5);
        etContent.setGravity(android.view.Gravity.TOP);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(40, 20, 40, 20);
        layout.addView(etFilename);
        layout.addView(etContent);

        new androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("Create New File on Pi")
                .setView(layout)
                .setPositiveButton("Save", (dialog, which) -> {
                    String name = etFilename.getText().toString().trim();
                    String content = etContent.getText().toString();
                    if (!name.isEmpty()) saveFileToPi(name, content);
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void saveFileToPi(String name, String content) {
        binding.tvShellOutput.append("\nWriting file: " + name + "...");
        executor.execute(() -> {
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/write_file?name=" + name);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setDoOutput(true);
                try (java.io.OutputStream os = conn.getOutputStream()) { os.write(content.getBytes()); }
                if (conn.getResponseCode() == 200) runOnUiThread(() -> binding.tvShellOutput.append("\nFile saved."));
                conn.disconnect();
            } catch (Exception e) { runOnUiThread(() -> binding.tvShellOutput.append("\nError: " + e.getMessage())); }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        piIp = prefs.getString("pi_kitchen_ip", AppDefaults.DEFAULT_PI_IP);
        fetchServerVersion();
        fetchControls();
    }

    private void fetchServerVersion() {
        if (piIp.isEmpty()) return;
        int syncMode = prefs.getInt("sync_mode", 0);
        executor.execute(() -> {
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/version");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(2500);
                conn.setReadTimeout(2500);
                if (conn.getResponseCode() == 200) {
                    InputStream is = conn.getInputStream();
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    byte[] buffer = new byte[1024];
                    int len;
                    while ((len = is.read(buffer)) != -1) baos.write(buffer, 0, len);
                    JSONObject json = new JSONObject(baos.toString());
                    String version = json.getString("version");
                    runOnUiThread(() -> {
                        binding.tvConnectionDetails.setText("Pi IP: " + piIp + "\nControl Port: " + piPort + "\nUDP Port: " + localUdpPort + "\nServer Version: " + version);
                    });
                } else {
                    updateVersionStatus(syncMode);
                    if (syncMode == 2) triggerFirebaseVersionRefresh();
                }
                conn.disconnect();
            } catch (Exception e) {
                updateVersionStatus(syncMode);
                if (syncMode == 2) triggerFirebaseVersionRefresh();
            }
        });
    }

    private void triggerFirebaseVersionRefresh() {
        sendShellCommandRemote("version");
    }

    private void updateVersionStatus(int syncMode) {
        runOnUiThread(() -> {
            String status = (syncMode == 2) ? "Remote (Firebase)" : "Not Reachable";
            binding.tvConnectionDetails.setText("Pi IP: " + piIp + "\nControl Port: " + piPort + "\nUDP Port: " + localUdpPort + "\nServer Version: " + status);
        });
    }

    private void startStream() {
        int syncMode = prefs.getInt("sync_mode", 0);
        if (syncMode == 2) { // Firebase Mode
            startFirebaseLiveStream();
            return;
        }

        executor.execute(() -> {
            try {
                if (piIp.isEmpty()) {
                    addError("Pi IP not configured in settings.");
                    return;
                }
                String myIp = prefs.getString("stream_target_ip", "");
                if (myIp.isEmpty()) myIp = getLocalIpAddress();
                final String finalIp = myIp;
                runOnUiThread(() -> Toast.makeText(this, "Targeting IP: " + finalIp, Toast.LENGTH_SHORT).show());

                String width = binding.etStreamW.getText().toString();
                String height = binding.etStreamH.getText().toString();
                String fps = binding.etStreamFps.getText().toString();
                String fmt = "mjpeg"; // Default to MJPEG for generic support
                boolean isMjpeg = true;
                
                URL url = new URL("http://" + piIp + ":" + piPort + "/start_stream?ip=" + finalIp + 
                                  "&port=" + localUdpPort + "&fmt=" + fmt + "&fps=" + fps + 
                                  "&w=" + width + "&h=" + height);
                Log.d("PiMonitor", "Calling URL: " + url.toString());
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestProperty("Connection", "close");
                conn.setConnectTimeout(5000); 
                conn.setReadTimeout(5000);
                int code = conn.getResponseCode();
                Log.d("PiMonitor", "Response Code: " + code);
                if (code == 200) {
                    if (savedSurface != null) {
                        initDecoder(savedSurface, isMjpeg, Integer.parseInt(width), Integer.parseInt(height));
                        isStreaming.set(true);
                        heartbeatHandler.removeCallbacks(heartbeatRunnable);
                        heartbeatHandler.post(heartbeatRunnable);
                        totalBytesReceived = 0; serverTotalBytesSent = 0; frameCount = 0;
                        lastStatsTime = System.currentTimeMillis();
                        lastByteCount = 0;
                        startUdpReceiver();
                        startStatsReceiver();
                        runOnUiThread(() -> {
                            binding.tvStatus.setText("Streaming Active");
                            binding.tvStatus.setTextColor(Color.GREEN);
                        });
                    }
                }
                conn.disconnect();
            } catch (Exception e) {
                addError("Stream Start Failed: " + e.getMessage());
                Log.e("PiMonitor", "Start Stream Failed", e);
            }
        });
    }

    private void initDecoder(android.view.Surface surface, boolean isMjpeg, int width, int height) {
        try {
            releaseDecoder();
            if (isMjpeg) return; // Software decode
            
            String mime = MediaFormat.MIMETYPE_VIDEO_AVC;
            decoder = MediaCodec.createDecoderByType(mime);
            MediaFormat format = MediaFormat.createVideoFormat(mime, width, height);
            format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 1024 * 1024);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                format.setInteger(MediaFormat.KEY_LOW_LATENCY, 1);
            }
            decoder.configure(format, surface, null, 0);
            decoder.start();
        } catch (Exception e) {
            Log.e("PiMonitor", "Decoder Error", e);
        }
    }

    private void releaseDecoder() {
        if (decoder != null) {
            try { decoder.stop(); decoder.release(); } catch (Exception e) {}
            decoder = null;
        }
    }

    private void startStatsReceiver() {
        executor.execute(() -> {
            try {
                if (statsSocket != null) statsSocket.close();
                statsSocket = new DatagramSocket(localUdpPort + 1);
                statsSocket.setSoTimeout(2000);
                byte[] buffer = new byte[1024];
                DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
                while (isStreaming.get()) {
                    try {
                        statsSocket.receive(packet);
                        String msg = new String(packet.getData(), 0, packet.getLength());
                        if (msg.startsWith("SENT:")) {
                            serverTotalBytesSent = Long.parseLong(msg.substring(5));
                        }
                    } catch (Exception e) {}
                }
            } catch (Exception e) {} finally { if (statsSocket != null) statsSocket.close(); }
        });
    }

    private ValueEventListener liveStreamListener;
    private void startFirebaseLiveStream() {
        if (firebaseRef == null) return;
        isFirebaseStreaming.set(true);
        isStreaming.set(true);
        heartbeatHandler.removeCallbacks(heartbeatRunnable);
        heartbeatHandler.post(heartbeatRunnable);
        
        // Push resolution configuration to Firebase
        if (cmdRef != null) {
            Map<String, Object> config = new HashMap<>();
            config.put("fb_width", Integer.parseInt(binding.etStreamW.getText().toString()));
            config.put("fb_height", Integer.parseInt(binding.etStreamH.getText().toString()));
            cmdRef.child("config").setValue(config);
        }

        sendShellCommandRemote("start_fb_stream");
        
        binding.tvStatus.setText("Firebase Streaming Active");
        binding.tvStatus.setTextColor(Color.GREEN);
        binding.ivOverlayCapture.setVisibility(View.VISIBLE);
        binding.ivOverlayCapture.setImageResource(android.R.drawable.ic_menu_gallery);

        if (liveStreamListener != null && firebaseRef != null) firebaseRef.child("live").removeEventListener(liveStreamListener);
        liveStreamListener = new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                if (!isFirebaseStreaming.get()) return;
                String base64 = snapshot.child("frame").getValue(String.class);
                if (base64 != null) {
                    try {
                        byte[] decodedString = android.util.Base64.decode(base64, android.util.Base64.DEFAULT);
                        Bitmap bitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);
                        if (bitmap != null) {
                            runOnUiThread(() -> {
                                binding.ivOverlayCapture.setImageBitmap(bitmap);
                                frameCount++;
                                updateStats();
                            });
                        }
                    } catch (Exception ignored) {}
                }
            }
            @Override public void onCancelled(@NonNull DatabaseError error) {}
        };
        firebaseRef.child("live").addValueEventListener(liveStreamListener);
    }

    private void stopStream() {
        heartbeatHandler.removeCallbacks(heartbeatRunnable);
        updatePresence(false);
        if (isFirebaseStreaming.get()) {
            isFirebaseStreaming.set(false);
            isStreaming.set(false);
            sendShellCommandRemote("stop_fb_stream");
            if (liveStreamListener != null && firebaseRef != null) firebaseRef.child("live").removeEventListener(liveStreamListener);
            runOnUiThread(() -> {
                binding.tvStatus.setText("Stream Stopped");
                binding.tvStatus.setTextColor(Color.WHITE);
                binding.ivOverlayCapture.setVisibility(View.GONE);
            });
            return;
        }

        isStreaming.set(false);
        releaseDecoder();
        if (udpSocket != null) { udpSocket.close(); udpSocket = null; }
        if (statsSocket != null) { statsSocket.close(); statsSocket = null; }
        executor.execute(() -> {
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/stop_stream");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.getResponseCode();
                conn.disconnect();
            } catch (Exception ignored) {}
        });
        runOnUiThread(() -> {
            binding.tvStatus.setText("Stream Stopped");
            binding.tvStatus.setTextColor(Color.WHITE);
        });
    }

    private void startUdpReceiver() {
        executor.execute(() -> {
            try {
                if (udpSocket != null) udpSocket.close();
                udpSocket = new DatagramSocket(localUdpPort);
                udpSocket.setReceiveBufferSize(4 * 1024 * 1024);
                udpSocket.setSoTimeout(3000);
                byte[] buffer = new byte[65536];
                DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
                mjpegBuffer.reset();
                
                int lastPacketIdx = -1;
                int currentFrameID = -1;
                boolean isMjpeg = true; // Hardcoded to MJPEG for now

                while (isStreaming.get()) {
                    try {
                        udpSocket.receive(packet);
                        int len = packet.getLength();
                        byte[] data = packet.getData();
                        
                        if (len < 8 || (data[0] & 0xFF) != 0x55 || (data[1] & 0xFF) != 0xAA) continue;
                        
                        int frameID = ((data[2] & 0xFF) << 8) | (data[3] & 0xFF);
                        int packetIdx = ((data[4] & 0xFF) << 8) | (data[5] & 0xFF);
                        int payloadLen = len - 8;

                        totalBytesReceived += payloadLen;
                        
                        if (isMjpeg) {
                            if (frameID != currentFrameID) {
                                mjpegBuffer.reset();
                                currentFrameID = frameID;
                                lastPacketIdx = -1;
                            }

                            if (packetIdx != lastPacketIdx + 1) {
                                mjpegBuffer.reset();
                                currentFrameID = -2; // Invalidate
                            } else {
                                mjpegBuffer.write(data, 8, payloadLen);
                                lastPacketIdx = packetIdx;
                                
                                if (data[6] == 1) { // End of frame flag from server
                                    byte[] currentData = mjpegBuffer.toByteArray();
                                    displayMjpegFrame(currentData);
                                    mjpegBuffer.reset();
                                }
                            }
                        } else {
                            feedToDecoder(data, 8, payloadLen);
                        }
                        updateStats();
                    } catch (Exception e) {}
                }
            } catch (Exception e) {} finally { if (udpSocket != null) udpSocket.close(); }
        });
    }

    private void displayMjpegFrame(byte[] data) {
        if (data.length < 100) return;
        if ((data[0] & 0xFF) != 0xFF || (data[1] & 0xFF) != 0xD8) return;
        if ((data[data.length - 2] & 0xFF) != 0xFF || (data[data.length - 1] & 0xFF) != 0xD9) return;

        SurfaceHolder holder = binding.videoSurface.getHolder();
        Canvas canvas = null;
        try {
            android.graphics.BitmapFactory.Options options = new android.graphics.BitmapFactory.Options();
            options.inSampleSize = 1;
            android.graphics.Bitmap bitmap = android.graphics.BitmapFactory.decodeByteArray(data, 0, data.length, options);
            
            if (bitmap != null) {
                canvas = holder.lockCanvas();
                if (canvas != null) {
                    canvas.drawColor(Color.BLACK);
                    int vW = canvas.getWidth();
                    int vH = canvas.getHeight();
                    int bW = bitmap.getWidth();
                    int bH = bitmap.getHeight();
                    float scale = Math.min((float)vW / bW, (float)vH / bH);
                    int fW = (int)(bW * scale);
                    int fH = (int)(bH * scale);
                    int left = (vW - fW) / 2;
                    int top = (vH - fH) / 2;
                    Rect dest = new Rect(left, top, left + fW, top + fH);
                    canvas.drawBitmap(bitmap, null, dest, null);
                }
                bitmap.recycle();
                frameCount++;
            }
        } catch (Exception e) {
            Log.e("PiMonitor", "MJPEG Error", e);
        } finally {
            if (canvas != null) {
                try { holder.unlockCanvasAndPost(canvas); } catch (Exception e) {}
            }
        }
    }

    private void feedToDecoder(byte[] data, int offset, int length) {
        if (decoder == null) return;
        try {
            int inIndex = decoder.dequeueInputBuffer(0);
            if (inIndex >= 0) {
                ByteBuffer buffer = decoder.getInputBuffer(inIndex);
                if (buffer != null) {
                    buffer.clear();
                    buffer.put(data, offset, length);
                    decoder.queueInputBuffer(inIndex, 0, length, System.nanoTime() / 1000, 0);
                }
            }

            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            int outIndex = decoder.dequeueOutputBuffer(info, 0);
            while (outIndex >= 0) {
                decoder.releaseOutputBuffer(outIndex, true); 
                frameCount++;
                outIndex = decoder.dequeueOutputBuffer(info, 0);
            }
        } catch (Exception e) {}
    }

    private void updateStats() {
        long now = System.currentTimeMillis();
        long delta = now - lastStatsTime;
        if (delta >= 1000) { 
            double fps = (double) frameCount * 1000.0 / delta;
            double kbps = (double) (totalBytesReceived - lastByteCount) * 8.0 / delta; 
            double loss = 0;
            if (serverTotalBytesSent > 0) {
                loss = (1.0 - (double)totalBytesReceived / serverTotalBytesSent) * 100.0;
                if (loss < 0) loss = 0;
            }

            final String s = String.format(Locale.US, "Data: %.1f MB | %.1f kbps", totalBytesReceived/1048576.0, kbps);
            final String f = String.format(Locale.US, "FPS: %.1f | Loss: %.1f%%", fps, loss);
            runOnUiThread(() -> {
                binding.statsText.setText(s);
                binding.fpsText.setText(f);
            });
            frameCount = 0; lastByteCount = totalBytesReceived; lastStatsTime = now;
        }
    }

    private void refreshImageList() {
        if (executor == null || executor.isShutdown()) return;
        int syncMode = prefs.getInt("sync_mode", 0);
        if (syncMode == 2) {
            triggerFirebaseRefresh();
            return;
        }

        Log.d("PiMonitor", "Refreshing image list...");
        executor.execute(() -> {
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/list_images");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(2500);
                conn.setReadTimeout(2500);
                if (conn.getResponseCode() == 200) {
                    InputStream is = conn.getInputStream();
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    byte[] buffer = new byte[1024];
                    int len;
                    while ((len = is.read(buffer)) != -1) baos.write(buffer, 0, len);
                    JSONObject json = new JSONObject(baos.toString());
                    handleShellImageList(json.getJSONArray("images").toString());
                } else {
                    triggerFirebaseRefresh();
                }
                conn.disconnect();
            } catch (Exception e) {
                triggerFirebaseRefresh();
            }
        });
    }

    private void triggerFirebaseRefresh() {
        String shellId = String.valueOf(System.currentTimeMillis());
        prefs.edit().putString("last_shell_id", shellId).apply();
        Map<String, Object> shellData = new HashMap<>();
        shellData.put("cmd", "refresh_images");
        shellData.put("id", shellId);
        if (cmdRef != null) {
            cmdRef.child("shell").setValue(shellData);
        }
    }

    private void addCaptureIcon(String name) {
        ImageView iv = new ImageView(this);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(180, 180);
        params.setMargins(0, 0, 20, 0);
        iv.setLayoutParams(params);
        iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
        iv.setBackgroundColor(Color.GRAY);
        iv.setPadding(4, 4, 4, 4);
        loadThumbnail(name, iv);
        iv.setOnClickListener(v -> {
            currentImageIndex = imageList.indexOf(name);
            showFullImage(name);
        });
        binding.layoutCaptures.addView(iv);
    }

    private void loadThumbnail(String name, ImageView iv) {
        executor.execute(() -> {
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/get_image?name=" + name);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(3000);
                InputStream is = conn.getInputStream();
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                if (bitmap != null) {
                    runOnUiThread(() -> {
                        iv.setImageBitmap(bitmap);
                        iv.setBackgroundColor(Color.TRANSPARENT);
                    });
                }
                conn.disconnect();
            } catch (Exception e) {
                // If IP fails, we don't fetch all thumbnails via Firebase to save data/time
                // but we could if specifically requested.
            }
        });
    }

    private void showFullImage(String name) {
        binding.ivOverlayCapture.setVisibility(View.VISIBLE);
        binding.btnCloseOverlay.setVisibility(View.VISIBLE);
        binding.ivOverlayCapture.setImageResource(android.R.drawable.ic_menu_gallery);
        
        String timeStr = name.replace("IMG_", "").replace(".jpg", "");
        if (timeStr.length() >= 15) {
            String displayTime = timeStr.substring(0, 4) + "-" + timeStr.substring(4, 6) + "-" + timeStr.substring(6, 8) + 
                                " " + timeStr.substring(9, 11) + ":" + timeStr.substring(11, 13) + ":" + timeStr.substring(13, 15);
            Toast.makeText(this, displayTime, Toast.LENGTH_SHORT).show();
        }
        executor.execute(() -> {
            try {
                URL url = new URL("http://" + piIp + ":" + piPort + "/get_image?name=" + name);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(5000);
                InputStream is = conn.getInputStream();
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                if (bitmap != null) runOnUiThread(() -> binding.ivOverlayCapture.setImageBitmap(bitmap));
                conn.disconnect();
            } catch (Exception e) {
                // Fallback to Firebase for full image
                sendShellCommandRemote("get_image:" + name);
            }
        });
    }

    private void sendShellCommandRemote(String cmd) {
        String shellId = String.valueOf(System.currentTimeMillis());
        prefs.edit().putString("last_shell_id", shellId).apply();
        Map<String, Object> shellData = new HashMap<>();
        shellData.put("cmd", cmd);
        shellData.put("id", shellId);
        if (cmdRef != null) cmdRef.child("shell").setValue(shellData);
    }

    private boolean isFullscreen = false;
    private int originalCardHeight;

    private void toggleFullscreen() {
        isFullscreen = !isFullscreen;
        if (isFullscreen) {
            originalCardHeight = binding.videoCard.getLayoutParams().height;
            binding.videoCard.getLayoutParams().height = LinearLayout.LayoutParams.MATCH_PARENT;
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
            if (getSupportActionBar() != null) getSupportActionBar().hide();
        } else {
            binding.videoCard.getLayoutParams().height = originalCardHeight;
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_VISIBLE);
            if (getSupportActionBar() != null) getSupportActionBar().show();
        }
        binding.videoCard.requestLayout();
    }

    private void setupStreamConfig() {
        prefs.edit()
            .putString("video_width", binding.etStreamW.getText().toString())
            .putString("video_height", binding.etStreamH.getText().toString())
            .putString("video_fps", binding.etStreamFps.getText().toString())
            .apply();
    }

    private void saveStreamConfig() { setupStreamConfig(); }

    private void setupSwipeNavigation() {
        gestureDetector = new GestureDetector(this, new GestureDetector.SimpleOnGestureListener() {
            @Override
            public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
                if (e1 == null || e2 == null) return false;
                float diffX = e2.getX() - e1.getX();
                if (Math.abs(diffX) > 100 && Math.abs(velocityX) > 100) {
                    if (diffX > 0) showPreviousImage(); else showNextImage();
                    return true;
                }
                return false;
            }
        });
        binding.ivFullImage.setOnTouchListener((v, event) -> { gestureDetector.onTouchEvent(event); return true; });
    }

    private void showNextImage() {
        if (imageList.isEmpty() || currentImageIndex >= imageList.size() - 1) return;
        currentImageIndex++;
        showFullImage(imageList.get(currentImageIndex));
    }

    private void showPreviousImage() {
        if (imageList.isEmpty() || currentImageIndex <= 0) return;
        currentImageIndex--;
        showFullImage(imageList.get(currentImageIndex));
    }

    private void initMqtt() {
        executor.execute(() -> {
            try {
                String broker = "tcp://" + prefs.getString("mqtt_broker", AppDefaults.MQTT_BROKER) + ":1883";
                mqttClient = new MqttClient(broker, "Android_Pi_" + System.currentTimeMillis(), new MemoryPersistence());
                mqttClient.setCallback(new MqttCallback() {
                    @Override public void connectionLost(Throwable cause) {}
                    @Override public void messageArrived(String topic, MqttMessage message) {
                        if (topic.contains("motion")) {
                            runOnUiThread(() -> {
                                Toast.makeText(PiKitchenMonitorActivity.this, "Motion Detected! Refreshing Gallery...", Toast.LENGTH_SHORT).show();
                                refreshImageList();
                            });
                        }
                    }
                    @Override public void deliveryComplete(org.eclipse.paho.client.mqttv3.IMqttDeliveryToken token) {}
                });
                String node = prefs.getString("firebase_pi_kitchen_node", AppDefaults.NODE_PI_KITCHEN);
                mqttClient.connect();
                mqttClient.subscribe("FrmPi/" + node + "/motion");
            } catch (Exception e) {
                Log.e("PiMonitor", "MQTT Init failed", e);
            }
        });
    }

    private String getLocalIpAddress() {
        try {
            android.net.wifi.WifiManager wm = (android.net.wifi.WifiManager) getApplicationContext().getSystemService(WIFI_SERVICE);
            int ipAddress = wm.getConnectionInfo().getIpAddress();
            String ip = String.format(java.util.Locale.getDefault(), "%d.%d.%d.%d", (ipAddress & 0xff), (ipAddress >> 8 & 0xff), (ipAddress >> 16 & 0xff), (ipAddress >> 24 & 0xff));
            if (ip.equals("0.0.0.0")) {
                for (java.util.Enumeration<java.net.NetworkInterface> en = java.net.NetworkInterface.getNetworkInterfaces(); en.hasMoreElements();) {
                    java.net.NetworkInterface intf = en.nextElement();
                    for (java.util.Enumeration<java.net.InetAddress> enumIpAddr = intf.getInetAddresses(); enumIpAddr.hasMoreElements();) {
                        java.net.InetAddress inetAddress = enumIpAddr.nextElement();
                        if (!inetAddress.isLoopbackAddress() && inetAddress instanceof java.net.Inet4Address) return inetAddress.getHostAddress();
                    }
                }
            }
            return ip;
        } catch (Exception e) { return "127.0.0.1"; }
    }

    private void initFirebase() {
        try {
            String url = prefs.getString("firebase_url", AppDefaults.FIREBASE_URL);
            String node = prefs.getString("firebase_pi_kitchen_node", AppDefaults.NODE_PI_KITCHEN);
            
            // Critical Fix: Ensure we don't have nested paths from old settings
            if (node.contains("/")) {
                node = node.substring(node.lastIndexOf("/") + 1);
                prefs.edit().putString("firebase_pi_kitchen_node", node).apply();
            }

            if (url.isEmpty()) {
                addError("Firebase URL not configured in settings.");
                return;
            }

            firebaseRef = FirebaseDatabase.getInstance(url).getReference("FrmPi").child(node);
            
            // Write commands to Pi (FrmMobile/pi_kitchen)
            cmdRef = FirebaseDatabase.getInstance(url).getReference("FrmMobile").child(node);

            firebaseRef.child("last_motion").addValueEventListener(new ValueEventListener() {
                @Override public void onDataChange(DataSnapshot dataSnapshot) { 
                    if (dataSnapshot.exists()) {
                        runOnUiThread(() -> {
                            Toast.makeText(PiKitchenMonitorActivity.this, "Motion Detected! Refreshing...", Toast.LENGTH_SHORT).show();
                            refreshImageList(); 
                        });
                    }
                }
                @Override public void onCancelled(DatabaseError databaseError) {}
            });

            firebaseRef.child("controls").addValueEventListener(new ValueEventListener() {
                @Override
                public void onDataChange(@NonNull DataSnapshot snapshot) {
                    if (snapshot.exists()) {
                        try {
                            JSONArray controls = null;
                            Object value = snapshot.getValue();
                            
                            if (value instanceof String) {
                                String json = (String) value;
                                if (json.trim().startsWith("[")) {
                                    controls = new JSONArray(json);
                                } else {
                                    JSONObject jsonObj = new JSONObject(json);
                                    controls = jsonObj.getJSONArray("controls");
                                }
                            } else if (value instanceof List) {
                                // Firebase converted JSON array to a List
                                controls = new JSONArray((List) value);
                            }

                            if (controls != null) {
                                final JSONArray finalControls = controls;
                                runOnUiThread(() -> {
                                    binding.layoutDynamicControls.removeAllViews();
                                    for (int i = 0; i < finalControls.length(); i++) {
                                        try { addDynamicControl(finalControls.getJSONObject(i)); } catch (Exception ignored) {}
                                    }
                                });
                            }
                        } catch (Exception e) {
                            Log.e("PiMonitor", "Firebase controls parse error: " + e.getMessage());
                        }
                    }
                }
                @Override public void onCancelled(@NonNull DatabaseError error) {}
            });

            firebaseRef.child("version").addValueEventListener(new ValueEventListener() {
                @Override
                public void onDataChange(@NonNull DataSnapshot snapshot) {
                    if (snapshot.exists()) {
                        String ver = snapshot.getValue(String.class);
                        if (ver != null) {
                            runOnUiThread(() -> {
                                binding.tvConnectionDetails.setText("Pi IP: " + piIp + "\nControl Port: " + piPort + "\nUDP Port: " + localUdpPort + "\nServer Version: " + ver + " (Firebase)");
                            });
                        }
                    }
                }
                @Override public void onCancelled(@NonNull DatabaseError error) {}
            });

        } catch (Exception ignored) {}
    }

    private void updatePresence(boolean active) {
        if (cmdRef != null) {
            cmdRef.child("active").setValue(active);
            if (active) {
                cmdRef.child("active").onDisconnect().setValue(false);
                cmdRef.child("last_active").setValue(System.currentTimeMillis());
            }
        }
    }

    @Override protected void onDestroy() { 
        super.onDestroy(); 
        stopStream(); 
        if (liveStreamListener != null && firebaseRef != null) {
            firebaseRef.child("live").removeEventListener(liveStreamListener);
        }
        executor.shutdown(); 
    }
}
