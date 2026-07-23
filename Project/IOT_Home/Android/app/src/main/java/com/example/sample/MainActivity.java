package com.example.sample;

import android.Manifest;
import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Rect;
import android.media.MediaCodec;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.View;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.example.sample.databinding.ActivityMainBinding;

import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.HttpURLConnection;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.URL;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Enumeration;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private ActivityMainBinding binding;
    private final ExecutorService networkExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService videoExecutor = Executors.newFixedThreadPool(2);
    
    private MediaCodec decoder;
    private final AtomicBoolean isStreaming = new AtomicBoolean(false);
    private DatagramSocket udpSocket;
    private DatagramSocket statsSocket;
    
    private long totalBytesReceived = 0;
    private volatile long serverTotalBytesSent = 0;
    private long frameCount = 0;
    private long lastStatsTime = 0;
    private long lastByteCount = 0;
    private long lastFrameSaveTime = 0;
    private String detectedIp = "0.0.0.0";
    private final int videoPort = 5000;
    private final ByteArrayOutputStream mjpegBuffer = new ByteArrayOutputStream();
    private android.view.Surface savedSurface;
    private DatabaseReference firebaseRef;
    private int carSyncMode = 0; // 0: IP, 1: Firebase

    private final ActivityResultLauncher<String[]> permissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), result -> {});

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
        try {
            binding = ActivityMainBinding.inflate(getLayoutInflater());
            setContentView(binding.getRoot());
            checkAndRequestPermissions();

            binding.rcConnectionStatus.setText(R.string.status_rc_not_connected);
            binding.cameraConnectionStatus.setText(R.string.status_camera_not_connected);

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

            detectedIp = getLocalIpAddress();
            if (detectedIp == null) detectedIp = "0.0.0.0";
            binding.receiverInfoText.setText(String.format(Locale.US, "Receiver: %s:%d", detectedIp, videoPort));

            setupCarControls();
            binding.btnBack.setOnClickListener(v -> finish());
            binding.btnSettings.setOnClickListener(v -> {
                android.content.Intent intent = new android.content.Intent(this, RcCarSettingsActivity.class);
                startActivity(intent);
            });
            binding.btnGallery.setOnClickListener(v -> {
                android.content.Intent intent = new android.content.Intent(this, ImageViewerActivity.class);
                startActivity(intent);
            });

            initFirebase();

            binding.btnConnect.setOnClickListener(v -> {
                if (savedSurface != null) {
                    boolean isMjpeg = binding.radioMjpeg.isChecked();
                    int w = 640;
                    int h = 480;
                    try {
                        w = Integer.parseInt(binding.videoWidthInput.getText().toString().trim());
                        h = Integer.parseInt(binding.videoHeightInput.getText().toString().trim());
                    } catch (Exception e) {}
                    initDecoder(savedSurface, isMjpeg, w, h);
                    startStreamingLoop();
                    triggerPiStreaming();
                    checkCarConnection();
                }
            });
        } catch (Exception e) {}
    }

    @Override
    protected void onResume() {
        super.onResume();
        android.content.SharedPreferences prefs = getSharedPreferences("RcCarPrefs", MODE_PRIVATE);
        carSyncMode = prefs.getInt("car_sync_mode", 0);
    }

    private void initFirebase() {
        android.content.SharedPreferences smartPrefs = getSharedPreferences("SmartHomePrefs", MODE_PRIVATE);
        String email = smartPrefs.getString("firebase_email", "");
        String password = smartPrefs.getString("firebase_password", "");
        String url = smartPrefs.getString("firebase_url", "");

        if (email.isEmpty() || password.isEmpty()) {
            if (!url.isEmpty()) {
                firebaseRef = FirebaseDatabase.getInstance(url).getReference("smart_home/rccar");
            }
            return;
        }

        FirebaseAuth.getInstance().signInWithEmailAndPassword(email, password)
            .addOnCompleteListener(task -> {
                if (task.isSuccessful()) {
                    if (!url.isEmpty()) {
                        firebaseRef = FirebaseDatabase.getInstance(url).getReference("smart_home/rccar");
                    }
                }
            });
    }

    private void initDecoder(android.view.Surface surface, boolean isMjpeg, int width, int height) {
        try {
            if (decoder != null) {
                try { decoder.stop(); decoder.release(); } catch (Exception e) {}
                decoder = null;
            }
            if (isMjpeg) {
                // MJPEG will be decoded in software via displayMjpegFrame
                return;
            }
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
            Log.e(TAG, "Decoder Error", e);
        }
    }

    private void startStreamingLoop() {
        if (isStreaming.get()) return;
        isStreaming.set(true);
        totalBytesReceived = 0; serverTotalBytesSent = 0; frameCount = 0;
        lastStatsTime = System.currentTimeMillis();
        lastByteCount = 0;

        String extension = binding.radioMjpeg.isChecked() ? ".mjpeg" : ".h264";

        // Stats Receiver Loop
        videoExecutor.execute(() -> {
            try {
                if (statsSocket != null) statsSocket.close();
                statsSocket = new DatagramSocket(videoPort + 1);
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
            } catch (Exception e) {
                Log.e(TAG, "Stats Socket Error", e);
            } finally {
                if (statsSocket != null) statsSocket.close();
            }
        });

        videoExecutor.execute(() -> {
            FileOutputStream fos = null;
            try {
                // Prepare Data Dump File in Downloads (Cross-platform)
                try {
                    android.content.SharedPreferences prefs = getSharedPreferences("RcCarPrefs", MODE_PRIVATE);
                    String subDir = prefs.getString("storage_path", getString(R.string.default_rc_path));

                    File dir = StorageUtils.getWorkDir(subDir);
                    File dumpFile = new File(dir, "stream_data" + extension);
                    fos = new FileOutputStream(dumpFile);
                } catch (Exception e) {}

                if (udpSocket != null) udpSocket.close();
                udpSocket = new DatagramSocket(videoPort);
                udpSocket.setSoTimeout(3000);
                udpSocket.setReceiveBufferSize(4 * 1024 * 1024);
                
                byte[] buffer = new byte[65536];
                DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
                mjpegBuffer.reset();

                int lastPacketIdx = -1;
                int currentFrameID = -1;

                while (isStreaming.get()) {
                    try {
                        udpSocket.receive(packet);
                        int len = packet.getLength();
                        byte[] data = packet.getData();
                        
                        // Header check: 0x55 0xAA
                        if (len < 6 || (data[0] & 0xFF) != 0x55 || (data[1] & 0xFF) != 0xAA) continue;
                        
                        int frameID = ((data[2] & 0xFF) << 8) | (data[3] & 0xFF);
                        int packetIdx = ((data[4] & 0xFF) << 8) | (data[5] & 0xFF);
                        int payloadLen = len - 6;

                        totalBytesReceived += payloadLen;
                        if (fos != null) fos.write(data, 6, payloadLen);
                        
                        if (binding.radioMjpeg.isChecked()) {
                            if (frameID != currentFrameID) {
                                mjpegBuffer.reset();
                                currentFrameID = frameID;
                                lastPacketIdx = -1;
                            }

                            if (packetIdx != lastPacketIdx + 1) {
                                // Packet loss! Discard current buffer
                                mjpegBuffer.reset();
                                currentFrameID = -2; // Invalidate current frame
                            } else {
                                mjpegBuffer.write(data, 6, payloadLen);
                                lastPacketIdx = packetIdx;
                                
                                byte[] currentData = mjpegBuffer.toByteArray();
                                if (currentData.length >= 2 && 
                                    (currentData[currentData.length - 2] & 0xFF) == 0xFF && 
                                    (currentData[currentData.length - 1] & 0xFF) == 0xD9) {
                                    
                                    displayMjpegFrame(currentData);
                                    long now = System.currentTimeMillis();
                                    if (now - lastFrameSaveTime >= 1000) {
                                        saveMjpegFrame(currentData);
                                        lastFrameSaveTime = now;
                                    }
                                    mjpegBuffer.reset();
                                }
                            }
                        } else {
                            feedToDecoder(data, 6, payloadLen);
                        }
                        updateStats();
                        
                    } catch (java.net.SocketTimeoutException e) {
                        runOnUiThread(() -> {
                            binding.cameraConnectionStatus.setText(R.string.status_camera_connecting);
                            binding.cameraConnectionStatus.setTextColor(Color.YELLOW);
                        });
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "UDP Error", e);
                runOnUiThread(() -> {
                    binding.cameraConnectionStatus.setText(R.string.status_camera_error);
                    binding.cameraConnectionStatus.setTextColor(Color.RED);
                });
            } finally {
                try { if (fos != null) fos.close(); } catch (Exception e) {}
                if (udpSocket != null) udpSocket.close();
            }
        });
    }

    private void saveMjpegFrame(byte[] data) {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String fileName = "MJPEG_" + timeStamp + ".jpg";
        
        try {
            android.content.SharedPreferences prefs = getSharedPreferences("RcCarPrefs", MODE_PRIVATE);
            String subDir = prefs.getString("storage_path", getString(R.string.default_rc_path));
            
            File dir = StorageUtils.getWorkDir(subDir);
            File file = new File(dir, fileName);
            try (FileOutputStream out = new FileOutputStream(file)) {
                out.write(data);
                out.flush();
            }
            Log.d(TAG, "Saved frame to: " + file.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, "Failed to save frame", e);
        }
    }

    private void displayMjpegFrame(byte[] data) {
        // Integrity check: Must start with SOI (FF D8) and end with EOI (FF D9)
        if (data.length < 100) return; // Minimum size for a sane JPEG
        if ((data[0] & 0xFF) != 0xFF || (data[1] & 0xFF) != 0xD8) return;
        if ((data[data.length - 2] & 0xFF) != 0xFF || (data[data.length - 1] & 0xFF) != 0xD9) return;

        SurfaceHolder holder = binding.videoSurface.getHolder();
        Canvas canvas = null;
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inSampleSize = 1; // Can be 2 to reduce CPU if needed
            Bitmap bitmap = BitmapFactory.decodeByteArray(data, 0, data.length, options);
            
            if (bitmap != null) {
                canvas = holder.lockCanvas();
                if (canvas != null) {
                    canvas.drawColor(Color.BLACK);
                    
                    int viewWidth = canvas.getWidth();
                    int viewHeight = canvas.getHeight();
                    int bmpWidth = bitmap.getWidth();
                    int bmpHeight = bitmap.getHeight();
                    
                    float scale = Math.min((float)viewWidth / bmpWidth, (float)viewHeight / bmpHeight);
                    int finalWidth = (int)(bmpWidth * scale);
                    int finalHeight = (int)(bmpHeight * scale);
                    int left = (viewWidth - finalWidth) / 2;
                    int top = (viewHeight - finalHeight) / 2;
                    
                    Rect dest = new Rect(left, top, left + finalWidth, top + finalHeight);
                    canvas.drawBitmap(bitmap, null, dest, null);
                }
                bitmap.recycle();
                frameCount++;
            }
        } catch (Exception e) {
            Log.e(TAG, "MJPEG Display Error", e);
        } finally {
            if (canvas != null) {
                try {
                    holder.unlockCanvasAndPost(canvas);
                } catch (Exception e) {}
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
                binding.cameraConnectionStatus.setText(frameCount > 0 ? "Camera: LIVE" : "Camera: Receiving...");
                binding.cameraConnectionStatus.setTextColor(frameCount > 0 ? Color.GREEN : Color.YELLOW);
                binding.statsText.setText(s);
                binding.fpsText.setText(f);
            });
            frameCount = 0; lastByteCount = totalBytesReceived; lastStatsTime = now;
        }
    }

    private void checkAndRequestPermissions() {
        List<String> permissions = new ArrayList<>();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.NEARBY_WIFI_DEVICES);
        } else {
            permissions.add(Manifest.permission.ACCESS_FINE_LOCATION);
        }
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            permissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }
        permissionLauncher.launch(permissions.toArray(new String[0]));
    }

    private String getLocalIpAddress() {
        try {
            for (Enumeration<NetworkInterface> en = NetworkInterface.getNetworkInterfaces(); en.hasMoreElements();) {
                NetworkInterface intf = en.nextElement();
                for (Enumeration<InetAddress> ips = intf.getInetAddresses(); ips.hasMoreElements();) {
                    InetAddress ip = ips.nextElement();
                    if (!ip.isLoopbackAddress() && ip instanceof Inet4Address) return ip.getHostAddress();
                }
            }
        } catch (Exception ex) {}
        return null;
    }

    private void triggerPiStreaming() {
        String piIp = binding.piIpInput.getText().toString().trim();
        String fps = binding.videoFpsInput.getText().toString().trim();
        String width = binding.videoWidthInput.getText().toString().trim();
        String height = binding.videoHeightInput.getText().toString().trim();
        String format = binding.radioMjpeg.isChecked() ? "mjpeg" : "h264";
        
        if (piIp.isEmpty() || detectedIp.equals("0.0.0.0")) return;

        networkExecutor.execute(() -> {
            try {
                URL url = new URL("http://" + piIp + ":5001/start_stream?ip=" + detectedIp + "&port=" + videoPort + "&fmt=" + format + "&fps=" + fps + "&w=" + width + "&h=" + height);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(5000);
                int responseCode = conn.getResponseCode();
                Log.d(TAG, "Pi Stream Trigger Response: " + responseCode);
                conn.disconnect();
            } catch (Exception e) {
                Log.e(TAG, "Failed to trigger Pi streaming", e);
            }
        });
    }

    private void checkCarConnection() {
        String ip = binding.ipAddressInput.getText().toString().trim();
        if (ip.isEmpty()) return;

        networkExecutor.execute(() -> {
            try {
                URL url = new URL("http://" + ip + "/");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(2000);
                boolean success = (conn.getResponseCode() == 200);
                conn.disconnect();
                runOnUiThread(() -> {
                    binding.rcConnectionStatus.setText(success ? R.string.status_rc_connected : R.string.status_rc_failed);
                    binding.rcConnectionStatus.setTextColor(success ? Color.GREEN : Color.RED);
                });
            } catch (Exception e) {
                Log.e(TAG, "Car Connection Check Failed", e);
                runOnUiThread(() -> {
                    binding.rcConnectionStatus.setText(R.string.status_rc_failed);
                    binding.rcConnectionStatus.setTextColor(Color.RED);
                });
            }
        });
    }

    private void setupCarControls() {
        // Reversed controls because Pi is at the back
        binding.btnUp.setOnTouchListener((v, event) -> handleCarTouch("B", event));
        binding.btnDown.setOnTouchListener((v, event) -> handleCarTouch("F", event));
        binding.btnLeft.setOnTouchListener((v, event) -> handleCarTouch("L", event));
        binding.btnRight.setOnTouchListener((v, event) -> handleCarTouch("R", event));
    }

    private boolean handleCarTouch(String cmd, MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_DOWN) { sendCommand(cmd); return true; }
        if (event.getAction() == MotionEvent.ACTION_UP || event.getAction() == MotionEvent.ACTION_CANCEL) { sendCommand("S"); return true; }
        return false;
    }

    private void sendCommand(String state) {
        if (carSyncMode == 1) {
            if (firebaseRef != null) {
                firebaseRef.child("command").setValue(state);
            }
            return;
        }
        String ip = binding.ipAddressInput.getText().toString().trim();
        networkExecutor.execute(() -> {
            try {
                URL url = new URL("http://" + ip + "/?State=" + state);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(800);
                conn.getResponseCode();
                conn.disconnect();
            } catch (Exception e) {}
        });
    }

    private void releaseDecoder() {
        isStreaming.set(false);
        if (decoder != null) { try { decoder.stop(); decoder.release(); } catch (Exception e) {} decoder = null; }
    }

    @Override protected void onDestroy() { super.onDestroy(); releaseDecoder(); networkExecutor.shutdown(); videoExecutor.shutdown(); }
}