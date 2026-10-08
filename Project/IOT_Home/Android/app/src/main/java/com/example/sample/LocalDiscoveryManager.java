package com.example.sample;

import android.content.Context;
import android.content.SharedPreferences;
import android.net.DhcpInfo;
import android.net.wifi.WifiManager;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class LocalDiscoveryManager {

    private static final String TAG = "LocalDiscoveryManager";
    public static final int UDP_DISCOVERY_PORT = 8888;
    private static final ExecutorService executor = Executors.newSingleThreadExecutor();

    @FunctionalInterface
    public interface DiscoveryCallback {
        void onDeviceDiscovered(String key, String deviceName, String ipAddress);
        default void onDiscoveryComplete(Map<String, String> discoveredMap) {}
    }

    public static void discoverDevices(Context context, DiscoveryCallback callback) {
        executor.execute(() -> {
            Map<String, String> discoveredMap = new HashMap<>();
            DatagramSocket socket = null;
            WifiManager.MulticastLock multicastLock = null;

            WifiManager wifi = (WifiManager) context.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
            if (wifi != null) {
                try {
                    multicastLock = wifi.createMulticastLock("LocalDiscoveryMulticastLock");
                    multicastLock.setReferenceCounted(true);
                    multicastLock.acquire();
                } catch (Exception e) {
                    Log.w(TAG, "Failed to acquire multicast lock: " + e.getMessage());
                }
            }

            try {
                socket = new DatagramSocket();
                socket.setBroadcast(true);
                socket.setSoTimeout(1500);

                InetAddress broadcastAddr = getBroadcastAddress(context);
                String msg = "DISCOVER";
                byte[] data = msg.getBytes();
                DatagramPacket sendPacket = new DatagramPacket(data, data.length, broadcastAddr, UDP_DISCOVERY_PORT);
                socket.send(sendPacket);
                Log.d(TAG, "Broadcast DISCOVER sent to " + broadcastAddr.getHostAddress() + ":" + UDP_DISCOVERY_PORT);

                // Also send fallback to 255.255.255.255
                try {
                    InetAddress globalBroadcast = InetAddress.getByName("255.255.255.255");
                    socket.send(new DatagramPacket(data, data.length, globalBroadcast, UDP_DISCOVERY_PORT));
                } catch (Exception ignored) {}

                // Parallel HTTP Subnet Probe Scan Fallback
                String subnetPrefix = getSubnetPrefix(context);
                if (subnetPrefix != null && !subnetPrefix.isEmpty()) {
                    runHttpSubnetProbe(context, subnetPrefix, discoveredMap, callback);
                }

                byte[] recvBuf = new byte[1024];
                long startTime = System.currentTimeMillis();
                SharedPreferences prefs = context.getSharedPreferences("SmartHomePrefs", Context.MODE_PRIVATE);

                while (System.currentTimeMillis() - startTime < 3000) {
                    try {
                        DatagramPacket recvPacket = new DatagramPacket(recvBuf, recvBuf.length);
                        socket.receive(recvPacket);
                        String senderIp = recvPacket.getAddress().getHostAddress();
                        String response = new String(recvPacket.getData(), 0, recvPacket.getLength()).trim();
                        Log.d(TAG, "Received UDP response from " + senderIp + ": " + response);

                        String deviceName = "";
                        String deviceIp = senderIp;

                        if (response.startsWith("{") && response.endsWith("}")) {
                            JSONObject json = new JSONObject(response);
                            if (json.has("name")) deviceName = json.getString("name");
                            if (json.has("id") && deviceName.isEmpty()) deviceName = json.getString("id");
                            if (json.has("ip")) deviceIp = json.getString("ip");
                        } else {
                            deviceName = response;
                        }

                        String prefKey = mapDeviceToKey(deviceName);
                        if (prefKey != null) {
                            prefs.edit().putString(prefKey, deviceIp).commit();
                            synchronized (discoveredMap) {
                                discoveredMap.put(prefKey, deviceIp);
                            }
                            Log.d(TAG, "Mapped device '" + deviceName + "' (" + deviceIp + ") to key: " + prefKey);

                            final String fKey = prefKey;
                            final String fName = deviceName;
                            final String fIp = deviceIp;
                            if (callback != null) {
                                new Handler(Looper.getMainLooper()).post(() ->
                                        callback.onDeviceDiscovered(fKey, fName, fIp)
                                );
                            }
                        }
                    } catch (Exception e) {
                        if (socket.isClosed()) break;
                    }
                }

            } catch (Exception e) {
                Log.e(TAG, "Discovery error", e);
            } finally {
                if (socket != null && !socket.isClosed()) {
                    socket.close();
                }
                if (multicastLock != null && multicastLock.isHeld()) {
                    try {
                        multicastLock.release();
                    } catch (Exception ignored) {}
                }
            }

            if (callback != null) {
                new Handler(Looper.getMainLooper()).post(() ->
                        callback.onDiscoveryComplete(discoveredMap)
                );
            }
        });
    }

    private static void runHttpSubnetProbe(Context context, String subnetPrefix, Map<String, String> discoveredMap, DiscoveryCallback callback) {
        ExecutorService probePool = Executors.newFixedThreadPool(15);
        SharedPreferences prefs = context.getSharedPreferences("SmartHomePrefs", Context.MODE_PRIVATE);

        for (int host = 1; host <= 254; host++) {
            final String targetIp = subnetPrefix + host;
            probePool.execute(() -> {
                try {
                    URL url = new URL("http://" + targetIp + "/status");
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setConnectTimeout(800);
                    conn.setReadTimeout(800);
                    if (conn.getResponseCode() == 200) {
                        BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                        StringBuilder sb = new StringBuilder();
                        String line;
                        while ((line = reader.readLine()) != null) sb.append(line);
                        String resp = sb.toString().trim();

                        String deviceName = "";
                        String deviceIp = targetIp;

                        if (resp.startsWith("{") && resp.endsWith("}")) {
                            JSONObject json = new JSONObject(resp);
                            if (json.has("name")) deviceName = json.getString("name");
                            if (json.has("id") && deviceName.isEmpty()) deviceName = json.getString("id");
                            if (json.has("ip")) deviceIp = json.getString("ip");
                        }

                        String prefKey = mapDeviceToKey(deviceName);
                        if (prefKey != null) {
                            prefs.edit().putString(prefKey, deviceIp).commit();
                            synchronized (discoveredMap) {
                                discoveredMap.put(prefKey, deviceIp);
                            }
                            Log.d(TAG, "HTTP Probe Mapped device '" + deviceName + "' (" + deviceIp + ") -> " + prefKey);

                            final String fKey = prefKey;
                            final String fName = deviceName;
                            final String fIp = deviceIp;
                            if (callback != null) {
                                new Handler(Looper.getMainLooper()).post(() ->
                                        callback.onDeviceDiscovered(fKey, fName, fIp)
                                );
                            }
                        }
                    }
                    conn.disconnect();
                } catch (Exception ignored) {}
            });
        }
        probePool.shutdown();
    }

    public static String mapDeviceToKey(String deviceIdentifier) {
        if (deviceIdentifier == null) return null;
        String idLower = deviceIdentifier.toLowerCase();

        if (idLower.contains("kitchen_fan") || idLower.contains("exhaust") || idLower.contains("kitchenfan")) {
            return "ip_kitchen_fan";
        } else if (idLower.contains("pi_kitchen") || idLower.contains("pikitchen")) {
            return "pi_kitchen_ip";
        } else if (idLower.contains("kitchen") || idLower.contains("esp32_kitchen")) {
            return "ip_kitchen";
        } else if (idLower.contains("study")) {
            return "ip_study";
        } else if (idLower.contains("door") || idLower.contains("main_door") || idLower.contains("esp32_cam") || idLower.contains("cam")) {
            return "ip_door";
        } else if (idLower.contains("toilet")) {
            return "ip_toilet";
        } else if (idLower.contains("ro_pump") || idLower.contains("ro") || idLower.contains("pump") || idLower.contains("water") || idLower.contains("waste")) {
            return "ip_ro_pump";
        } else if (idLower.contains("bedroom")) {
            return "ip_bedroom";
        } else if (idLower.contains("rccar") || idLower.contains("rc_car") || idLower.contains("car")) {
            return "ip_rccar";
        }
        return null;
    }

    public static String getIpKeyForContext(String contextKey) {
        if (contextKey == null) return "local_node_ip";
        switch (contextKey) {
            case "study": return "ip_study";
            case "kitchen": return "ip_kitchen";
            case "door": return "ip_door";
            case "toilet": return "ip_toilet";
            case "ro_pump": return "ip_ro_pump";
            case "kitchen_fan": return "ip_kitchen_fan";
            case "pi_kitchen": return "pi_kitchen_ip";
            case "bedroom": return "ip_bedroom";
            case "rccar":
            case "rc_car":
            case "car": return "ip_rccar";
            default: return "local_node_ip";
        }
    }

    public static String getDeviceIp(SharedPreferences prefs, String key) {
        String specificKey = getIpKeyForContext(key);
        String fallback = prefs.getString("local_node_ip", AppDefaults.DEFAULT_NODE_IP);
        if ("pi_kitchen".equals(key)) {
            return prefs.getString("pi_kitchen_ip", AppDefaults.DEFAULT_PI_IP);
        }
        if ("rccar".equals(key) || "rc_car".equals(key) || "car".equals(key)) {
            return prefs.getString("ip_rccar", AppDefaults.DEFAULT_RCCAR_IP);
        }
        return prefs.getString(specificKey, fallback);
    }

    private static String getSubnetPrefix(Context context) {
        try {
            WifiManager wifi = (WifiManager) context.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
            if (wifi != null) {
                DhcpInfo dhcp = wifi.getDhcpInfo();
                if (dhcp != null && dhcp.ipAddress != 0) {
                    int ip = dhcp.ipAddress;
                    String ipStr = String.format(java.util.Locale.US, "%d.%d.%d.%d",
                            (ip & 0xFF), (ip >> 8 & 0xFF), (ip >> 16 & 0xFF), (ip >> 24 & 0xFF));
                    int lastDot = ipStr.lastIndexOf('.');
                    if (lastDot != -1) {
                        return ipStr.substring(0, lastDot + 1);
                    }
                }
            }
        } catch (Exception ignored) {}
        return "192.168.0.";
    }

    private static InetAddress getBroadcastAddress(Context context) throws Exception {
        WifiManager wifi = (WifiManager) context.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        if (wifi != null) {
            DhcpInfo dhcp = wifi.getDhcpInfo();
            if (dhcp != null && dhcp.gateway != 0) {
                int broadcast = (dhcp.ipAddress & dhcp.netmask) | ~dhcp.netmask;
                byte[] quads = new byte[4];
                for (int k = 0; k < 4; k++) {
                    quads[k] = (byte) ((broadcast >> (k * 8)) & 0xFF);
                }
                return InetAddress.getByAddress(quads);
            }
        }
        return InetAddress.getByName("255.255.255.255");
    }
}
