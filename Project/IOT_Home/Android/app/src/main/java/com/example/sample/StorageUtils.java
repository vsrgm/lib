package com.example.sample;

import java.io.File;

public class StorageUtils {
    public static File getDownloadsDir() {
        try {
            // Check if we are on Android
            Class.forName("android.os.Environment");
            return android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOWNLOADS);
        } catch (Exception e) {
            // Desktop fallback (Windows/Linux)
            return new File(System.getProperty("user.home"), "Downloads");
        }
    }

    public static File getWorkDir(String subPath) {
        File dir = new File(getDownloadsDir(), subPath);
        if (!dir.exists()) dir.mkdirs();
        return dir;
    }
}
