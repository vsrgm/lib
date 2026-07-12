package com.example.sample;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class ImageViewerActivity extends AppCompatActivity {

    private GridView gridView;
    private TextView emptyText;
    private List<File> imageFiles = new ArrayList<>();

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
        setContentView(R.layout.activity_image_viewer);

        findViewById(R.id.btn_back).setOnClickListener(v -> finish());
        gridView = findViewById(R.id.image_grid);
        emptyText = findViewById(R.id.empty_text);

        loadImages();
    }

    private void loadImages() {
        imageFiles.clear();
        
        File downloadsDir = StorageUtils.getDownloadsDir();

        // Load dynamic paths from preferences
        SharedPreferences rcPrefs = getSharedPreferences("RcCarPrefs", MODE_PRIVATE);
        String rcPath = rcPrefs.getString("storage_path", getString(R.string.default_rc_path));
        addImagesFromDir(new File(downloadsDir, rcPath));

        SharedPreferences shPrefs = getSharedPreferences("SmartHomePrefs", MODE_PRIVATE);
        String kitchenPath = shPrefs.getString("kitchen_path", getString(R.string.default_kitchen_path));
        addImagesFromDir(new File(downloadsDir, kitchenPath));

        String doorPath = shPrefs.getString("door_path", getString(R.string.default_door_path));
        addImagesFromDir(new File(downloadsDir, doorPath));

        // Load from legacy internal storage if it exists
        addImagesFromDir(new File(getFilesDir(), "IOT_HOME/RCCAR"));

        // Load from legacy Pictures/Sample if it exists
        try {
            Class<?> envClass = Class.forName("android.os.Environment");
            File picDir = (File) envClass.getMethod("getExternalStoragePublicDirectory", String.class)
                    .invoke(null, envClass.getField("DIRECTORY_PICTURES").get(null));
            addImagesFromDir(new File(picDir, "Sample"));
        } catch (Exception ignored) {}

        Collections.sort(imageFiles, (f1, f2) -> Long.compare(f2.lastModified(), f1.lastModified()));

        if (imageFiles.isEmpty()) {
            emptyText.setVisibility(View.VISIBLE);
            gridView.setVisibility(View.GONE);
        } else {
            emptyText.setVisibility(View.GONE);
            gridView.setVisibility(View.VISIBLE);
            gridView.setAdapter(new ImageAdapter(this, imageFiles));
        }
    }

    private void addImagesFromDir(File directory) {
        if (directory.exists() && directory.isDirectory()) {
            File[] files = directory.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".jpeg"));
            if (files != null) {
                imageFiles.addAll(Arrays.asList(files));
            }
        }
    }

    private static class ImageAdapter extends BaseAdapter {
        private Context context;
        private List<File> files;

        public ImageAdapter(Context context, List<File> files) {
            this.context = context;
            this.files = files;
        }

        @Override
        public int getCount() { return files.size(); }

        @Override
        public Object getItem(int position) { return files.get(position); }

        @Override
        public long getItemId(int position) { return position; }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            ImageView imageView;
            if (convertView == null) {
                imageView = new ImageView(context);
                imageView.setLayoutParams(new GridView.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 400));
                imageView.setScaleType(ImageView.ScaleType.CENTER_CROP);
                imageView.setPadding(4, 4, 4, 4);
            } else {
                imageView = (ImageView) convertView;
            }

            // Simple loading - in a real app, use Glide or Picasso
            Bitmap bitmap = BitmapFactory.decodeFile(files.get(position).getAbsolutePath());
            imageView.setImageBitmap(bitmap);

            return imageView;
        }
    }
}
