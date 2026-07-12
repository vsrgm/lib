package com.example.sample;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.sample.databinding.ActivityRcCarSettingsBinding;

public class RcCarSettingsActivity extends AppCompatActivity {

    private ActivityRcCarSettingsBinding binding;
    private SharedPreferences prefs;

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
        binding = ActivityRcCarSettingsBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        prefs = getSharedPreferences("RcCarPrefs", MODE_PRIVATE);

        loadSettings();

        binding.storagePath.addTextChangedListener(new android.text.TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) {
                binding.tvPathInfo.setText("Path: [Downloads]/" + s.toString());
            }
            @Override public void afterTextChanged(android.text.Editable s) {}
        });

        binding.btnBack.setOnClickListener(v -> finish());
        binding.btnSave.setOnClickListener(v -> {
            saveSettings();
            Toast.makeText(this, "Settings Saved", Toast.LENGTH_SHORT).show();
            finish();
        });
    }

    private void loadSettings() {
        binding.storagePath.setText(prefs.getString("storage_path", "IOT_HOME/RCCAR"));
        binding.tvPathInfo.setText("Path: [Downloads]/" + binding.storagePath.getText().toString());
        
        int mode = prefs.getInt("car_sync_mode", 0); // 0: IP, 1: Firebase
        if (mode == 0) {
            binding.rbCarModeIp.setChecked(true);
        } else {
            binding.rbCarModeFirebase.setChecked(true);
        }
    }

    private void saveSettings() {
        String path = binding.storagePath.getText().toString().trim();
        if (path.isEmpty()) path = "IOT_HOME/RCCAR";

        int mode = binding.rbCarModeIp.isChecked() ? 0 : 1;

        prefs.edit()
            .putString("storage_path", path)
            .putInt("car_sync_mode", mode)
            .apply();
    }
}
