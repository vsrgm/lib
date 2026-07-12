package com.example.sample;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import com.example.sample.BuildConfig;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatDelegate;
import com.example.sample.databinding.ActivityMenuBinding;

public class MenuActivity extends AppCompatActivity {

    private ActivityMenuBinding binding;
    private SharedPreferences themePrefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Apply saved theme before super.onCreate
        themePrefs = getSharedPreferences("ThemePrefs", MODE_PRIVATE);
        applySavedTheme();

        super.onCreate(savedInstanceState);
        binding = ActivityMenuBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.cardRcCar.setOnClickListener(v -> {
            Intent intent = new Intent(MenuActivity.this, MainActivity.class);
            startActivity(intent);
        });

        binding.cardSmartHome.setOnClickListener(v -> {
            Intent intent = new Intent(MenuActivity.this, SmartHomeActivity.class);
            startActivity(intent);
        });

        binding.btnSettings.setOnClickListener(v -> showThemeDialog());

        binding.tvAppVersion.setText("Version " + BuildConfig.VERSION_NAME);
    }

    private void applySavedTheme() {
        int themeMode = themePrefs.getInt("theme_mode", 0);
        switch (themeMode) {
            case 1:
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO);
                break;
            case 2:
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES);
                break;
            default:
                AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM);
                break;
        }
    }

    private void showThemeDialog() {
        String[] options = {
                getString(R.string.theme_system),
                getString(R.string.theme_light),
                getString(R.string.theme_dark)
        };
        int currentSelection = themePrefs.getInt("theme_mode", 0);

        new AlertDialog.Builder(this)
                .setTitle(R.string.theme_title)
                .setSingleChoiceItems(options, currentSelection, (dialog, which) -> {
                    themePrefs.edit().putInt("theme_mode", which).apply();
                    applySavedTheme();
                    dialog.dismiss();
                    // Restart activity to apply theme change immediately
                    recreate();
                })
                .setNegativeButton(android.R.string.cancel, null)
                .show();
    }
}
