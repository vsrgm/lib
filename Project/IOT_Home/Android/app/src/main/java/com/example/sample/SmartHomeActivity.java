package com.example.sample;

import android.content.Intent;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import com.example.sample.databinding.ActivitySmartHomeBinding;

public class SmartHomeActivity extends AppCompatActivity {

    private ActivitySmartHomeBinding binding;

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
        binding = ActivitySmartHomeBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        binding.btnBack.setOnClickListener(v -> finish());

        binding.cardStudyRoom.setOnClickListener(v -> {
            startActivity(new Intent(this, StudyRoomActivity.class));
        });

        binding.cardMainDoor.setOnClickListener(v -> {
            startActivity(new Intent(this, SecurityMainDoorActivity.class));
        });

        binding.cardKitchen.setOnClickListener(v -> {
            startActivity(new Intent(this, KitchenMonitorActivity.class));
        });

        binding.cardToilet.setOnClickListener(v -> {
            startActivity(new Intent(this, ToiletAssistanceActivity.class));
        });

        binding.cardRoPump.setOnClickListener(v -> {
            startActivity(new Intent(this, RoWasteWaterActivity.class));
        });

        binding.cardKitchenExhaust.setOnClickListener(v -> {
            startActivity(new Intent(this, KitchenExhaustFanActivity.class));
        });

        binding.cardPiKitchen.setOnClickListener(v -> {
            startActivity(new Intent(this, PiKitchenMonitorActivity.class));
        });
    }
}
