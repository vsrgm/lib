package com.example.sample;

public class AppDefaults {
    public static final String FIREBASE_URL = "https://gapsmarthome-default-rtdb.asia-southeast1.firebasedatabase.app";
    
    public static final String DEFAULT_PI_IP = "192.168.0.113";
    public static final String DEFAULT_NODE_IP = "192.168.0.107";
    
    public static final boolean ENABLE_MQTT = false; // Set to true to enable MQTT globally
    public static final String MQTT_BROKER = "broker.hivemq.com";
    public static final String MQTT_PORTS = "1883,8000";
    
    public static final String DEFAULT_WIDTH = "640";
    public static final String DEFAULT_HEIGHT = "480";
    public static final String DEFAULT_FPS = "20";
    public static final String DEFAULT_EXP = "128";
    public static final String DEFAULT_BR = "128";

    public static final String NODE_KITCHEN = "FrmEsp32/kitchen";
    public static final String NODE_STUDY = "study";
    public static final String NODE_DOOR = "FrmEsp32/Securitymaindoor";
    public static final String NODE_TOILET = "FrmEsp32/toilet";
    public static final String NODE_RO_PUMP = "ro_pump";
    public static final String NODE_KITCHEN_FAN = "kitchen_fan";
    public static final String NODE_PI_KITCHEN = "pi_kitchen";
}
