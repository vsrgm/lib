plugins {
    alias(libs.plugins.android.application)
    id("com.google.gms.google-services")
}

android {
    namespace = "com.example.sample"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.smarthome.iot"
        minSdk = 24
        targetSdk = 33
        versionCode = 159
        versionName = "1.0.159"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        viewBinding = true
        buildConfig = true
    }
}

tasks.register("autoUpdateVersion") {
    group = "versioning"
    doLast {
        val isMinor = project.hasProperty("minorUpdate")
        val gradleFile = file("build.gradle.kts")
        val javaFile = file("src/main/java/com/example/sample/SmartHomeActivity.java")
        val studyInoFile = file("../../NodeMcu/StudyRoomMonitor/StudyRoomMonitor.ino")

        var content = gradleFile.readText()
        val versionMatch = Regex("versionName = \"(.*?)\"").find(content)
        if (versionMatch != null) {
            val currentVersion = versionMatch.groupValues[1]
            val parts = currentVersion.split(".").toMutableList()
            while (parts.size < 3) parts.add("0")
            
            var major = parts[0].toInt()
            var minor = parts[1].toInt()
            var patch = parts[2].toInt()

            if (isMinor) {
                minor++
                patch = 0
            } else {
                patch++
            }

            val newVersion = "$major.$minor.$patch"
            
            // Update Gradle
            var newContent = content.replace(Regex("versionName = \".*?\""), "versionName = \"$newVersion\"")
            val vcMatch = Regex("versionCode = (\\d+)").find(newContent)
            if (vcMatch != null) {
                val newVc = vcMatch.groupValues[1].toInt() + 1
                newContent = newContent.replace(Regex("versionCode = \\d+"), "versionCode = $newVc")
            }
            gradleFile.writeText(newContent)

            // Update Java
            if (javaFile.exists()) {
                val javaContent = javaFile.readText()
                val newJava = javaContent.replace(Regex("private static final String APP_VERSION = \".*?\";"), 
                                                "private static final String APP_VERSION = \"$newVersion\";")
                javaFile.writeText(newJava)
            }

            // Update INO
            if (studyInoFile.exists()) {
                val inoContent = studyInoFile.readText()
                val newIno = inoContent.replace(Regex("const String SW_VERSION = \".*?\";"), 
                                              "const String SW_VERSION = \"$newVersion\";")
                studyInoFile.writeText(newIno)
            }
            
            println("Version automatically updated to $newVersion")
        }
    }
}

tasks.named("preBuild") {
    dependsOn("autoUpdateVersion")
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.constraintlayout)
    implementation(libs.material)
    implementation("org.eclipse.paho:org.eclipse.paho.client.mqttv3:1.2.5")
    implementation(libs.androidx.media3.exoplayer)
    implementation(libs.androidx.media3.exoplayer.rtsp)
    implementation(libs.androidx.media3.ui)
    implementation(platform(libs.firebase.bom))
    implementation(libs.firebase.database)
    implementation(libs.firebase.auth)
    testImplementation(libs.junit)
    androidTestImplementation(libs.espresso.core)
    androidTestImplementation(libs.ext.junit)
}