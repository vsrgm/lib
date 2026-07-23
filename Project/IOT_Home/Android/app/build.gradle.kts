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
        versionCode = 304
        versionName = "1.0.304"

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
        val inoFiles = listOf(
            file("../../NodeMcu/StudyRoomMonitor/StudyRoomMonitor.ino"),
            file("../../NodeMcu/RO_Waste_WaterPump/RO_Waste_WaterPump.ino"),
            file("../../NodeMcu/ToiletAssistance/ToiletAssistance.ino"),
            file("../../NodeMcu/Kitchen_Exhaust_Fan/Kitchen_Exhaust_Fan.ino"),
            file("../../ESP32/ESP32_CAM_Kitchen/ESP32_CAM_Kitchen.ino"),
            file("../../ESP32/ESP32_CAM_MainDoor/ESP32_CAM_MainDoor.ino")
        )
        val cFiles = listOf(
            file("../../pi/pikitchenMonitor/server.c"),
            file("../../pi/rccar/server.c")
        )

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

            // Update all INO files
            inoFiles.forEach { inoFile ->
                if (inoFile.exists()) {
                    val inoContent = inoFile.readText()
                    val newIno = inoContent.replace(Regex("const String SW_VERSION = \".*?\";"), 
                                                  "const String SW_VERSION = \"$newVersion\";")
                    inoFile.writeText(newIno)
                    println("Updated ${inoFile.name} to version $newVersion")
                }
            }

            // Update all C files
            cFiles.forEach { cFile ->
                if (cFile.exists()) {
                    val cContent = cFile.readText()
                    val newC = if (cContent.contains("#define SW_VERSION")) {
                        cContent.replace(Regex("#define SW_VERSION \".*?\""), 
                                       "#define SW_VERSION \"$newVersion\"")
                    } else {
                        // If it doesn't have it, add it after includes
                        val includesEnd = cContent.lastIndexOf("#include")
                        val nextLine = cContent.indexOf("\n", includesEnd) + 1
                        cContent.substring(0, nextLine) + "\n#define SW_VERSION \"$newVersion\"\n" + cContent.substring(nextLine)
                    }
                    cFile.writeText(newC)
                    println("Updated ${cFile.name} to version $newVersion")
                }
            }
            
            println("Project version automatically updated to $newVersion")
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