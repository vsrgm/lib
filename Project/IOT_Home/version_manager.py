import os
import re
import sys

# Get the directory where this script is located (the project root)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths relative to project root
ANDROID_GRADLE_PATH = os.path.join(ROOT_DIR, "Android/app/build.gradle.kts")
ANDROID_JAVA_PATH = os.path.join(ROOT_DIR, "Android/app/src/main/java/com/example/sample/SmartHomeActivity.java")
INO_PATH = os.path.join(ROOT_DIR, "NodeMcu/SmartHome/SmartHome.ino")

def get_current_version():
    if not os.path.exists(ANDROID_GRADLE_PATH):
        return "1.0.0"
    with open(ANDROID_GRADLE_PATH, "r") as f:
        content = f.read()
        match = re.search(r'versionName = "(.*?)"', content)
        if match:
            return match.group(1)
    return "1.0.0"

def update_version(new_version):
    # 1. Update build.gradle.kts
    if os.path.exists(ANDROID_GRADLE_PATH):
        with open(ANDROID_GRADLE_PATH, "r") as f:
            content = f.read()
        new_content = re.sub(r'versionName = ".*?"', f'versionName = "{new_version}"', content)
        # Also increment versionCode
        vc_match = re.search(r'versionCode = (\d+)', new_content)
        if vc_match:
            new_vc = int(vc_match.group(1)) + 1
            new_content = re.sub(r'versionCode = \d+', f'versionCode = {new_vc}', new_content)
        
        with open(ANDROID_GRADLE_PATH, "w") as f:
            f.write(new_content)

    # 2. Update SmartHomeActivity.java
    if os.path.exists(ANDROID_JAVA_PATH):
        with open(ANDROID_JAVA_PATH, "r") as f:
            java_content = f.read()
        java_new = re.sub(r'private static final String APP_VERSION = ".*?";', 
                          f'private static final String APP_VERSION = "{new_version}";', java_content)
        with open(ANDROID_JAVA_PATH, "w") as f:
            f.write(java_new)

    # 3. Update SmartHome.ino
    if os.path.exists(INO_PATH):
        with open(INO_PATH, "r") as f:
            ino_content = f.read()
        ino_new = re.sub(r'const String SW_VERSION = ".*?";', 
                         f'const String SW_VERSION = "{new_version}";', ino_content)
        with open(INO_PATH, "w") as f:
            f.write(ino_new)
    
    print(f"Version updated to {new_version}")

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "patch"
    current = get_current_version()
    parts = current.split(".")
    while len(parts) < 3: parts.append("0")
    
    try:
        major, minor, patch = map(int, parts)
    except ValueError:
        major, minor, patch = 1, 0, 0
    
    if mode == "minor":
        minor += 1
        patch = 0
    else: # patch
        patch += 1

    new_version = f"{major}.{minor}.{patch}"
    update_version(new_version)

if __name__ == "__main__":
    main()
