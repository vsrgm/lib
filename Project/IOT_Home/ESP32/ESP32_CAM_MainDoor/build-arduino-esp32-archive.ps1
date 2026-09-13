param(
  [string]$ArduinoEsp32Version = "3.3.11"
)

$ErrorActionPreference = "Stop"

$SourceRoot = $PSScriptRoot
$ArduinoPackage = Join-Path $env:LOCALAPPDATA "Arduino15\packages\esp32"
$Sdk = Join-Path $ArduinoPackage "tools\esp32-libs\$ArduinoEsp32Version"
$Core = Join-Path $ArduinoPackage "hardware\esp32\$ArduinoEsp32Version\cores\esp32"
$Toolchain = Join-Path $ArduinoPackage "tools\esp-x32\2601\bin"
$Gcc = Join-Path $Toolchain "xtensa-esp32-elf-gcc.exe"
$Gxx = Join-Path $Toolchain "xtensa-esp32-elf-g++.exe"
$Ar = Join-Path $Toolchain "xtensa-esp32-elf-gcc-ar.exe"
$BuildDir = Join-Path $SourceRoot "build-arduino-esp32-$ArduinoEsp32Version"
$ObjDir = Join-Path $BuildDir "obj"
$Archive = Join-Path $BuildDir "libespressif__esp32-camera.a"
$InstallArchive = Join-Path $Sdk "lib\libespressif__esp32-camera.a"

foreach ($Path in @($Sdk, $Core, $Gcc, $Gxx, $Ar)) {
  if (-not (Test-Path $Path)) {
    throw "Required path not found: $Path"
  }
}

New-Item -ItemType Directory -Force -Path $ObjDir | Out-Null
Remove-Item -Path (Join-Path $ObjDir "*.obj") -Force -ErrorAction SilentlyContinue

$CommonArgs = @(
  "-w",
  "-Os",
  "-Werror=return-type",
  "-Werror=unused-result",
  "-DF_CPU=240000000L",
  "-DARDUINO=10607",
  "-DARDUINO_ESP32_DEV",
  "-DARDUINO_ARCH_ESP32",
  "-DARDUINO_BOARD=`"ESP32_DEV`"",
  "-DARDUINO_VARIANT=`"esp32`"",
  "-DARDUINO_PARTITION_default",
  "-DARDUINO_HOST_OS=`"windows`"",
  "-DARDUINO_FQBN=`"esp32:esp32:esp32`"",
  "-DESP32=ESP32",
  "-DCORE_DEBUG_LEVEL=0",
  "-DARDUINO_USB_CDC_ON_BOOT=0",
  "@$Sdk\flags\defines",
  "-I$SourceRoot",
  "-I$Core",
  "-iprefix",
  "$Sdk\include\",
  "@$Sdk\flags\includes",
  "-I$Sdk\qio_qspi\include",
  "-I$SourceRoot\esp32-camera\driver\include",
  "-I$SourceRoot\esp32-camera\conversions\include",
  "-I$SourceRoot\esp32-camera\driver\private_include",
  "-I$SourceRoot\esp32-camera\conversions\private_include",
  "-I$SourceRoot\esp32-camera\sensors\private_include",
  "-I$SourceRoot\esp32-camera\target\private_include",
  "-I$SourceRoot\esp32-camera\target\esp32\private_include"
)

$CSources = @(
  "esp32-camera/conversions/yuv.c",
  "esp32-camera/conversions/to_bmp.c",
  "esp32-camera/driver/esp_camera.c",
  "esp32-camera/driver/esp_camera_af.c",
  "esp32-camera/driver/cam_hal.c",
  "esp32-camera/driver/sensor.c",
  "esp32-camera/sensors/ov2640.c",
  "esp32-camera/sensors/ov3660.c",
  "esp32-camera/sensors/ov5640.c",
  "esp32-camera/sensors/ov5640_af.c",
  "esp32-camera/sensors/ov7725.c",
  "esp32-camera/sensors/ov7670.c",
  "esp32-camera/sensors/nt99141.c",
  "esp32-camera/sensors/gc0308.c",
  "esp32-camera/sensors/gc2145.c",
  "esp32-camera/sensors/gc032a.c",
  "esp32-camera/sensors/bf3005.c",
  "esp32-camera/sensors/bf20a6.c",
  "esp32-camera/sensors/sc101iot.c",
  "esp32-camera/sensors/sc030iot.c",
  "esp32-camera/sensors/sc031gs.c",
  "esp32-camera/sensors/mega_ccm.c",
  "esp32-camera/sensors/hm1055.c",
  "esp32-camera/sensors/hm0360.c",
  "esp32-camera/target/xclk.c",
  "esp32-camera/target/esp32/ll_cam.c",
  "esp32-camera/driver/sccb-ng.c"
)

$CppSources = @(
  "esp32-camera/conversions/to_jpg.cpp",
  "esp32-camera/conversions/jpge.cpp"
)

$Objects = New-Object System.Collections.Generic.List[string]

foreach ($Source in $CSources) {
  $Object = Join-Path $ObjDir ((Split-Path $Source -Leaf) + ".obj")
  Write-Host "CC  $Source"
  & $Gcc @("-MMD", "-c", "@$Sdk\flags\c_flags") @CommonArgs (Join-Path $SourceRoot $Source) @("-o", $Object)
  if ($LASTEXITCODE -ne 0) { throw "Compile failed: $Source" }
  $Objects.Add($Object)
}

foreach ($Source in $CppSources) {
  $Object = Join-Path $ObjDir ((Split-Path $Source -Leaf) + ".obj")
  Write-Host "CXX $Source"
  & $Gxx @("-MMD", "-c", "@$Sdk\flags\cpp_flags") @CommonArgs (Join-Path $SourceRoot $Source) @("-o", $Object)
  if ($LASTEXITCODE -ne 0) { throw "Compile failed: $Source" }
  $Objects.Add($Object)
}

Remove-Item -Path $Archive -Force -ErrorAction SilentlyContinue
& $Ar cr $Archive @Objects
if ($LASTEXITCODE -ne 0) { throw "Archive creation failed" }

Copy-Item -Path $Archive -Destination $InstallArchive -Force
& $Ar t $InstallArchive | Set-Content (Join-Path $BuildDir "installed-archive-members.txt")

Write-Host "Built:     $Archive"
Write-Host "Installed: $InstallArchive"
Write-Host "Objects:   $($Objects.Count)"