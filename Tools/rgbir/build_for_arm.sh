source /opt/sec-auto/3.1.21/environment-setup-aarch64-poky-linux
uic rgbir.ui -o ui_rgbir.h
$CXX `/opt/sec-auto/3.1.21/sysroots/x86_64-pokysdk-linux/usr/bin/pkg-config --cflags --libs Qt5Gui Qt5Core Qt5Widgets` main.cpp  rgbir.cpp moc_rgbir.cpp -I. -o rgbir.elf -lpthread
#To Execute in Target
#QT_QPA_FONTDIR=/usr/share/fonts/ttf/ ./convertor.elf
