# -------------------------------------------------
# IoT Home Qt x86 Desktop Application
# -------------------------------------------------
TARGET = iothome
TEMPLATE = app

QT += core gui widgets network

HEADERS += \
    mainwindow.h

SOURCES += \
    main.cpp \
    mainwindow.cpp

FORMS += \
    mainwindow.ui

CONFIG += c++11
