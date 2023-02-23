# -------------------------------------------------
# Project created by QtCreator 2012-01-13T12:13:03
# -------------------------------------------------
TARGET = convertor
TEMPLATE = app
SOURCES += main.cpp \
    convert.cpp \
    fmt_convert.cpp \
	bayer.c
HEADERS += convert.h \
    fmt_convert.h \
	bayer.h

FORMS += convert.ui
TEMPLATE += app
QT += gui widgets
