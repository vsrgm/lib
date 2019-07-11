/********************************************************************************
** Form generated from reading UI file 'convert.ui'
**
** Created by: Qt User Interface Compiler version 5.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CONVERT_H
#define UI_CONVERT_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_convert
{
public:
    QWidget *centralWidget;
    QLabel *draw_window;
    QLineEdit *file_path;
    QComboBox *pixel_fmt;
    QLineEdit *width;
    QLineEdit *height;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QLabel *label_5;
    QLabel *Source_img_integrity;
    QSpinBox *src_img_count;
    QLabel *num_frames;
    QLabel *label_7;
    QLabel *file_size;
    QLabel *label_8;
    QFrame *line;
    QFrame *line_2;
    QLineEdit *frame_stride;
    QLabel *label_6;
    QLineEdit *bpp_pad;
    QLabel *label_9;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *convert)
    {
        if (convert->objectName().isEmpty())
            convert->setObjectName(QStringLiteral("convert"));
        convert->resize(770, 515);
        centralWidget = new QWidget(convert);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        draw_window = new QLabel(centralWidget);
        draw_window->setObjectName(QStringLiteral("draw_window"));
        draw_window->setGeometry(QRect(20, 119, 461, 321));
        draw_window->setScaledContents(true);
        file_path = new QLineEdit(centralWidget);
        file_path->setObjectName(QStringLiteral("file_path"));
        file_path->setGeometry(QRect(10, 20, 151, 21));
        pixel_fmt = new QComboBox(centralWidget);
        pixel_fmt->setObjectName(QStringLiteral("pixel_fmt"));
        pixel_fmt->setGeometry(QRect(630, 20, 111, 21));
        width = new QLineEdit(centralWidget);
        width->setObjectName(QStringLiteral("width"));
        width->setGeometry(QRect(180, 20, 81, 21));
        height = new QLineEdit(centralWidget);
        height->setObjectName(QStringLiteral("height"));
        height->setGeometry(QRect(270, 20, 81, 21));
        label = new QLabel(centralWidget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(10, 0, 141, 17));
        label_2 = new QLabel(centralWidget);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(180, 0, 62, 17));
        label_3 = new QLabel(centralWidget);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(270, 0, 62, 17));
        label_4 = new QLabel(centralWidget);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(640, 0, 111, 17));
        label_5 = new QLabel(centralWidget);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(10, 50, 111, 17));
        Source_img_integrity = new QLabel(centralWidget);
        Source_img_integrity->setObjectName(QStringLiteral("Source_img_integrity"));
        Source_img_integrity->setGeometry(QRect(30, 70, 62, 17));
        src_img_count = new QSpinBox(centralWidget);
        src_img_count->setObjectName(QStringLiteral("src_img_count"));
        src_img_count->setGeometry(QRect(470, 20, 61, 21));
        num_frames = new QLabel(centralWidget);
        num_frames->setObjectName(QStringLiteral("num_frames"));
        num_frames->setGeometry(QRect(390, 20, 51, 17));
        num_frames->setAlignment(Qt::AlignCenter);
        label_7 = new QLabel(centralWidget);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setGeometry(QRect(360, 0, 131, 17));
        file_size = new QLabel(centralWidget);
        file_size->setObjectName(QStringLiteral("file_size"));
        file_size->setGeometry(QRect(360, 70, 101, 17));
        file_size->setAlignment(Qt::AlignCenter);
        label_8 = new QLabel(centralWidget);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setGeometry(QRect(360, 50, 62, 17));
        line = new QFrame(centralWidget);
        line->setObjectName(QStringLiteral("line"));
        line->setGeometry(QRect(0, 40, 791, 16));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);
        line_2 = new QFrame(centralWidget);
        line_2->setObjectName(QStringLiteral("line_2"));
        line_2->setGeometry(QRect(0, 90, 791, 16));
        line_2->setFrameShape(QFrame::HLine);
        line_2->setFrameShadow(QFrame::Sunken);
        frame_stride = new QLineEdit(centralWidget);
        frame_stride->setObjectName(QStringLiteral("frame_stride"));
        frame_stride->setGeometry(QRect(180, 70, 81, 21));
        label_6 = new QLabel(centralWidget);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(180, 50, 111, 17));
        bpp_pad = new QLineEdit(centralWidget);
        bpp_pad->setObjectName(QStringLiteral("bpp_pad"));
        bpp_pad->setGeometry(QRect(550, 20, 41, 21));
        label_9 = new QLabel(centralWidget);
        label_9->setObjectName(QStringLiteral("label_9"));
        label_9->setGeometry(QRect(550, 0, 61, 17));
        convert->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(convert);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 770, 21));
        convert->setMenuBar(menuBar);
        mainToolBar = new QToolBar(convert);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        convert->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(convert);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        convert->setStatusBar(statusBar);

        retranslateUi(convert);

        QMetaObject::connectSlotsByName(convert);
    } // setupUi

    void retranslateUi(QMainWindow *convert)
    {
        convert->setWindowTitle(QApplication::translate("convert", "convert", Q_NULLPTR));
        draw_window->setText(QString());
        file_path->setText(QApplication::translate("convert", "C:\\Users\\agururaj\\Downloads\\ispinput", Q_NULLPTR));
        width->setText(QApplication::translate("convert", "160", Q_NULLPTR));
        height->setText(QApplication::translate("convert", "120", Q_NULLPTR));
        label->setText(QApplication::translate("convert", "File path", Q_NULLPTR));
        label_2->setText(QApplication::translate("convert", "Width", Q_NULLPTR));
        label_3->setText(QApplication::translate("convert", "Height", Q_NULLPTR));
        label_4->setText(QApplication::translate("convert", "Pixel format", Q_NULLPTR));
        label_5->setText(QApplication::translate("convert", "Source integrity", Q_NULLPTR));
        Source_img_integrity->setText(QApplication::translate("convert", "FAIL", Q_NULLPTR));
        num_frames->setText(QApplication::translate("convert", "0", Q_NULLPTR));
        label_7->setText(QApplication::translate("convert", "Number of frames", Q_NULLPTR));
        file_size->setText(QApplication::translate("convert", "0", Q_NULLPTR));
        label_8->setText(QApplication::translate("convert", "Filesize", Q_NULLPTR));
        frame_stride->setText(QApplication::translate("convert", "0", Q_NULLPTR));
        label_6->setText(QApplication::translate("convert", "Frame stride", Q_NULLPTR));
        bpp_pad->setText(QApplication::translate("convert", "0", Q_NULLPTR));
        label_9->setText(QApplication::translate("convert", "Bpp padded", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class convert: public Ui_convert {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONVERT_H
