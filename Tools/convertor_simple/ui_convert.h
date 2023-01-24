/********************************************************************************
** Form generated from reading UI file 'convert.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CONVERT_H
#define UI_CONVERT_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QMainWindow>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_convert
{
public:
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout_2;
    QVBoxLayout *verticalLayout_7;
    QHBoxLayout *horizontalLayout;
    QVBoxLayout *verticalLayout;
    QLabel *label;
    QLineEdit *file_path;
    QLabel *label_5;
    QLabel *Source_img_integrity;
    QSpacerItem *horizontalSpacer;
    QVBoxLayout *verticalLayout_2;
    QLabel *label_2;
    QLineEdit *width;
    QLabel *label_6;
    QLineEdit *frame_stride;
    QSpacerItem *horizontalSpacer_2;
    QVBoxLayout *verticalLayout_3;
    QLabel *label_3;
    QLineEdit *height;
    QSpacerItem *verticalSpacer;
    QSpacerItem *horizontalSpacer_3;
    QVBoxLayout *verticalLayout_4;
    QLabel *label_7;
    QLabel *num_frames;
    QLabel *label_8;
    QLabel *file_size;
    QSpacerItem *horizontalSpacer_4;
    QVBoxLayout *verticalLayout_5;
    QLabel *label_9;
    QLineEdit *bpp_pad;
    QSpinBox *src_img_count;
    QSpacerItem *verticalSpacer_3;
    QSpacerItem *horizontalSpacer_5;
    QVBoxLayout *verticalLayout_6;
    QLabel *label_4;
    QComboBox *pixel_fmt;
    QSpacerItem *verticalSpacer_2;
    QSpacerItem *verticalSpacer_4;
    QLabel *draw_window;

    void setupUi(QMainWindow *convert)
    {
        if (convert->objectName().isEmpty())
            convert->setObjectName(QString::fromUtf8("convert"));
        convert->setWindowModality(Qt::NonModal);
        convert->setEnabled(true);
        centralWidget = new QWidget(convert);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        horizontalLayout_2 = new QHBoxLayout(centralWidget);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        verticalLayout_7 = new QVBoxLayout();
        verticalLayout_7->setSpacing(6);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        verticalLayout_7->setSizeConstraint(QLayout::SetNoConstraint);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        label = new QLabel(centralWidget);
        label->setObjectName(QString::fromUtf8("label"));

        verticalLayout->addWidget(label);

        file_path = new QLineEdit(centralWidget);
        file_path->setObjectName(QString::fromUtf8("file_path"));

        verticalLayout->addWidget(file_path);

        label_5 = new QLabel(centralWidget);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        verticalLayout->addWidget(label_5);

        Source_img_integrity = new QLabel(centralWidget);
        Source_img_integrity->setObjectName(QString::fromUtf8("Source_img_integrity"));

        verticalLayout->addWidget(Source_img_integrity);


        horizontalLayout->addLayout(verticalLayout);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        label_2 = new QLabel(centralWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        verticalLayout_2->addWidget(label_2);

        width = new QLineEdit(centralWidget);
        width->setObjectName(QString::fromUtf8("width"));

        verticalLayout_2->addWidget(width);

        label_6 = new QLabel(centralWidget);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        verticalLayout_2->addWidget(label_6);

        frame_stride = new QLineEdit(centralWidget);
        frame_stride->setObjectName(QString::fromUtf8("frame_stride"));

        verticalLayout_2->addWidget(frame_stride);


        horizontalLayout->addLayout(verticalLayout_2);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        label_3 = new QLabel(centralWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        verticalLayout_3->addWidget(label_3);

        height = new QLineEdit(centralWidget);
        height->setObjectName(QString::fromUtf8("height"));

        verticalLayout_3->addWidget(height);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);


        horizontalLayout->addLayout(verticalLayout_3);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_3);

        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        label_7 = new QLabel(centralWidget);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        verticalLayout_4->addWidget(label_7);

        num_frames = new QLabel(centralWidget);
        num_frames->setObjectName(QString::fromUtf8("num_frames"));
        num_frames->setAlignment(Qt::AlignCenter);

        verticalLayout_4->addWidget(num_frames);

        label_8 = new QLabel(centralWidget);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        verticalLayout_4->addWidget(label_8);

        file_size = new QLabel(centralWidget);
        file_size->setObjectName(QString::fromUtf8("file_size"));
        file_size->setAlignment(Qt::AlignCenter);

        verticalLayout_4->addWidget(file_size);


        horizontalLayout->addLayout(verticalLayout_4);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_4);

        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        label_9 = new QLabel(centralWidget);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        verticalLayout_5->addWidget(label_9);

        bpp_pad = new QLineEdit(centralWidget);
        bpp_pad->setObjectName(QString::fromUtf8("bpp_pad"));

        verticalLayout_5->addWidget(bpp_pad);

        src_img_count = new QSpinBox(centralWidget);
        src_img_count->setObjectName(QString::fromUtf8("src_img_count"));

        verticalLayout_5->addWidget(src_img_count);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer_3);


        horizontalLayout->addLayout(verticalLayout_5);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_5);

        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        label_4 = new QLabel(centralWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        verticalLayout_6->addWidget(label_4);

        pixel_fmt = new QComboBox(centralWidget);
        pixel_fmt->setObjectName(QString::fromUtf8("pixel_fmt"));

        verticalLayout_6->addWidget(pixel_fmt);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_6->addItem(verticalSpacer_2);


        horizontalLayout->addLayout(verticalLayout_6);


        verticalLayout_7->addLayout(horizontalLayout);

        verticalSpacer_4 = new QSpacerItem(QSizePolicy::Expanding, QSizePolicy::Minimum);

        verticalLayout_7->addItem(verticalSpacer_4);

        draw_window = new QLabel(centralWidget);
        draw_window->setObjectName(QString::fromUtf8("draw_window"));
        draw_window->setMinimumSize(QSize(640, 480));
        draw_window->setMaximumSize(QSize(1280, 800));
        draw_window->setScaledContents(true);

        verticalLayout_7->addWidget(draw_window);

        verticalLayout_7->setStretch(0, 1);

        horizontalLayout_2->addLayout(verticalLayout_7);

        convert->setCentralWidget(centralWidget);

        retranslateUi(convert);

        QMetaObject::connectSlotsByName(convert);
    } // setupUi

    void retranslateUi(QMainWindow *convert)
    {
        convert->setWindowTitle(QApplication::translate("convert", "convert", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("convert", "File path", 0, QApplication::UnicodeUTF8));
        file_path->setText(QApplication::translate("convert", "/mnt/hgfs/Host/frame_0_50.raw", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("convert", "Source integrity", 0, QApplication::UnicodeUTF8));
        Source_img_integrity->setText(QApplication::translate("convert", "FAIL", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("convert", "Width", 0, QApplication::UnicodeUTF8));
        width->setText(QApplication::translate("convert", "1600", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("convert", "Frame stride", 0, QApplication::UnicodeUTF8));
        frame_stride->setText(QApplication::translate("convert", "0", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("convert", "Height", 0, QApplication::UnicodeUTF8));
        height->setText(QApplication::translate("convert", "1300", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("convert", "Number of frames", 0, QApplication::UnicodeUTF8));
        num_frames->setText(QApplication::translate("convert", "0", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("convert", "Filesize", 0, QApplication::UnicodeUTF8));
        file_size->setText(QApplication::translate("convert", "0", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("convert", "Bpp padded", 0, QApplication::UnicodeUTF8));
        bpp_pad->setText(QApplication::translate("convert", "0", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("convert", "Pixel format", 0, QApplication::UnicodeUTF8));
        draw_window->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class convert: public Ui_convert {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONVERT_H
