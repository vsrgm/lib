/********************************************************************************
** Form generated from reading UI file 'convert.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
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
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

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
            convert->setObjectName(QStringLiteral("convert"));
        convert->setWindowModality(Qt::NonModal);
        convert->setEnabled(true);
        convert->resize(1073, 642);
        centralWidget = new QWidget(convert);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        horizontalLayout_2 = new QHBoxLayout(centralWidget);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        verticalLayout_7 = new QVBoxLayout();
        verticalLayout_7->setSpacing(6);
        verticalLayout_7->setObjectName(QStringLiteral("verticalLayout_7"));
        verticalLayout_7->setSizeConstraint(QLayout::SetNoConstraint);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        label = new QLabel(centralWidget);
        label->setObjectName(QStringLiteral("label"));

        verticalLayout->addWidget(label);

        file_path = new QLineEdit(centralWidget);
        file_path->setObjectName(QStringLiteral("file_path"));

        verticalLayout->addWidget(file_path);

        label_5 = new QLabel(centralWidget);
        label_5->setObjectName(QStringLiteral("label_5"));

        verticalLayout->addWidget(label_5);

        Source_img_integrity = new QLabel(centralWidget);
        Source_img_integrity->setObjectName(QStringLiteral("Source_img_integrity"));

        verticalLayout->addWidget(Source_img_integrity);


        horizontalLayout->addLayout(verticalLayout);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        label_2 = new QLabel(centralWidget);
        label_2->setObjectName(QStringLiteral("label_2"));

        verticalLayout_2->addWidget(label_2);

        width = new QLineEdit(centralWidget);
        width->setObjectName(QStringLiteral("width"));

        verticalLayout_2->addWidget(width);

        label_6 = new QLabel(centralWidget);
        label_6->setObjectName(QStringLiteral("label_6"));

        verticalLayout_2->addWidget(label_6);

        frame_stride = new QLineEdit(centralWidget);
        frame_stride->setObjectName(QStringLiteral("frame_stride"));

        verticalLayout_2->addWidget(frame_stride);


        horizontalLayout->addLayout(verticalLayout_2);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        label_3 = new QLabel(centralWidget);
        label_3->setObjectName(QStringLiteral("label_3"));

        verticalLayout_3->addWidget(label_3);

        height = new QLineEdit(centralWidget);
        height->setObjectName(QStringLiteral("height"));

        verticalLayout_3->addWidget(height);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);


        horizontalLayout->addLayout(verticalLayout_3);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_3);

        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        label_7 = new QLabel(centralWidget);
        label_7->setObjectName(QStringLiteral("label_7"));

        verticalLayout_4->addWidget(label_7);

        num_frames = new QLabel(centralWidget);
        num_frames->setObjectName(QStringLiteral("num_frames"));
        num_frames->setAlignment(Qt::AlignCenter);

        verticalLayout_4->addWidget(num_frames);

        label_8 = new QLabel(centralWidget);
        label_8->setObjectName(QStringLiteral("label_8"));

        verticalLayout_4->addWidget(label_8);

        file_size = new QLabel(centralWidget);
        file_size->setObjectName(QStringLiteral("file_size"));
        file_size->setAlignment(Qt::AlignCenter);

        verticalLayout_4->addWidget(file_size);


        horizontalLayout->addLayout(verticalLayout_4);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_4);

        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setObjectName(QStringLiteral("verticalLayout_5"));
        label_9 = new QLabel(centralWidget);
        label_9->setObjectName(QStringLiteral("label_9"));

        verticalLayout_5->addWidget(label_9);

        bpp_pad = new QLineEdit(centralWidget);
        bpp_pad->setObjectName(QStringLiteral("bpp_pad"));

        verticalLayout_5->addWidget(bpp_pad);

        src_img_count = new QSpinBox(centralWidget);
        src_img_count->setObjectName(QStringLiteral("src_img_count"));

        verticalLayout_5->addWidget(src_img_count);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer_3);


        horizontalLayout->addLayout(verticalLayout_5);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_5);

        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setObjectName(QStringLiteral("verticalLayout_6"));
        label_4 = new QLabel(centralWidget);
        label_4->setObjectName(QStringLiteral("label_4"));

        verticalLayout_6->addWidget(label_4);

        pixel_fmt = new QComboBox(centralWidget);
        pixel_fmt->setObjectName(QStringLiteral("pixel_fmt"));

        verticalLayout_6->addWidget(pixel_fmt);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_6->addItem(verticalSpacer_2);


        horizontalLayout->addLayout(verticalLayout_6);


        verticalLayout_7->addLayout(horizontalLayout);

        verticalSpacer_4 = new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_7->addItem(verticalSpacer_4);

        draw_window = new QLabel(centralWidget);
        draw_window->setObjectName(QStringLiteral("draw_window"));
        draw_window->setMinimumSize(QSize(640, 480));
        draw_window->setMaximumSize(QSize(1280, 800));
        draw_window->setScaledContents(true);

        verticalLayout_7->addWidget(draw_window);


        horizontalLayout_2->addLayout(verticalLayout_7);

        convert->setCentralWidget(centralWidget);

        retranslateUi(convert);

        QMetaObject::connectSlotsByName(convert);
    } // setupUi

    void retranslateUi(QMainWindow *convert)
    {
        convert->setWindowTitle(QApplication::translate("convert", "convert", Q_NULLPTR));
        label->setText(QApplication::translate("convert", "File path", Q_NULLPTR));
        file_path->setText(QApplication::translate("convert", "/home/pie5zk/sample_skip10_frame.raw", Q_NULLPTR));
        label_5->setText(QApplication::translate("convert", "Source integrity", Q_NULLPTR));
        Source_img_integrity->setText(QApplication::translate("convert", "FAIL", Q_NULLPTR));
        label_2->setText(QApplication::translate("convert", "Width", Q_NULLPTR));
        width->setText(QApplication::translate("convert", "2592", Q_NULLPTR));
        label_6->setText(QApplication::translate("convert", "Frame stride", Q_NULLPTR));
        frame_stride->setText(QApplication::translate("convert", "0", Q_NULLPTR));
        label_3->setText(QApplication::translate("convert", "Height", Q_NULLPTR));
        height->setText(QApplication::translate("convert", "1944", Q_NULLPTR));
        label_7->setText(QApplication::translate("convert", "Number of frames", Q_NULLPTR));
        num_frames->setText(QApplication::translate("convert", "0", Q_NULLPTR));
        label_8->setText(QApplication::translate("convert", "Filesize", Q_NULLPTR));
        file_size->setText(QApplication::translate("convert", "0", Q_NULLPTR));
        label_9->setText(QApplication::translate("convert", "Bpp padded", Q_NULLPTR));
        bpp_pad->setText(QApplication::translate("convert", "0", Q_NULLPTR));
        label_4->setText(QApplication::translate("convert", "Pixel format", Q_NULLPTR));
        draw_window->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class convert: public Ui_convert {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONVERT_H
