/********************************************************************************
** Form generated from reading UI file 'convert.ui'
**
** Created by: Qt User Interface Compiler version 5.15.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CONVERT_H
#define UI_CONVERT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
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
    QSpacerItem *verticalSpacer_7;
    QFrame *line;
    QVBoxLayout *verticalLayout_2;
    QLabel *label_2;
    QLineEdit *width;
    QLabel *label_6;
    QLineEdit *frame_stride;
    QSpacerItem *verticalSpacer_6;
    QFrame *line_2;
    QVBoxLayout *verticalLayout_3;
    QLabel *label_3;
    QLineEdit *height;
    QSpacerItem *verticalSpacer;
    QFrame *line_3;
    QVBoxLayout *verticalLayout_4;
    QLabel *label_7;
    QLabel *num_frames;
    QLabel *label_8;
    QLabel *file_size;
    QSpacerItem *verticalSpacer_8;
    QFrame *line_4;
    QVBoxLayout *verticalLayout_5;
    QLabel *label_9;
    QLineEdit *bpp_pad;
    QLabel *label_10;
    QSpinBox *src_img_count;
    QSpacerItem *verticalSpacer_4;
    QFrame *line_5;
    QVBoxLayout *verticalLayout_6;
    QLabel *label_4;
    QComboBox *pixel_fmt;
    QSpacerItem *verticalSpacer_5;
    QSpacerItem *horizontalSpacer;
    QFrame *line_6;
    QHBoxLayout *horizontalLayout_5;
    QLabel *draw_window;
    QFrame *line_8;
    QVBoxLayout *verticalLayout_8;
    QPushButton *equalize;
    QSpacerItem *verticalSpacer_2;
    QFrame *line_7;
    QLabel *modified_draw_window;
    QFrame *line_9;
    QSpacerItem *verticalSpacer_3;

    void setupUi(QMainWindow *convert)
    {
        if (convert->objectName().isEmpty())
            convert->setObjectName(QString::fromUtf8("convert"));
        convert->setWindowModality(Qt::NonModal);
        convert->setEnabled(true);
        convert->resize(1394, 653);
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
        verticalLayout_7->setContentsMargins(-1, -1, -1, 9);
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

        verticalSpacer_7 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer_7);


        horizontalLayout->addLayout(verticalLayout);

        line = new QFrame(centralWidget);
        line->setObjectName(QString::fromUtf8("line"));
        line->setFrameShape(QFrame::VLine);
        line->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line);

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

        verticalSpacer_6 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer_6);


        horizontalLayout->addLayout(verticalLayout_2);

        line_2 = new QFrame(centralWidget);
        line_2->setObjectName(QString::fromUtf8("line_2"));
        line_2->setFrameShape(QFrame::VLine);
        line_2->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line_2);

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

        line_3 = new QFrame(centralWidget);
        line_3->setObjectName(QString::fromUtf8("line_3"));
        line_3->setFrameShape(QFrame::VLine);
        line_3->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line_3);

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

        verticalSpacer_8 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_4->addItem(verticalSpacer_8);


        horizontalLayout->addLayout(verticalLayout_4);

        line_4 = new QFrame(centralWidget);
        line_4->setObjectName(QString::fromUtf8("line_4"));
        line_4->setFrameShape(QFrame::VLine);
        line_4->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line_4);

        verticalLayout_5 = new QVBoxLayout();
        verticalLayout_5->setSpacing(6);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        label_9 = new QLabel(centralWidget);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        verticalLayout_5->addWidget(label_9);

        bpp_pad = new QLineEdit(centralWidget);
        bpp_pad->setObjectName(QString::fromUtf8("bpp_pad"));

        verticalLayout_5->addWidget(bpp_pad);

        label_10 = new QLabel(centralWidget);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        verticalLayout_5->addWidget(label_10);

        src_img_count = new QSpinBox(centralWidget);
        src_img_count->setObjectName(QString::fromUtf8("src_img_count"));

        verticalLayout_5->addWidget(src_img_count);

        verticalSpacer_4 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_5->addItem(verticalSpacer_4);


        horizontalLayout->addLayout(verticalLayout_5);

        line_5 = new QFrame(centralWidget);
        line_5->setObjectName(QString::fromUtf8("line_5"));
        line_5->setFrameShape(QFrame::VLine);
        line_5->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line_5);

        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        label_4 = new QLabel(centralWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        verticalLayout_6->addWidget(label_4);

        pixel_fmt = new QComboBox(centralWidget);
        pixel_fmt->setObjectName(QString::fromUtf8("pixel_fmt"));

        verticalLayout_6->addWidget(pixel_fmt);

        verticalSpacer_5 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_6->addItem(verticalSpacer_5);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        verticalLayout_6->addItem(horizontalSpacer);


        horizontalLayout->addLayout(verticalLayout_6);


        verticalLayout_7->addLayout(horizontalLayout);

        line_6 = new QFrame(centralWidget);
        line_6->setObjectName(QString::fromUtf8("line_6"));
        line_6->setFrameShape(QFrame::HLine);
        line_6->setFrameShadow(QFrame::Sunken);

        verticalLayout_7->addWidget(line_6);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        horizontalLayout_5->setSizeConstraint(QLayout::SetFixedSize);
        draw_window = new QLabel(centralWidget);
        draw_window->setObjectName(QString::fromUtf8("draw_window"));
        draw_window->setMinimumSize(QSize(640, 480));
        draw_window->setScaledContents(true);

        horizontalLayout_5->addWidget(draw_window);

        line_8 = new QFrame(centralWidget);
        line_8->setObjectName(QString::fromUtf8("line_8"));
        line_8->setFrameShape(QFrame::VLine);
        line_8->setFrameShadow(QFrame::Sunken);

        horizontalLayout_5->addWidget(line_8);

        verticalLayout_8 = new QVBoxLayout();
        verticalLayout_8->setSpacing(10);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        verticalLayout_8->setSizeConstraint(QLayout::SetFixedSize);
        equalize = new QPushButton(centralWidget);
        equalize->setObjectName(QString::fromUtf8("equalize"));
        equalize->setMaximumSize(QSize(60, 40));

        verticalLayout_8->addWidget(equalize);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_8->addItem(verticalSpacer_2);


        horizontalLayout_5->addLayout(verticalLayout_8);

        line_7 = new QFrame(centralWidget);
        line_7->setObjectName(QString::fromUtf8("line_7"));
        line_7->setFrameShape(QFrame::VLine);
        line_7->setFrameShadow(QFrame::Sunken);

        horizontalLayout_5->addWidget(line_7);

        modified_draw_window = new QLabel(centralWidget);
        modified_draw_window->setObjectName(QString::fromUtf8("modified_draw_window"));
        modified_draw_window->setMinimumSize(QSize(640, 480));
        modified_draw_window->setScaledContents(true);

        horizontalLayout_5->addWidget(modified_draw_window);


        verticalLayout_7->addLayout(horizontalLayout_5);

        line_9 = new QFrame(centralWidget);
        line_9->setObjectName(QString::fromUtf8("line_9"));
        line_9->setFrameShape(QFrame::HLine);
        line_9->setFrameShadow(QFrame::Sunken);

        verticalLayout_7->addWidget(line_9);

        verticalSpacer_3 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_7->addItem(verticalSpacer_3);


        horizontalLayout_2->addLayout(verticalLayout_7);

        convert->setCentralWidget(centralWidget);

        retranslateUi(convert);

        QMetaObject::connectSlotsByName(convert);
    } // setupUi

    void retranslateUi(QMainWindow *convert)
    {
        convert->setWindowTitle(QCoreApplication::translate("convert", "convert", nullptr));
        label->setText(QCoreApplication::translate("convert", "File path", nullptr));
        file_path->setText(QCoreApplication::translate("convert", "/home/pie5zk/sample_skip10_frame.raw", nullptr));
        label_5->setText(QCoreApplication::translate("convert", "Source integrity", nullptr));
        Source_img_integrity->setText(QCoreApplication::translate("convert", "FAIL", nullptr));
        label_2->setText(QCoreApplication::translate("convert", "Width", nullptr));
        width->setText(QCoreApplication::translate("convert", "2592", nullptr));
        label_6->setText(QCoreApplication::translate("convert", "Frame stride", nullptr));
        frame_stride->setText(QCoreApplication::translate("convert", "0", nullptr));
        label_3->setText(QCoreApplication::translate("convert", "Height", nullptr));
        height->setText(QCoreApplication::translate("convert", "1944", nullptr));
        label_7->setText(QCoreApplication::translate("convert", "Number of frames", nullptr));
        num_frames->setText(QCoreApplication::translate("convert", "0", nullptr));
        label_8->setText(QCoreApplication::translate("convert", "Filesize", nullptr));
        file_size->setText(QCoreApplication::translate("convert", "0", nullptr));
        label_9->setText(QCoreApplication::translate("convert", "Bpp padded", nullptr));
        bpp_pad->setText(QCoreApplication::translate("convert", "0", nullptr));
        label_10->setText(QCoreApplication::translate("convert", "Image counter", nullptr));
        label_4->setText(QCoreApplication::translate("convert", "Pixel format", nullptr));
        draw_window->setText(QString());
        equalize->setText(QCoreApplication::translate("convert", "equalize", nullptr));
        modified_draw_window->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class convert: public Ui_convert {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONVERT_H
