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
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
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
    QLabel *label_10;
    QSpinBox *src_img_count;
    QSpacerItem *verticalSpacer_4;
    QFrame *line_5;
    QFormLayout *formLayout;
    QLabel *label_4;
    QComboBox *pixel_fmt;
    QLineEdit *bpp;
    QLabel *label_11;
    QLineEdit *bpp_pad;
    QLabel *label_9;
    QVBoxLayout *verticalLayout_6;
    QFrame *line_6;
    QHBoxLayout *horizontalLayout_5;
    QLabel *draw_window;
    QFrame *line_8;
    QGridLayout *gridLayout;
    QLineEdit *crop_height;
    QLineEdit *crop_x;
    QLabel *label_14;
    QPushButton *Crop;
    QSpacerItem *verticalSpacer_2;
    QLineEdit *crop_y;
    QPushButton *equalize;
    QLabel *label_15;
    QLabel *label_12;
    QLineEdit *crop_width;
    QLabel *label_13;
    QFrame *line_13;
    QFrame *line_10;
    QFrame *line_11;
    QFrame *line_12;
    QFrame *line_14;
    QFrame *line_15;
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

        formLayout = new QFormLayout();
        formLayout->setSpacing(6);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        label_4 = new QLabel(centralWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label_4);

        pixel_fmt = new QComboBox(centralWidget);
        pixel_fmt->setObjectName(QString::fromUtf8("pixel_fmt"));

        formLayout->setWidget(0, QFormLayout::FieldRole, pixel_fmt);

        bpp = new QLineEdit(centralWidget);
        bpp->setObjectName(QString::fromUtf8("bpp"));

        formLayout->setWidget(1, QFormLayout::FieldRole, bpp);

        label_11 = new QLabel(centralWidget);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label_11);

        bpp_pad = new QLineEdit(centralWidget);
        bpp_pad->setObjectName(QString::fromUtf8("bpp_pad"));

        formLayout->setWidget(2, QFormLayout::FieldRole, bpp_pad);

        label_9 = new QLabel(centralWidget);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        formLayout->setWidget(2, QFormLayout::LabelRole, label_9);


        horizontalLayout->addLayout(formLayout);

        verticalLayout_6 = new QVBoxLayout();
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));

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

        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setSizeConstraint(QLayout::SetFixedSize);
        crop_height = new QLineEdit(centralWidget);
        crop_height->setObjectName(QString::fromUtf8("crop_height"));

        gridLayout->addWidget(crop_height, 5, 2, 1, 1);

        crop_x = new QLineEdit(centralWidget);
        crop_x->setObjectName(QString::fromUtf8("crop_x"));

        gridLayout->addWidget(crop_x, 2, 2, 1, 1);

        label_14 = new QLabel(centralWidget);
        label_14->setObjectName(QString::fromUtf8("label_14"));

        gridLayout->addWidget(label_14, 4, 1, 1, 1);

        Crop = new QPushButton(centralWidget);
        Crop->setObjectName(QString::fromUtf8("Crop"));

        gridLayout->addWidget(Crop, 2, 0, 1, 1);

        verticalSpacer_2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout->addItem(verticalSpacer_2, 7, 0, 1, 1);

        crop_y = new QLineEdit(centralWidget);
        crop_y->setObjectName(QString::fromUtf8("crop_y"));

        gridLayout->addWidget(crop_y, 3, 2, 1, 1);

        equalize = new QPushButton(centralWidget);
        equalize->setObjectName(QString::fromUtf8("equalize"));

        gridLayout->addWidget(equalize, 0, 0, 1, 1);

        label_15 = new QLabel(centralWidget);
        label_15->setObjectName(QString::fromUtf8("label_15"));

        gridLayout->addWidget(label_15, 5, 1, 1, 1);

        label_12 = new QLabel(centralWidget);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        gridLayout->addWidget(label_12, 2, 1, 1, 1);

        crop_width = new QLineEdit(centralWidget);
        crop_width->setObjectName(QString::fromUtf8("crop_width"));

        gridLayout->addWidget(crop_width, 4, 2, 1, 1);

        label_13 = new QLabel(centralWidget);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        gridLayout->addWidget(label_13, 3, 1, 1, 1);

        line_13 = new QFrame(centralWidget);
        line_13->setObjectName(QString::fromUtf8("line_13"));
        line_13->setFrameShape(QFrame::HLine);
        line_13->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line_13, 1, 1, 1, 1);

        line_10 = new QFrame(centralWidget);
        line_10->setObjectName(QString::fromUtf8("line_10"));
        line_10->setFrameShape(QFrame::HLine);
        line_10->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line_10, 6, 1, 1, 1);

        line_11 = new QFrame(centralWidget);
        line_11->setObjectName(QString::fromUtf8("line_11"));
        line_11->setFrameShape(QFrame::HLine);
        line_11->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line_11, 6, 0, 1, 1);

        line_12 = new QFrame(centralWidget);
        line_12->setObjectName(QString::fromUtf8("line_12"));
        line_12->setFrameShape(QFrame::HLine);
        line_12->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line_12, 6, 2, 1, 1);

        line_14 = new QFrame(centralWidget);
        line_14->setObjectName(QString::fromUtf8("line_14"));
        line_14->setFrameShape(QFrame::HLine);
        line_14->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line_14, 1, 0, 1, 1);

        line_15 = new QFrame(centralWidget);
        line_15->setObjectName(QString::fromUtf8("line_15"));
        line_15->setFrameShape(QFrame::HLine);
        line_15->setFrameShadow(QFrame::Sunken);

        gridLayout->addWidget(line_15, 1, 2, 1, 1);


        horizontalLayout_5->addLayout(gridLayout);

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
        file_path->setText(QCoreApplication::translate("convert", "./sample/YVYU_1824x940.raw", nullptr));
        label_5->setText(QCoreApplication::translate("convert", "Source integrity", nullptr));
        Source_img_integrity->setText(QCoreApplication::translate("convert", "FAIL", nullptr));
        label_2->setText(QCoreApplication::translate("convert", "Width", nullptr));
        width->setText(QCoreApplication::translate("convert", "1824", nullptr));
        label_6->setText(QCoreApplication::translate("convert", "Frame stride", nullptr));
        frame_stride->setText(QCoreApplication::translate("convert", "0", nullptr));
        label_3->setText(QCoreApplication::translate("convert", "Height", nullptr));
        height->setText(QCoreApplication::translate("convert", "940", nullptr));
        label_7->setText(QCoreApplication::translate("convert", "Number of frames", nullptr));
        num_frames->setText(QCoreApplication::translate("convert", "0", nullptr));
        label_8->setText(QCoreApplication::translate("convert", "Filesize", nullptr));
        file_size->setText(QCoreApplication::translate("convert", "0", nullptr));
        label_10->setText(QCoreApplication::translate("convert", "Image counter", nullptr));
        label_4->setText(QCoreApplication::translate("convert", "Pixel format", nullptr));
        label_11->setText(QCoreApplication::translate("convert", "Bpp", nullptr));
        bpp_pad->setText(QCoreApplication::translate("convert", "0", nullptr));
        label_9->setText(QCoreApplication::translate("convert", "Bpp padded", nullptr));
        draw_window->setText(QString());
        label_14->setText(QCoreApplication::translate("convert", "Width", nullptr));
        Crop->setText(QCoreApplication::translate("convert", "Crop", nullptr));
        equalize->setText(QCoreApplication::translate("convert", "Equalize", nullptr));
        label_15->setText(QCoreApplication::translate("convert", "Height", nullptr));
        label_12->setText(QCoreApplication::translate("convert", "X", nullptr));
        label_13->setText(QCoreApplication::translate("convert", "Y", nullptr));
        modified_draw_window->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class convert: public Ui_convert {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONVERT_H
