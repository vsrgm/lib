#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QUdpSocket>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QTimer>
#include <QSettings>
#include <QJsonObject>
#include <QJsonDocument>
#include <QMap>
#include <QList>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // Navigation & Sync Mode
    void on_syncModeChanged();
    void on_btnDiscover_clicked();
    void on_btnSaveSettings_clicked();
    void on_btnSyncNodeConfig_clicked();
    void on_btnTriggerOta_clicked();

    // Study Room Slots
    void on_btnStudySync_clicked();
    void on_btnStudyClearHistory_clicked();
    void on_chkStudyOverride_toggled(bool checked);
    void on_chkStudyLight_toggled(bool checked);
    void on_chkStudyBuzzer_toggled(bool checked);
    void on_chkStudyIrRecv_toggled(bool checked);
    void on_sliderStudyBuzzerFreq_valueChanged(int value);

    // Bedroom Slots
    void on_btnBedSync_clicked();
    void on_btnBedClearHistory_clicked();
    void on_chkBedOverride_toggled(bool checked);
    void on_btnFanOn_clicked();
    void on_btnFanOff_clicked();
    void on_btnFanSpeed1_clicked();
    void on_btnFanSpeed2_clicked();
    void on_btnFanSpeed3_clicked();
    void on_btnFanSpeed4_clicked();
    void on_btnFanSpeed5_clicked();
    void on_btnAcOn_clicked();
    void on_btnAcOff_clicked();
    void on_btnAc26_clicked();
    void on_btnSendCustomIr_clicked();

    // Kitchen Slots
    void on_btnKitchenSync_clicked();
    void on_chkKitchenOverride_toggled(bool checked);
    void on_chkKitchenRelay_toggled(bool checked);
    void on_chkKitchenBuzzer_toggled(bool checked);
    void on_btnKitchenCapture_clicked();

    // Exhaust Fan Slots
    void on_btnExhaustSync_clicked();
    void on_chkExhaustOverride_toggled(bool checked);
    void on_btnExhaustUpdateConfig_clicked();

    // Toilet Slots
    void on_btnToiletSync_clicked();
    void on_chkToiletOverride_toggled(bool checked);
    void on_chkToiletLight_toggled(bool checked);
    void on_chkToiletBuzzer_toggled(bool checked);
    void on_btnToiletPanic_clicked();

    // RO Water Slots
    void on_btnRoSync_clicked();
    void on_chkRoOverride_toggled(bool checked);
    void on_chkRoPump_toggled(bool checked);

    // Main Door Slots
    void on_btnDoorSync_clicked();
    void on_btnDoorUnlock_clicked();
    void on_btnDoorBuzzer_clicked();
    void on_btnDoorFlash_clicked();
    void on_btnDoorCapture_clicked();
    void on_btnDoorReboot_clicked();

    // Pi Kitchen & RC Car
    void on_btnPiFetch_clicked();
    void on_btnRcForward_clicked();
    void on_btnRcBackward_clicked();
    void on_btnRcLeft_clicked();
    void on_btnRcRight_clicked();
    void on_btnRcStop_clicked();

    // Networking Callbacks
    void processPendingUdpDatagrams();
    void onPeriodicTimerTimeout();
    void onStreamTimerTimeout();

private:
    Ui::MainWindow *ui;

    QUdpSocket *udpSocket;
    QNetworkAccessManager *networkManager;
    QNetworkAccessManager *streamManager;
    QTimer *periodicTimer;
    QTimer *streamTimer;
    QSettings *settings;

    int syncMode; // 0: MQTT, 1: Local IP, 2: Firebase
    QMap<QString, QString> nodeIps;
    QMap<QString, QString> discoveredNodes;

    void loadSettings();
    void saveSettings();
    void detectAppIp();
    void updateCameraStream(const QString &nodeKey, class QLabel *targetLabel);
    void mapDiscoveredDevice(const QString &name, const QString &ip);
    QString getIpForNode(const QString &nodeKey);
    void sendNodeCommand(const QString &nodeKey, const QString &cmd);
    void fetchNodeStatus(const QString &nodeKey);
    void logMessage(const QString &tag, const QString &msg);
    void addHistoryRow(const QString &tableWidgetName, const QString &time, const QString &event, const QString &details);

    void parseStatusJson(const QString &nodeKey, const QJsonObject &json);
    void updateStudyRoomUi(const QJsonObject &json);
    void updateBedRoomUi(const QJsonObject &json);
    void updateKitchenUi(const QJsonObject &json);
    void updateExhaustFanUi(const QJsonObject &json);
    void updateToiletUi(const QJsonObject &json);
    void updateRoWaterUi(const QJsonObject &json);
    void updateMainDoorUi(const QJsonObject &json);
    void updatePiKitchenUi(const QJsonObject &json);
};

#endif // MAINWINDOW_H
