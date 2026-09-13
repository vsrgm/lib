#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QMessageBox>
#include <QDateTime>
#include <QDir>
#include <QDebug>
#include <QUrlQuery>
#include <QNetworkInterface>
#include <QPixmap>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    settings = new QSettings("IOT_Home", "IoT_Home_Control_Center", this);
    networkManager = new QNetworkAccessManager(this);
    streamManager = new QNetworkAccessManager(this);

    udpSocket = new QUdpSocket(this);
    udpSocket->bind(8888, QUdpSocket::ShareAddress | QUdpSocket::ReuseAddressHint);
    connect(udpSocket, &QUdpSocket::readyRead, this, &MainWindow::processPendingUdpDatagrams);

    periodicTimer = new QTimer(this);
    connect(periodicTimer, &QTimer::timeout, this, &MainWindow::onPeriodicTimerTimeout);
    periodicTimer->start(5000);

    streamTimer = new QTimer(this);
    connect(streamTimer, &QTimer::timeout, this, &MainWindow::onStreamTimerTimeout);
    streamTimer->start(1500);

    connect(ui->rbMqtt, &QRadioButton::toggled, this, &MainWindow::on_syncModeChanged);
    connect(ui->rbLocalIp, &QRadioButton::toggled, this, &MainWindow::on_syncModeChanged);
    connect(ui->rbFirebase, &QRadioButton::toggled, this, &MainWindow::on_syncModeChanged);

    detectAppIp();
    loadSettings();
}

MainWindow::~MainWindow()
{
    saveSettings();
    delete ui;
}

void MainWindow::detectAppIp()
{
    QString detectedIp = "127.0.0.1";
    QString preferredIp = "";

    const QList<QNetworkInterface> interfaces = QNetworkInterface::allInterfaces();
    for (const QNetworkInterface &iface : interfaces) {
        if (!iface.flags().testFlag(QNetworkInterface::IsUp) ||
            iface.flags().testFlag(QNetworkInterface::IsLoopBack)) {
            continue;
        }

        QString nameLower = iface.humanReadableName().toLower();
        bool isVirtual = nameLower.contains("vmware") || nameLower.contains("virtual") ||
                         nameLower.contains("vbox") || nameLower.contains("hyper-v") ||
                         nameLower.contains("vethernet") || nameLower.contains("wsl");

        for (const QNetworkAddressEntry &entry : iface.addressEntries()) {
            if (entry.ip().protocol() == QAbstractSocket::IPv4Protocol) {
                QString ip = entry.ip().toString();
                if (detectedIp == "127.0.0.1") detectedIp = ip;
                if (!isVirtual && preferredIp.isEmpty()) {
                    preferredIp = ip;
                }
            }
        }
    }

    if (!preferredIp.isEmpty()) detectedIp = preferredIp;
    ui->lblAppIp->setText("App IP: " + detectedIp);
}

void MainWindow::loadSettings()
{
    syncMode = settings->value("sync_mode", 1).toInt();
    if (syncMode == 0) ui->rbMqtt->setChecked(true);
    else if (syncMode == 1) ui->rbLocalIp->setChecked(true);
    else ui->rbFirebase->setChecked(true);

    ui->txtIpStudy->setText(settings->value("ip_study", "192.168.1.10").toString());
    ui->txtIpBed->setText(settings->value("ip_bedroom", "192.168.1.11").toString());
    ui->txtIpKitchen->setText(settings->value("ip_kitchen", "192.168.1.12").toString());
    ui->txtIpExhaust->setText(settings->value("ip_kitchen_fan", "192.168.1.13").toString());
    ui->txtIpToilet->setText(settings->value("ip_toilet", "192.168.1.14").toString());
    ui->txtIpRo->setText(settings->value("ip_ro_pump", "192.168.1.15").toString());
    ui->txtIpDoor->setText(settings->value("ip_door", "192.168.1.16").toString());
    ui->txtIpPi->setText(settings->value("pi_kitchen_ip", "192.168.1.50").toString());

    ui->txtMqttBroker->setText(settings->value("mqtt_broker", "broker.hivemq.com").toString());
    ui->txtMqttPorts->setText(settings->value("mqtt_ports", "1883, 8000, 8883, 8884").toString());
    ui->txtFirebaseUrl->setText(settings->value("firebase_url", "https://gapsmarthome-default-rtdb.asia-southeast1.firebasedatabase.app").toString());

    nodeIps["study"] = ui->txtIpStudy->text();
    nodeIps["bedroom"] = ui->txtIpBed->text();
    nodeIps["kitchen"] = ui->txtIpKitchen->text();
    nodeIps["exhaust"] = ui->txtIpExhaust->text();
    nodeIps["toilet"] = ui->txtIpToilet->text();
    nodeIps["ro_pump"] = ui->txtIpRo->text();
    nodeIps["door"] = ui->txtIpDoor->text();
    nodeIps["pi_kitchen"] = ui->txtIpPi->text();

    ui->lblDashStudyIp->setText("IP: " + (ui->txtIpStudy->text().isEmpty() ? "Unmapped" : ui->txtIpStudy->text()));
    ui->lblDashBedIp->setText("IP: " + (ui->txtIpBed->text().isEmpty() ? "Unmapped" : ui->txtIpBed->text()));
    ui->lblDashKitchenIp->setText("IP: " + (ui->txtIpKitchen->text().isEmpty() ? "Unmapped" : ui->txtIpKitchen->text()));
    ui->lblDashExhaustIp->setText("IP: " + (ui->txtIpExhaust->text().isEmpty() ? "Unmapped" : ui->txtIpExhaust->text()));
    ui->lblDashToiletIp->setText("IP: " + (ui->txtIpToilet->text().isEmpty() ? "Unmapped" : ui->txtIpToilet->text()));
    ui->lblDashRoIp->setText("IP: " + (ui->txtIpRo->text().isEmpty() ? "Unmapped" : ui->txtIpRo->text()));
    ui->lblDashDoorIp->setText("IP: " + (ui->txtIpDoor->text().isEmpty() ? "Unmapped" : ui->txtIpDoor->text()));
    ui->lblDashPiIp->setText("IP: " + (ui->txtIpPi->text().isEmpty() ? "Unmapped" : ui->txtIpPi->text()));

    ui->lblSyncMode->setText(syncMode == 0 ? "Mode: MQTT" : (syncMode == 1 ? "Mode: Local Network (IP)" : "Mode: Firebase Cloud"));
}

void MainWindow::saveSettings()
{
    if (ui->rbMqtt->isChecked()) syncMode = 0;
    else if (ui->rbLocalIp->isChecked()) syncMode = 1;
    else syncMode = 2;

    settings->setValue("sync_mode", syncMode);
    settings->setValue("ip_study", ui->txtIpStudy->text().trimmed());
    settings->setValue("ip_bedroom", ui->txtIpBed->text().trimmed());
    settings->setValue("ip_kitchen", ui->txtIpKitchen->text().trimmed());
    settings->setValue("ip_kitchen_fan", ui->txtIpExhaust->text().trimmed());
    settings->setValue("ip_toilet", ui->txtIpToilet->text().trimmed());
    settings->setValue("ip_ro_pump", ui->txtIpRo->text().trimmed());
    settings->setValue("ip_door", ui->txtIpDoor->text().trimmed());
    settings->setValue("pi_kitchen_ip", ui->txtIpPi->text().trimmed());

    settings->setValue("mqtt_broker", ui->txtMqttBroker->text().trimmed());
    settings->setValue("mqtt_ports", ui->txtMqttPorts->text().trimmed());
    settings->setValue("firebase_url", ui->txtFirebaseUrl->text().trimmed());

    nodeIps["study"] = ui->txtIpStudy->text().trimmed();
    nodeIps["bedroom"] = ui->txtIpBed->text().trimmed();
    nodeIps["kitchen"] = ui->txtIpKitchen->text().trimmed();
    nodeIps["exhaust"] = ui->txtIpExhaust->text().trimmed();
    nodeIps["toilet"] = ui->txtIpToilet->text().trimmed();
    nodeIps["ro_pump"] = ui->txtIpRo->text().trimmed();
    nodeIps["door"] = ui->txtIpDoor->text().trimmed();
    nodeIps["pi_kitchen"] = ui->txtIpPi->text().trimmed();

    ui->lblDashStudyIp->setText("IP: " + (ui->txtIpStudy->text().isEmpty() ? "Unmapped" : ui->txtIpStudy->text()));
    ui->lblDashBedIp->setText("IP: " + (ui->txtIpBed->text().isEmpty() ? "Unmapped" : ui->txtIpBed->text()));
    ui->lblDashKitchenIp->setText("IP: " + (ui->txtIpKitchen->text().isEmpty() ? "Unmapped" : ui->txtIpKitchen->text()));
    ui->lblDashExhaustIp->setText("IP: " + (ui->txtIpExhaust->text().isEmpty() ? "Unmapped" : ui->txtIpExhaust->text()));
    ui->lblDashToiletIp->setText("IP: " + (ui->txtIpToilet->text().isEmpty() ? "Unmapped" : ui->txtIpToilet->text()));
    ui->lblDashRoIp->setText("IP: " + (ui->txtIpRo->text().isEmpty() ? "Unmapped" : ui->txtIpRo->text()));
    ui->lblDashDoorIp->setText("IP: " + (ui->txtIpDoor->text().isEmpty() ? "Unmapped" : ui->txtIpDoor->text()));
    ui->lblDashPiIp->setText("IP: " + (ui->txtIpPi->text().isEmpty() ? "Unmapped" : ui->txtIpPi->text()));

    ui->lblSyncMode->setText(syncMode == 0 ? "Mode: MQTT" : (syncMode == 1 ? "Mode: Local Network (IP)" : "Mode: Firebase Cloud"));
}

void MainWindow::on_syncModeChanged()
{
    saveSettings();
}

void MainWindow::on_btnSaveSettings_clicked()
{
    saveSettings();
    QMessageBox::information(this, "Settings Saved", "Settings saved successfully!");
}

void MainWindow::on_btnDiscover_clicked()
{
    logMessage("DISCOVERY", "Broadcasting UDP DISCOVER & running HTTP subnet probe...");
    QByteArray data = "DISCOVER";

    // 1. Send via main socket
    udpSocket->writeDatagram(data, QHostAddress::Broadcast, 8888);

    // 2. Force Windows socket layer to route broadcast out through each active network adapter
    const QList<QNetworkInterface> interfaces = QNetworkInterface::allInterfaces();
    QString localSubnetPrefix = "";

    for (const QNetworkInterface &iface : interfaces) {
        if (!iface.flags().testFlag(QNetworkInterface::IsUp) ||
            iface.flags().testFlag(QNetworkInterface::IsLoopBack)) {
            continue;
        }

        for (const QNetworkAddressEntry &entry : iface.addressEntries()) {
            if (entry.ip().protocol() == QAbstractSocket::IPv4Protocol) {
                QHostAddress localIp = entry.ip();
                QHostAddress bcast = entry.broadcast();

                QString ipStr = localIp.toString();
                if (!ipStr.startsWith("127.") && !ipStr.startsWith("169.254.") && !ipStr.contains(".230.")) {
                    int lastDot = ipStr.lastIndexOf('.');
                    if (lastDot != -1) {
                        localSubnetPrefix = ipStr.left(lastDot + 1);
                    }
                }

                QUdpSocket interfaceSocket;
                if (interfaceSocket.bind(localIp, 0, QUdpSocket::ShareAddress | QUdpSocket::ReuseAddressHint)) {
                    if (!bcast.isNull()) {
                        interfaceSocket.writeDatagram(data, bcast, 8888);
                    }
                    interfaceSocket.writeDatagram(data, QHostAddress::Broadcast, 8888);
                    logMessage("DISCOVERY", QString("Sent DISCOVER via %1 (%2 -> %3)").arg(iface.humanReadableName(), localIp.toString(), bcast.toString()));
                }
            }
        }
    }

    // 3. HTTP Probe Fallback: Fast HTTP scan on local subnet (e.g. 192.168.0.x)
    if (localSubnetPrefix.isEmpty()) {
        localSubnetPrefix = "192.168.0.";
    }

    logMessage("DISCOVERY", QString("Probing HTTP status across subnet %1x...").arg(localSubnetPrefix));

    // Probe common host IPs (100-150 first, then 1-99, 151-254)
    QList<int> targetHosts;
    for (int i = 100; i <= 150; ++i) targetHosts.append(i);
    for (int i = 1; i < 100; ++i) targetHosts.append(i);
    for (int i = 151; i <= 254; ++i) targetHosts.append(i);

    for (int host : targetHosts) {
        QString probeIp = localSubnetPrefix + QString::number(host);
        QUrl url(QString("http://%1/status").arg(probeIp));
        QNetworkRequest req(url);
        req.setTransferTimeout(1200);

        QNetworkReply *reply = networkManager->get(req);
        connect(reply, &QNetworkReply::finished, [this, reply, probeIp]() {
            if (reply->error() == QNetworkReply::NoError) {
                QByteArray data = reply->readAll();
                QJsonDocument doc = QJsonDocument::fromJson(data);
                if (!doc.isNull() && doc.isObject()) {
                    QJsonObject obj = doc.object();
                    QString devName = obj.value("name").toString();
                    QString devId = obj.value("id").toString();
                    if (devName.isEmpty()) devName = devId;

                    logMessage("DISCOVERY", QString("HTTP Probe found node at %1 (%2 / %3)").arg(probeIp, devName, devId));
                    mapDiscoveredDevice(devName + " " + devId + " door main_door", probeIp);
                }
            }
            reply->deleteLater();
        });
    }
}

void MainWindow::processPendingUdpDatagrams()
{
    while (udpSocket->hasPendingDatagrams()) {
        QByteArray datagram;
        datagram.resize(int(udpSocket->pendingDatagramSize()));
        QHostAddress senderIp;
        quint16 senderPort;

        udpSocket->readDatagram(datagram.data(), datagram.size(), &senderIp, &senderPort);
        QString response = QString::fromUtf8(datagram).trimmed();
        QString ipStr = senderIp.toString();
        if (ipStr.startsWith("::ffff:")) ipStr = ipStr.mid(7);

        logMessage("DISCOVERY", QString("Received from %1: %2").arg(ipStr, response));

        QString deviceName = response;
        QString deviceId = "";
        if (response.startsWith("{") && response.endsWith("}")) {
            QJsonDocument doc = QJsonDocument::fromJson(datagram);
            if (!doc.isNull() && doc.isObject()) {
                QJsonObject obj = doc.object();
                if (obj.contains("name")) deviceName = obj["name"].toString();
                if (obj.contains("id")) deviceId = obj["id"].toString();
                if (obj.contains("ip") && !obj["ip"].toString().isEmpty()) {
                    ipStr = obj["ip"].toString();
                }
            }
        }

        QString searchKey = (deviceName + " " + deviceId + " " + response).trimmed();
        mapDiscoveredDevice(searchKey, ipStr);
    }
}

void MainWindow::mapDiscoveredDevice(const QString &name, const QString &ip)
{
    QString idLower = name.toLower();
    if (idLower.contains("kitchen_fan") || idLower.contains("exhaust")) {
        ui->txtIpExhaust->setText(ip);
        ui->lblDashExhaustIp->setText("IP: " + ip);
        logMessage("MAPPED", "Mapped Exhaust Fan -> " + ip);
    } else if (idLower.contains("pi_kitchen") || idLower.contains("pikitchen")) {
        ui->txtIpPi->setText(ip);
        ui->lblDashPiIp->setText("IP: " + ip);
        logMessage("MAPPED", "Mapped Pi Kitchen -> " + ip);
    } else if (idLower.contains("kitchen") || idLower.contains("esp32_kitchen")) {
        ui->txtIpKitchen->setText(ip);
        ui->lblDashKitchenIp->setText("IP: " + ip);
        logMessage("MAPPED", "Mapped Kitchen -> " + ip);
    } else if (idLower.contains("study")) {
        ui->txtIpStudy->setText(ip);
        ui->lblDashStudyIp->setText("IP: " + ip);
        logMessage("MAPPED", "Mapped Study Room -> " + ip);
    } else if (idLower.contains("door") || idLower.contains("main_door") || idLower.contains("cam") || idLower.contains("securitymaindoor")) {
        ui->txtIpDoor->setText(ip);
        ui->lblDashDoorIp->setText("IP: " + ip);
        logMessage("MAPPED", "Mapped Main Door -> " + ip);
    } else if (idLower.contains("toilet")) {
        ui->txtIpToilet->setText(ip);
        ui->lblDashToiletIp->setText("IP: " + ip);
        logMessage("MAPPED", "Mapped Toilet -> " + ip);
    } else if (idLower.contains("ro_pump") || idLower.contains("ro") || idLower.contains("pump") || idLower.contains("water")) {
        ui->txtIpRo->setText(ip);
        ui->lblDashRoIp->setText("IP: " + ip);
        logMessage("MAPPED", "Mapped RO Pump -> " + ip);
    } else if (idLower.contains("bedroom")) {
        ui->txtIpBed->setText(ip);
        ui->lblDashBedIp->setText("IP: " + ip);
        logMessage("MAPPED", "Mapped Bed Room -> " + ip);
    }
    saveSettings();
}

QString MainWindow::getIpForNode(const QString &nodeKey)
{
    if (nodeKey == "study") return ui->txtIpStudy->text().trimmed();
    if (nodeKey == "bedroom") return ui->txtIpBed->text().trimmed();
    if (nodeKey == "kitchen") return ui->txtIpKitchen->text().trimmed();
    if (nodeKey == "exhaust") return ui->txtIpExhaust->text().trimmed();
    if (nodeKey == "toilet") return ui->txtIpToilet->text().trimmed();
    if (nodeKey == "ro_pump") return ui->txtIpRo->text().trimmed();
    if (nodeKey == "door") return ui->txtIpDoor->text().trimmed();
    if (nodeKey == "pi_kitchen") return ui->txtIpPi->text().trimmed();
    return "";
}

void MainWindow::fetchNodeStatus(const QString &nodeKey)
{
    QString ip = getIpForNode(nodeKey);
    if (ip.isEmpty()) return;

    QUrl url;
    if (syncMode == 1) { // Local IP
        url = QUrl(QString("http://%1/status").arg(ip));
    } else if (syncMode == 2) { // Firebase
        QString baseUrl = ui->txtFirebaseUrl->text().trimmed();
        url = QUrl(QString("%1/FrmNodeMcu/%2/status.json").arg(baseUrl, nodeKey));
    } else {
        return;
    }

    QNetworkRequest request(url);
    request.setTransferTimeout(3000);
    QNetworkReply *reply = networkManager->get(request);

    connect(reply, &QNetworkReply::finished, [this, reply, nodeKey]() {
        if (reply->error() == QNetworkReply::NoError) {
            QByteArray data = reply->readAll();
            QJsonDocument doc = QJsonDocument::fromJson(data);
            if (!doc.isNull() && doc.isObject()) {
                parseStatusJson(nodeKey, doc.object());
            }
        }
        reply->deleteLater();
    });
}

void MainWindow::sendNodeCommand(const QString &nodeKey, const QString &cmd)
{
    QString ip = getIpForNode(nodeKey);
    if (syncMode == 1 && !ip.isEmpty()) { // Local IP
        QUrl url;
        if (cmd.startsWith("CONFIG:")) {
            url = QUrl(QString("http://%1/config").arg(ip));
            QNetworkRequest request(url);
            request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
            QByteArray body = cmd.mid(7).toUtf8();
            networkManager->post(request, body);
        } else {
            url = QUrl(QString("http://%1/control?cmd=%2").arg(ip, cmd));
            QNetworkRequest request(url);
            networkManager->get(request);
        }
        logMessage(nodeKey, "Sent IP Cmd: " + cmd);
    } else if (syncMode == 2) { // Firebase
        QString baseUrl = ui->txtFirebaseUrl->text().trimmed();
        QUrl url(QString("%1/FrmMobile/%2/command.json").arg(baseUrl, nodeKey));
        QNetworkRequest request(url);
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
        QByteArray body = QString("\"%1\"").arg(cmd).toUtf8();
        networkManager->put(request, body);
        logMessage(nodeKey, "Sent Firebase Cmd: " + cmd);
    }
}

void MainWindow::updateCameraStream(const QString &nodeKey, QLabel *targetLabel)
{
    QString ip = getIpForNode(nodeKey);
    if (ip.isEmpty() || !targetLabel) return;

    QUrl url;
    if (syncMode == 1) { // Local IP
        url = QUrl(QString("http://%1/capture").arg(ip));
    } else if (syncMode == 2) { // Firebase Cloud
        QString baseUrl = ui->txtFirebaseUrl->text().trimmed();
        url = QUrl(QString("%1/FrmEsp32/%2/status.json").arg(baseUrl, nodeKey == "door" ? "door" : nodeKey));
    } else {
        return;
    }

    QNetworkRequest request(url);
    request.setTransferTimeout(2000);
    QNetworkReply *reply = streamManager->get(request);

    connect(reply, &QNetworkReply::finished, [this, reply, targetLabel, currentSyncMode = this->syncMode]() {
        if (reply->error() == QNetworkReply::NoError) {
            QByteArray data = reply->readAll();
            QPixmap pixmap;
            if (currentSyncMode == 2) { // Firebase
                QJsonDocument doc = QJsonDocument::fromJson(data);
                if (!doc.isNull() && doc.isObject()) {
                    QJsonObject obj = doc.object();
                    if (obj.contains("base64")) {
                        QByteArray imgBytes = QByteArray::fromBase64(obj["base64"].toString().toUtf8());
                        pixmap.loadFromData(imgBytes);
                    }
                }
            } else {
                pixmap.loadFromData(data);
            }

            if (!pixmap.isNull()) {
                targetLabel->setPixmap(pixmap.scaled(targetLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
            }
        }
        reply->deleteLater();
    });
}

void MainWindow::onPeriodicTimerTimeout()
{
    if (syncMode == 1 || syncMode == 2) {
        fetchNodeStatus("study");
        fetchNodeStatus("bedroom");
        fetchNodeStatus("kitchen");
        fetchNodeStatus("exhaust");
        fetchNodeStatus("toilet");
        fetchNodeStatus("ro_pump");
        fetchNodeStatus("door");
    }
}

void MainWindow::onStreamTimerTimeout()
{
    int currentTab = ui->tabWidget->currentIndex();
    if (currentTab == 7) { // Main Door tab
        updateCameraStream("door", ui->lblDoorStreamView);
    } else if (currentTab == 3) { // Kitchen tab
        updateCameraStream("kitchen", ui->lblKitchenCamStream);
    } else if (currentTab == 8) { // Pi Kitchen tab
        updateCameraStream("pi_kitchen", ui->lblPiStreamView);
    }
}

void MainWindow::on_btnSyncNodeConfig_clicked()
{
    QString target = ui->cmbOtaTargetNode->currentText();
    QString ip = getIpForNode(target);
    if (ip.isEmpty()) {
        QMessageBox::warning(this, "Node Error", "Target Node IP is empty!");
        return;
    }

    QJsonObject json;
    json["mqtt_broker"] = ui->txtMqttBroker->text().trimmed();
    json["mqtt_port"] = 1883;

    sendNodeCommand(target, "CONFIG:" + QJsonDocument(json).toJson(QJsonDocument::Compact));
    QMessageBox::information(this, "Config Synced", "Synced settings to node " + target);
}

void MainWindow::on_btnTriggerOta_clicked()
{
    QString fullUrl = ui->txtFullOtaUrl->text().trimmed();
    QString targetNode = ui->cmbOtaTargetNode->currentText();
    if (fullUrl.isEmpty()) {
        QMessageBox::warning(this, "OTA Error", "Full Firmware URL required!");
        return;
    }

    sendNodeCommand(targetNode, "OTA:" + fullUrl);
    QMessageBox::information(this, "OTA Sent", "OTA update trigger sent to " + targetNode);
}

void MainWindow::parseStatusJson(const QString &nodeKey, const QJsonObject &json)
{
    QString nowStr = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    ui->lblLastSync->setText("Last Synced: " + nowStr);

    if (nodeKey == "study") updateStudyRoomUi(json);
    else if (nodeKey == "bedroom") updateBedRoomUi(json);
    else if (nodeKey == "kitchen") updateKitchenUi(json);
    else if (nodeKey == "exhaust") updateExhaustFanUi(json);
    else if (nodeKey == "toilet") updateToiletUi(json);
    else if (nodeKey == "ro_pump") updateRoWaterUi(json);
    else if (nodeKey == "door") updateMainDoorUi(json);
    else if (nodeKey == "pi_kitchen") updatePiKitchenUi(json);
}

// ---------------- Study Room ----------------
void MainWindow::updateStudyRoomUi(const QJsonObject &json)
{
    double temp = json.value("temp").toDouble();
    double hum = json.value("hum").toDouble();
    int ldr = json.value("ldr").toInt();
    int heap = json.value("heap").toInt();
    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblStudyTemp->setText(QString::number(temp, 'f', 1) + " °C");
    ui->lblStudyHum->setText(QString::number(hum, 'f', 1) + " %");
    ui->lblStudyLdr->setText(QString::number(ldr));
    ui->lblStudyHeap->setText(QString::number(heap) + " bytes");
    ui->lblStudyNodeIp->setText(getIpForNode("study"));
    ui->lblStudyLastSync->setText(now);

    ui->lblDashStudyIp->setText("IP: " + getIpForNode("study"));
    ui->lblDashStudyStatus->setText(QString("Temp: %1 °C | Hum: %2 % | Light: %3")
                                    .arg(QString::number(temp, 'f', 1), QString::number(hum, 'f', 1), QString::number(ldr)));

    addHistoryRow("tblStudyHistory", now, "Telemetry", QString("Temp: %1, Hum: %2").arg(temp).arg(hum));
}

void MainWindow::on_btnStudySync_clicked() { fetchNodeStatus("study"); }
void MainWindow::on_btnStudyClearHistory_clicked() { ui->tblStudyHistory->setRowCount(0); }
void MainWindow::on_chkStudyOverride_toggled(bool checked) { sendNodeCommand("study", checked ? "OVERRIDE_ON" : "OVERRIDE_OFF"); }
void MainWindow::on_chkStudyLight_toggled(bool checked) { sendNodeCommand("study", checked ? "LIGHT_ON" : "LIGHT_OFF"); }
void MainWindow::on_chkStudyBuzzer_toggled(bool checked) { sendNodeCommand("study", checked ? "BUZZER_ON" : "BUZZER_OFF"); }
void MainWindow::on_chkStudyIrRecv_toggled(bool checked) { sendNodeCommand("study", checked ? "IR_RECV_ON" : "IR_RECV_OFF"); }
void MainWindow::on_sliderStudyBuzzerFreq_valueChanged(int value) {
    ui->lblStudyBuzzerFreqVal->setText(QString::number(value) + " Hz");
    sendNodeCommand("study", QString("FREQ:%1").arg(value));
}

// ---------------- Bed Room ----------------
void MainWindow::updateBedRoomUi(const QJsonObject &json)
{
    double temp = json.value("temp").toDouble();
    double hum = json.value("hum").toDouble();
    int gasA = json.value("gas_a").toInt();
    bool pir = json.value("pir").toBool();
    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblBedTemp->setText(QString::number(temp, 'f', 1) + " °C");
    ui->lblBedHum->setText(QString::number(hum, 'f', 1) + " %");
    ui->lblBedMq135Analog->setText(QString::number(gasA));
    ui->lblBedPir->setText(pir ? "ACTIVE" : "Clear");
    ui->lblBedNodeIp->setText(getIpForNode("bedroom"));
    ui->lblBedLastSync->setText(now);

    ui->lblDashBedIp->setText("IP: " + getIpForNode("bedroom"));
    ui->lblDashBedStatus->setText(QString("Temp: %1 °C | Gas: %2 | PIR: %3")
                                  .arg(QString::number(temp, 'f', 1)).arg(gasA).arg(pir ? "Motion" : "Clear"));

    addHistoryRow("tblBedHistory", now, "Telemetry", QString("Temp: %1, Gas: %2").arg(temp).arg(gasA));
}

void MainWindow::on_btnBedSync_clicked() { fetchNodeStatus("bedroom"); }
void MainWindow::on_btnBedClearHistory_clicked() { ui->tblBedHistory->setRowCount(0); }
void MainWindow::on_chkBedOverride_toggled(bool checked) { sendNodeCommand("bedroom", checked ? "OVERRIDE_ON" : "OVERRIDE_OFF"); }
void MainWindow::on_btnFanOn_clicked() { sendNodeCommand("bedroom", "IR_PRESET:FAN_ON"); }
void MainWindow::on_btnFanOff_clicked() { sendNodeCommand("bedroom", "IR_PRESET:FAN_OFF"); }
void MainWindow::on_btnFanSpeed1_clicked() { sendNodeCommand("bedroom", "IR_PRESET:FAN_SPD1"); }
void MainWindow::on_btnFanSpeed2_clicked() { sendNodeCommand("bedroom", "IR_PRESET:FAN_SPD2"); }
void MainWindow::on_btnFanSpeed3_clicked() { sendNodeCommand("bedroom", "IR_PRESET:FAN_SPD3"); }
void MainWindow::on_btnFanSpeed4_clicked() { sendNodeCommand("bedroom", "IR_PRESET:FAN_SPD4"); }
void MainWindow::on_btnFanSpeed5_clicked() { sendNodeCommand("bedroom", "IR_PRESET:FAN_SPD5"); }
void MainWindow::on_btnAcOn_clicked() { sendNodeCommand("bedroom", "IR_PRESET:AC_ON"); }
void MainWindow::on_btnAcOff_clicked() { sendNodeCommand("bedroom", "IR_PRESET:AC_OFF"); }
void MainWindow::on_btnAc26_clicked() { sendNodeCommand("bedroom", "IR_PRESET:AC_26C"); }
void MainWindow::on_btnSendCustomIr_clicked() {
    QString hex = ui->txtIrHex->text().trimmed();
    int bits = ui->spinIrBits->value();
    sendNodeCommand("bedroom", QString("IR_HEX:%1,%2").arg(hex).arg(bits));
}

// ---------------- Kitchen ----------------
void MainWindow::updateKitchenUi(const QJsonObject &json)
{
    bool pir = json.value("pir").toBool();
    bool gasD = json.value("gas_d").toBool();
    int gasA = json.value("gas_a").toInt();
    double temp1 = json.value("temp_lm358").toDouble();
    double temp2 = json.value("temp_bmp").toDouble();
    double pres = json.value("pres").toDouble();
    bool relay = json.value("relay").toBool();
    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblKitchenPir->setText(pir ? "Motion" : "Clear");
    ui->lblKitchenGasDigital->setText(gasD ? "ALERT!" : "NORMAL");
    ui->lblKitchenGasAnalog->setText(QString::number(gasA));
    ui->lblKitchenTempLm358->setText(QString::number(temp1, 'f', 1) + " °C");
    ui->lblKitchenTempBmp->setText(QString::number(temp2, 'f', 1) + " °C");
    ui->lblKitchenPres->setText(QString::number(pres, 'f', 1) + " hPa");
    ui->lblKitchenNodeIp->setText(getIpForNode("kitchen"));
    ui->lblKitchenLastSync->setText(now);

    ui->lblDashKitchenIp->setText("IP: " + getIpForNode("kitchen"));
    ui->lblDashKitchenStatus->setText(QString("Gas: %1 | Lamp: %2 | Temp: %3 °C")
                                      .arg(gasD ? "LEAK!" : "Normal").arg(relay ? "ON" : "OFF").arg(QString::number(temp1, 'f', 1)));
}

void MainWindow::on_btnKitchenSync_clicked() { fetchNodeStatus("kitchen"); }
void MainWindow::on_chkKitchenOverride_toggled(bool checked) { sendNodeCommand("kitchen", checked ? "OVERRIDE_ON" : "OVERRIDE_OFF"); }
void MainWindow::on_chkKitchenRelay_toggled(bool checked) { sendNodeCommand("kitchen", checked ? "RELAY_ON" : "RELAY_OFF"); }
void MainWindow::on_chkKitchenBuzzer_toggled(bool checked) { sendNodeCommand("kitchen", checked ? "BUZZER_ON" : "BUZZER_OFF"); }
void MainWindow::on_btnKitchenCapture_clicked() { sendNodeCommand("kitchen", "CAPTURE"); }

// ---------------- Exhaust Fan ----------------
void MainWindow::updateExhaustFanUi(const QJsonObject &json)
{
    int smokeA = json.value("smoke_a").toInt();
    bool smokeD = json.value("smoke_d").toBool();
    bool fan = json.value("fan").toBool();
    bool buzzer = json.value("buzzer").toBool();
    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblExhaustSmokeAnalog->setText(QString::number(smokeA));
    ui->lblExhaustSmokeDigital->setText(smokeD ? "ALERT!" : "NORMAL");
    ui->lblExhaustFanStatus->setText(fan ? "ON" : "OFF");
    ui->lblExhaustBuzzerStatus->setText(buzzer ? "ON" : "OFF");
    ui->lblExhaustNodeIp->setText(getIpForNode("exhaust"));
    ui->lblExhaustLastSync->setText(now);

    ui->lblDashExhaustIp->setText("IP: " + getIpForNode("exhaust"));
    ui->lblDashExhaustStatus->setText(QString("Smoke: %1 | Fan: %2").arg(smokeA).arg(fan ? "ON" : "OFF"));
    addHistoryRow("tblExhaustHistory", now, "Smoke Read", QString("Smoke: %1").arg(smokeA));
}

void MainWindow::on_btnExhaustSync_clicked() { fetchNodeStatus("exhaust"); }
void MainWindow::on_chkExhaustOverride_toggled(bool checked) { sendNodeCommand("exhaust", checked ? "OVERRIDE_ON" : "OVERRIDE_OFF"); }
void MainWindow::on_btnExhaustUpdateConfig_clicked() {
    QJsonObject obj;
    obj["smoke_th_on"] = ui->spinSmokeOnTh->value();
    obj["smoke_th_off"] = ui->spinSmokeOffTh->value();
    obj["buzzer_th"] = ui->spinBuzzerTh->value();
    sendNodeCommand("exhaust", "CONFIG:" + QJsonDocument(obj).toJson(QJsonDocument::Compact));
}

// ---------------- Toilet ----------------
void MainWindow::updateToiletUi(const QJsonObject &json)
{
    bool pir = json.value("pir").toBool();
    bool light = json.value("light").toBool();
    bool buzzer = json.value("buzzer").toBool();
    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblToiletPir->setText(pir ? "Motion" : "Clear");
    ui->lblToiletLight->setText(light ? "ON" : "OFF");
    ui->lblToiletBuzzer->setText(buzzer ? "ON" : "OFF");
    ui->lblToiletNodeIp->setText(getIpForNode("toilet"));
    ui->lblToiletLastSync->setText(now);

    ui->lblDashToiletIp->setText("IP: " + getIpForNode("toilet"));
    ui->lblDashToiletStatus->setText(QString("PIR: %1 | Light: %2").arg(pir ? "Motion" : "Clear").arg(light ? "ON" : "OFF"));
    addHistoryRow("tblToiletHistory", now, "Toilet Status", QString("PIR: %1").arg(pir ? 1 : 0));
}

void MainWindow::on_btnToiletSync_clicked() { fetchNodeStatus("toilet"); }
void MainWindow::on_chkToiletOverride_toggled(bool checked) { sendNodeCommand("toilet", checked ? "OVERRIDE_ON" : "OVERRIDE_OFF"); }
void MainWindow::on_chkToiletLight_toggled(bool checked) { sendNodeCommand("toilet", checked ? "LIGHT_ON" : "LIGHT_OFF"); }
void MainWindow::on_chkToiletBuzzer_toggled(bool checked) { sendNodeCommand("toilet", checked ? "BUZZER_ON" : "BUZZER_OFF"); }
void MainWindow::on_btnToiletPanic_clicked() { sendNodeCommand("toilet", "PANIC"); }

// ---------------- RO Water Pump ----------------
void MainWindow::updateRoWaterUi(const QJsonObject &json)
{
    bool level = json.value("level").toBool();
    bool pump = json.value("pump").toBool();
    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblRoLevel->setText(level ? "HIGH / FULL" : "NORMAL");
    ui->lblRoPumpStatus->setText(pump ? "ON" : "OFF");
    ui->lblRoNodeIp->setText(getIpForNode("ro_pump"));
    ui->lblRoLastSync->setText(now);

    ui->lblDashRoIp->setText("IP: " + getIpForNode("ro_pump"));
    ui->lblDashRoStatus->setText(QString("Water: %1 | Pump: %2").arg(level ? "FULL" : "Normal").arg(pump ? "ON" : "OFF"));
    addHistoryRow("tblRoHistory", now, "RO Pump", QString("Pump: %1").arg(pump ? "ON" : "OFF"));
}

void MainWindow::on_btnRoSync_clicked() { fetchNodeStatus("ro_pump"); }
void MainWindow::on_chkRoOverride_toggled(bool checked) { sendNodeCommand("ro_pump", checked ? "MANUAL_ON" : "MANUAL_OFF"); }
void MainWindow::on_chkRoPump_toggled(bool checked) { sendNodeCommand("ro_pump", checked ? "PUMP_ON" : "PUMP_OFF"); }

// ---------------- Main Door ----------------
void MainWindow::updateMainDoorUi(const QJsonObject &json)
{
    bool pir = json.value("pir").toBool();
    bool relay = json.value("relay").toBool();
    bool buzzer = json.value("buzzer").toBool();
    bool flash = json.value("flash").toBool();
    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblDoorPir->setText(pir ? "Motion" : "Clear");
    ui->lblDoorRelay->setText(relay ? "UNLOCKED" : "Locked");
    ui->lblDoorBuzzer->setText(buzzer ? "ON" : "OFF");
    ui->lblDoorFlash->setText(flash ? "ON" : "OFF");
    ui->lblDoorNodeIp->setText(getIpForNode("door"));
    ui->lblDoorLastSync->setText(now);

    ui->lblDashDoorIp->setText("IP: " + getIpForNode("door"));
    ui->lblDashDoorStatus->setText(QString("Door: %1 | Motion: %2").arg(relay ? "UNLOCKED" : "Locked").arg(pir ? "Motion" : "Clear"));
}

void MainWindow::on_btnDoorSync_clicked() { fetchNodeStatus("door"); }
void MainWindow::on_btnDoorUnlock_clicked() { sendNodeCommand("door", "RELAY_ON"); }
void MainWindow::on_btnDoorBuzzer_clicked() { sendNodeCommand("door", "BUZZER_ON"); }
void MainWindow::on_btnDoorFlash_clicked() { sendNodeCommand("door", "FLASH_ON"); }
void MainWindow::on_btnDoorCapture_clicked() { sendNodeCommand("door", "CAPTURE"); }
void MainWindow::on_btnDoorReboot_clicked() { sendNodeCommand("door", "REBOOT"); }

// ---------------- Pi Kitchen & RC Car ----------------
void MainWindow::updatePiKitchenUi(const QJsonObject &json)
{
    QString ver = json.value("version").toString();
    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");
    ui->lblPiVersion->setText(ver);
    ui->lblPiLastSync->setText(now);
    ui->lblDashPiIp->setText("IP: " + getIpForNode("pi_kitchen"));
    ui->lblDashPiStatus->setText("Version: " + ver);
}

void MainWindow::on_btnPiFetch_clicked() { fetchNodeStatus("pi_kitchen"); }
void MainWindow::on_btnRcForward_clicked() { sendNodeCommand("rccar", "FORWARD"); }
void MainWindow::on_btnRcBackward_clicked() { sendNodeCommand("rccar", "BACKWARD"); }
void MainWindow::on_btnRcLeft_clicked() { sendNodeCommand("rccar", "LEFT"); }
void MainWindow::on_btnRcRight_clicked() { sendNodeCommand("rccar", "RIGHT"); }
void MainWindow::on_btnRcStop_clicked() { sendNodeCommand("rccar", "STOP"); }

// Helper Logging & Tables
void MainWindow::logMessage(const QString &tag, const QString &msg)
{
    QString logLine = QString("[%1] [%2] %3")
                          .arg(QDateTime::currentDateTime().toString("HH:mm:ss"), tag, msg);

    ui->txtStudyLog->append(logLine);
    ui->txtBedLog->append(logLine);
    ui->txtKitchenLog->append(logLine);
    ui->txtExhaustLog->append(logLine);
    ui->txtToiletLog->append(logLine);
    ui->txtRoLog->append(logLine);
    ui->txtDoorLog->append(logLine);
    ui->txtPiLog->append(logLine);
    ui->txtRcLog->append(logLine);
}

void MainWindow::addHistoryRow(const QString &tableWidgetName, const QString &time, const QString &event, const QString &details)
{
    QTableWidget *table = findChild<QTableWidget*>(tableWidgetName);
    if (!table) return;

    table->insertRow(0);
    table->setItem(0, 0, new QTableWidgetItem(time));
    table->setItem(0, 1, new QTableWidgetItem(event));
    table->setItem(0, 2, new QTableWidgetItem(details));
}
