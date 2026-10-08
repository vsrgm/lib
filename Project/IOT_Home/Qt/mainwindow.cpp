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
    streamTimer->start(500);

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
    } else if (idLower.contains("rccar") || idLower.contains("rc_car") || idLower.contains("car")) {
        discoveredNodes["rccar"] = ip;
        logMessage("MAPPED", "Mapped RC Car -> " + ip);
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
    if (nodeKey == "rccar") return discoveredNodes.value("rccar", "");
    return "";
}

QString MainWindow::getFirebaseStatusPath(const QString &nodeKey)
{
    if (nodeKey == "kitchen") return "FrmEsp32/kitchen";
    if (nodeKey == "exhaust") return "FrmNodeMcu/kitchen_fan";
    if (nodeKey == "door") return "FrmEsp32/Securitymaindoor";
    if (nodeKey == "toilet") return "FrmEsp32/toilet";
    if (nodeKey == "study") return "FrmNodeMcu/study";
    if (nodeKey == "bedroom") return "FrmNodeMcu/bedroom";
    if (nodeKey == "ro_pump") return "FrmNodeMcu/ro_pump";
    if (nodeKey == "pi_kitchen") return "FrmMobile/pi_kitchen";
    if (nodeKey == "rccar") return "FrmEsp32/rccar";
    return "FrmNodeMcu/" + nodeKey;
}

QString MainWindow::getFirebaseCommandPath(const QString &nodeKey)
{
    if (nodeKey == "kitchen") return "FrmMobile/kitchen";
    if (nodeKey == "exhaust") return "FrmMobile/kitchen_fan";
    if (nodeKey == "door") return "FrmMobile/main_door";
    if (nodeKey == "toilet") return "FrmMobile/toilet";
    if (nodeKey == "study") return "FrmMobile/study";
    if (nodeKey == "bedroom") return "FrmMobile/bedroom";
    if (nodeKey == "ro_pump") return "FrmMobile/ro_pump";
    if (nodeKey == "pi_kitchen") return "FrmMobile/pi_kitchen";
    if (nodeKey == "rccar") return "FrmEsp32/rccar";
    return "FrmMobile/" + nodeKey;
}

void MainWindow::fetchNodeStatus(const QString &nodeKey)
{
    QUrl url;
    if (syncMode == 1) { // Local IP
        QString ip = getIpForNode(nodeKey);
        if (ip.isEmpty()) return;
        url = QUrl(QString("http://%1/status").arg(ip));
    } else if (syncMode == 2) { // Firebase
        QString baseUrl = ui->txtFirebaseUrl->text().trimmed();
        if (baseUrl.endsWith("/")) baseUrl.chop(1);
        if (baseUrl.isEmpty()) return;
        QString secret = settings->value("firebase_secret", FIREBASE_SECRET).toString().trimmed();
        QString path = getFirebaseStatusPath(nodeKey);
        url = QUrl(QString("%1/%2/status.json?auth=%3").arg(baseUrl, path, secret));
    } else {
        return;
    }

    QNetworkRequest request(url);
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
    request.setTransferTimeout(3000);
    QNetworkReply *reply = networkManager->get(request);

    connect(reply, &QNetworkReply::finished, [this, reply, nodeKey]() {
        if (reply->error() == QNetworkReply::NoError) {
            QByteArray data = reply->readAll().trimmed();
            if (data.startsWith('"') && data.endsWith('"')) {
                QString strData = QString::fromUtf8(data);
                strData = strData.mid(1, strData.length() - 2);
                strData.replace("\\\"", "\"");
                strData.replace("\\\\", "\\");
                data = strData.toUtf8();
            }
            QJsonDocument doc = QJsonDocument::fromJson(data);
            if (!doc.isNull() && doc.isObject()) {
                parseStatusJson(nodeKey, doc.object());
            } else {
                logMessage(nodeKey, "Invalid JSON received: " + QString::fromUtf8(data));
            }
        } else {
            logMessage(nodeKey, "Fetch status error: " + reply->errorString());
        }
        reply->deleteLater();
    });
}

void MainWindow::sendNodeCommand(const QString &nodeKey, const QString &cmd)
{
    if (syncMode == 1) { // Local IP
        QString ip = getIpForNode(nodeKey);
        if (ip.isEmpty()) return;
        QUrl url;
        if (cmd.startsWith("CONFIG:")) {
            url = QUrl(QString("http://%1/config").arg(ip));
            QNetworkRequest request(url);
            request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
            request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
            QByteArray body = cmd.mid(7).toUtf8();
            networkManager->post(request, body);
        } else {
            url = QUrl(QString("http://%1/control?cmd=%2").arg(ip, cmd));
            QNetworkRequest request(url);
            request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
            networkManager->get(request);
        }
        logMessage(nodeKey, "Sent IP Cmd: " + cmd);
    } else if (syncMode == 2) { // Firebase
        QString baseUrl = ui->txtFirebaseUrl->text().trimmed();
        if (baseUrl.endsWith("/")) baseUrl.chop(1);
        if (baseUrl.isEmpty()) return;
        QString secret = settings->value("firebase_secret", "1q2w3e4r%T").toString().trimmed();
        QString path = getFirebaseCommandPath(nodeKey);
        QUrl url(QString("%1/%2/command.json?auth=%3").arg(baseUrl, path, secret));
        QNetworkRequest request(url);
        request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
        request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
        QByteArray body = QString("\"%1\"").arg(cmd).toUtf8();
        networkManager->put(request, body);
        logMessage(nodeKey, "Sent Firebase Cmd: " + cmd);
    }
}

void MainWindow::updateCameraStream(const QString &nodeKey, QLabel *targetLabel)
{
    QString ip = getIpForNode(nodeKey);
    if (!targetLabel) return;

    QUrl url;
    if (syncMode == 2) { // Firebase Cloud
        QString baseUrl = ui->txtFirebaseUrl->text().trimmed();
        if (baseUrl.endsWith("/")) baseUrl.chop(1);
        QString secret = settings->value("firebase_secret", "1q2w3e4r%T").toString().trimmed();
        if (nodeKey == "door") {
            url = QUrl(QString("%1/FrmEsp32/Securitymaindoor/last_image.json?auth=%2").arg(baseUrl, secret));
        } else {
            url = QUrl(QString("%1/FrmEsp32/%2/status.json?auth=%3").arg(baseUrl, nodeKey, secret));
        }
    } else { // Local IP Mode (or fallback when IP is present)
        if (ip.isEmpty()) return;
        url = QUrl(QString("http://%1/capture").arg(ip));
    }

    QNetworkRequest request(url);
    request.setTransferTimeout(2000);
    QNetworkReply *reply = streamManager->get(request);

    connect(reply, &QNetworkReply::finished, [this, reply, targetLabel, currentSyncMode = this->syncMode]() {
        if (reply->error() == QNetworkReply::NoError) {
            QByteArray data = reply->readAll();
            QPixmap pixmap;
            if (currentSyncMode == 2) { // Firebase
                QString strData = QString::fromUtf8(data).trimmed();
                if (strData.startsWith("\"")) strData = strData.mid(1);
                if (strData.endsWith("\"")) strData = strData.left(strData.length() - 1);

                QByteArray imgBytes = QByteArray::fromBase64(strData.toUtf8());
                pixmap.loadFromData(imgBytes);

                if (pixmap.isNull()) {
                    QJsonDocument doc = QJsonDocument::fromJson(data);
                    if (!doc.isNull() && doc.isObject()) {
                        QJsonObject obj = doc.object();
                        if (obj.contains("last_image")) {
                            pixmap.loadFromData(QByteArray::fromBase64(obj["last_image"].toString().toUtf8()));
                        } else if (obj.contains("base64")) {
                            pixmap.loadFromData(QByteArray::fromBase64(obj["base64"].toString().toUtf8()));
                        }
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
        // Send periodic heartbeat to keep NodeMCU/ESP32 streaming and active
        sendNodeCommand("exhaust", "HEARTBEAT");
        sendNodeCommand("kitchen", "HEARTBEAT");
        sendNodeCommand("study", "HEARTBEAT");
        sendNodeCommand("bedroom", "HEARTBEAT");
        sendNodeCommand("toilet", "HEARTBEAT");
        sendNodeCommand("ro_pump", "HEARTBEAT");
        sendNodeCommand("door", "HEARTBEAT");

        // Fetch latest status
        fetchNodeStatus("study");
        fetchNodeStatus("bedroom");
        fetchNodeStatus("kitchen");
        fetchNodeStatus("exhaust");
        fetchNodeStatus("toilet");
        fetchNodeStatus("ro_pump");
        fetchNodeStatus("door");
        fetchNodeStatus("pi_kitchen");
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

    if (json.contains("ip")) {
        QString discoveredIp = json.value("ip").toString().trimmed();
        if (!discoveredIp.isEmpty() && discoveredIp != "0.0.0.0") {
            QString searchKey = nodeKey;
            if (nodeKey == "exhaust") searchKey = "kitchen_fan exhaust";
            else if (nodeKey == "kitchen") searchKey = "kitchen";
            mapDiscoveredDevice(searchKey + " " + json.value("id").toString() + " " + json.value("name").toString(), discoveredIp);
        }
    }

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
    bool light = json.value("light").toBool();
    bool buzzer = json.value("buzzer").toBool();
    bool overrideState = json.value("override").toBool();
    bool irRecv = json.value("ir_recv").toBool();

    QString ip = json.value("ip").toString();
    if (ip.isEmpty()) ip = getIpForNode("study");

    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblStudyTemp->setText(QString::number(temp, 'f', 1) + " °C");
    ui->lblStudyHum->setText(QString::number(hum, 'f', 1) + " %");
    ui->lblStudyLdr->setText(QString::number(ldr));
    ui->lblStudyHeap->setText(QString::number(heap) + " bytes");
    ui->lblStudyNodeIp->setText(ip);
    ui->lblStudyLastSync->setText(now);

    ui->chkStudyOverride->blockSignals(true);
    ui->chkStudyOverride->setChecked(overrideState);
    ui->chkStudyOverride->blockSignals(false);

    ui->chkStudyLight->blockSignals(true);
    ui->chkStudyLight->setChecked(light);
    ui->chkStudyLight->blockSignals(false);

    ui->chkStudyBuzzer->blockSignals(true);
    ui->chkStudyBuzzer->setChecked(buzzer);
    ui->chkStudyBuzzer->blockSignals(false);

    ui->chkStudyIrRecv->blockSignals(true);
    ui->chkStudyIrRecv->setChecked(irRecv);
    ui->chkStudyIrRecv->blockSignals(false);

    ui->lblDashStudyIp->setText("IP: " + ip);
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
    if (!json.contains("gas_a") && json.contains("mq135")) gasA = json.value("mq135").toInt();

    bool pir = false;
    if (json.contains("pir")) {
        if (json.value("pir").isBool()) pir = json.value("pir").toBool();
        else pir = (json.value("pir").toInt() == 1) || (json.value("pir").toString() == "ACTIVE") || (json.value("pir").toString() == "Motion");
    }

    bool overrideState = false;
    if (json.contains("override")) {
        if (json.value("override").isBool()) overrideState = json.value("override").toBool();
        else overrideState = (json.value("override").toString() == "ON") || (json.value("override").toInt() == 1);
    }

    QString ip = json.value("ip").toString();
    if (ip.isEmpty()) ip = getIpForNode("bedroom");

    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblBedTemp->setText(QString::number(temp, 'f', 1) + " °C");
    ui->lblBedHum->setText(QString::number(hum, 'f', 1) + " %");
    ui->lblBedMq135Analog->setText(QString::number(gasA));
    ui->lblBedPir->setText(pir ? "ACTIVE" : "Clear");
    ui->lblBedNodeIp->setText(ip);
    ui->lblBedLastSync->setText(now);

    ui->chkBedOverride->blockSignals(true);
    ui->chkBedOverride->setChecked(overrideState);
    ui->chkBedOverride->blockSignals(false);

    ui->lblDashBedIp->setText("IP: " + ip);
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
    bool pir = false;
    if (json.contains("pir")) {
        if (json.value("pir").isBool()) pir = json.value("pir").toBool();
        else pir = (json.value("pir").toInt() == 1) || (json.value("pir").toString() == "1") || (json.value("pir").toString() == "Motion");
    }

    bool gasD = false;
    if (json.contains("gas_d")) {
        if (json.value("gas_d").isBool()) gasD = json.value("gas_d").toBool();
        else gasD = (json.value("gas_d").toInt() == 1) || (json.value("gas_d").toString() == "ALERT!");
    }

    int gasA = json.value("gas_a").toInt();
    if (!json.contains("gas_a") && json.contains("gas_analog")) gasA = json.value("gas_analog").toInt();

    double temp1 = json.value("temp_lm358").toDouble();
    if (!json.contains("temp_lm358") && json.contains("temp")) temp1 = json.value("temp").toDouble();

    double temp2 = json.value("temp_bmp").toDouble();
    double pres = json.value("pres").toDouble();

    bool relay = false;
    if (json.contains("relay")) {
        if (json.value("relay").isBool()) relay = json.value("relay").toBool();
        else relay = (json.value("relay").toInt() == 1) || (json.value("relay").toString() == "ON");
    }

    bool buzzer = false;
    if (json.contains("buzzer")) {
        if (json.value("buzzer").isBool()) buzzer = json.value("buzzer").toBool();
        else buzzer = (json.value("buzzer").toInt() == 1) || (json.value("buzzer").toString() == "ON");
    }

    bool overrideState = false;
    if (json.contains("manual_override")) {
        if (json.value("manual_override").isBool()) overrideState = json.value("manual_override").toBool();
        else overrideState = (json.value("manual_override").toString() == "ON") || (json.value("manual_override").toInt() == 1);
    } else if (json.contains("override")) {
        overrideState = json.value("override").toBool();
    }

    QString ip = json.value("ip").toString();
    if (ip.isEmpty()) ip = getIpForNode("kitchen");

    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblKitchenPir->setText(pir ? "Motion" : "Clear");
    ui->lblKitchenGasDigital->setText(gasD ? "ALERT!" : "NORMAL");
    ui->lblKitchenGasAnalog->setText(QString::number(gasA));
    ui->lblKitchenTempLm358->setText(QString::number(temp1, 'f', 1) + " °C");
    ui->lblKitchenTempBmp->setText(QString::number(temp2, 'f', 1) + " °C");
    ui->lblKitchenPres->setText(QString::number(pres, 'f', 1) + " hPa");
    ui->lblKitchenNodeIp->setText(ip);
    ui->lblKitchenLastSync->setText(now);

    ui->chkKitchenOverride->blockSignals(true);
    ui->chkKitchenOverride->setChecked(overrideState);
    ui->chkKitchenOverride->blockSignals(false);

    ui->chkKitchenRelay->blockSignals(true);
    ui->chkKitchenRelay->setChecked(relay);
    ui->chkKitchenRelay->blockSignals(false);

    ui->chkKitchenBuzzer->blockSignals(true);
    ui->chkKitchenBuzzer->setChecked(buzzer);
    ui->chkKitchenBuzzer->blockSignals(false);

    ui->lblDashKitchenIp->setText("IP: " + ip);
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
    int smokeA = 0;
    if (json.contains("mq2_a")) smokeA = json.value("mq2_a").toInt();
    else if (json.contains("smoke_a")) smokeA = json.value("smoke_a").toInt();

    QString smokeD = "NORMAL";
    if (json.contains("mq2_d")) {
        smokeD = json.value("mq2_d").isString() ? json.value("mq2_d").toString() : (json.value("mq2_d").toBool() ? "SMOKE" : "CLEAR");
    } else if (json.contains("smoke_d")) {
        smokeD = json.value("smoke_d").toBool() ? "ALERT!" : "NORMAL";
    }

    QString fanStatus = "OFF";
    bool fanOn = false;
    if (json.contains("fan")) {
        if (json.value("fan").isString()) {
            fanStatus = json.value("fan").toString();
            fanOn = (fanStatus == "ON");
        } else {
            fanOn = json.value("fan").toBool();
            fanStatus = fanOn ? "ON" : "OFF";
        }
    }

    QString buzzerStatus = "OFF";
    bool buzzerOn = false;
    if (json.contains("buzzer")) {
        if (json.value("buzzer").isString()) {
            buzzerStatus = json.value("buzzer").toString();
            buzzerOn = (buzzerStatus == "ON");
        } else {
            buzzerOn = json.value("buzzer").toBool();
            buzzerStatus = buzzerOn ? "ON" : "OFF";
        }
    }

    bool overrideState = false;
    if (json.contains("manual")) {
        if (json.value("manual").isString()) {
            overrideState = (json.value("manual").toString() == "ON");
        } else {
            overrideState = json.value("manual").toBool();
        }
    } else if (json.contains("override")) {
        overrideState = json.value("override").toBool();
    }

    QString ldrVal = json.value("ldr").toString();
    if (ldrVal.isEmpty() && json.contains("ldr")) ldrVal = QString::number(json.value("ldr").toInt());
    if (ldrVal.isEmpty()) ldrVal = "LIGHT";

    QString userSwVal = json.value("user_sw").toString();
    if (userSwVal.isEmpty()) userSwVal = "State 2";

    QString ip = json.value("ip").toString();
    if (ip.isEmpty()) ip = getIpForNode("exhaust");

    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblExhaustSmokeAnalog->setText(QString::number(smokeA));
    ui->lblExhaustSmokeDigital->setText(smokeD);
    ui->lblExhaustFanStatus->setText(fanStatus);
    ui->lblExhaustBuzzerStatus->setText(buzzerStatus);
    ui->lblExhaustLdr->setText(ldrVal);
    ui->lblExhaustUserSw->setText(userSwVal);
    ui->lblExhaustOverrideVal->setText(overrideState ? "ON" : "OFF");
    ui->lblExhaustNodeIp->setText(ip);
    ui->lblExhaustLastSync->setText(now);

    ui->chkExhaustFan->blockSignals(true);
    ui->chkExhaustFan->setChecked(fanOn);
    ui->chkExhaustFan->blockSignals(false);

    ui->chkExhaustBuzzer->blockSignals(true);
    ui->chkExhaustBuzzer->setChecked(buzzerOn);
    ui->chkExhaustBuzzer->blockSignals(false);

    ui->chkExhaustOverride->blockSignals(true);
    ui->chkExhaustOverride->setChecked(overrideState);
    ui->chkExhaustOverride->blockSignals(false);

    ui->lblDashExhaustIp->setText("IP: " + ip);
    ui->lblDashExhaustStatus->setText(QString("Smoke: %1 | Fan: %2").arg(smokeA).arg(fanStatus));
    addHistoryRow("tblExhaustHistory", now, "Smoke Read", QString("Smoke: %1").arg(smokeA));
}

void MainWindow::on_btnExhaustSync_clicked() { fetchNodeStatus("exhaust"); }
void MainWindow::on_chkExhaustOverride_toggled(bool checked) { sendNodeCommand("exhaust", checked ? "OVERRIDE_ON" : "OVERRIDE_OFF"); }
void MainWindow::on_chkExhaustFan_toggled(bool checked) { sendNodeCommand("exhaust", checked ? "RELAY_ON" : "RELAY_OFF"); }
void MainWindow::on_chkExhaustBuzzer_toggled(bool checked) { sendNodeCommand("exhaust", checked ? "BUZZER_ON" : "BUZZER_OFF"); }
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
    bool pir = json.value("pir").toBool() || (json.value("pir").toInt() == 1) || (json.value("pir").toString() == "Motion");
    bool light = json.value("light").toBool() || (json.value("light").toInt() == 1) || (json.value("light").toString() == "ON");
    bool buzzer = json.value("buzzer").toBool() || (json.value("buzzer").toInt() == 1) || (json.value("buzzer").toString() == "ON");
    bool overrideState = json.value("override").toBool() || (json.value("override").toInt() == 1) || (json.value("override").toString() == "ON");

    QString ip = json.value("ip").toString();
    if (ip.isEmpty()) ip = getIpForNode("toilet");

    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblToiletPir->setText(pir ? "Motion" : "Clear");
    ui->lblToiletLight->setText(light ? "ON" : "OFF");
    ui->lblToiletBuzzer->setText(buzzer ? "ON" : "OFF");
    ui->lblToiletNodeIp->setText(ip);
    ui->lblToiletLastSync->setText(now);

    ui->chkToiletOverride->blockSignals(true);
    ui->chkToiletOverride->setChecked(overrideState);
    ui->chkToiletOverride->blockSignals(false);

    ui->chkToiletLight->blockSignals(true);
    ui->chkToiletLight->setChecked(light);
    ui->chkToiletLight->blockSignals(false);

    ui->chkToiletBuzzer->blockSignals(true);
    ui->chkToiletBuzzer->setChecked(buzzer);
    ui->chkToiletBuzzer->blockSignals(false);

    ui->lblDashToiletIp->setText("IP: " + ip);
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
    bool level = json.value("level").toBool() || (json.value("level").toInt() == 1) || (json.value("level").toString() == "FULL") || (json.value("level").toString() == "HIGH");
    bool pump = json.value("pump").toBool() || (json.value("pump").toInt() == 1) || (json.value("pump").toString() == "ON");
    bool overrideState = json.value("override").toBool() || json.value("manual").toBool() || (json.value("manual").toString() == "ON");

    QString ip = json.value("ip").toString();
    if (ip.isEmpty()) ip = getIpForNode("ro_pump");

    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblRoLevel->setText(level ? "HIGH / FULL" : "NORMAL");
    ui->lblRoPumpStatus->setText(pump ? "ON" : "OFF");
    ui->lblRoNodeIp->setText(ip);
    ui->lblRoLastSync->setText(now);

    ui->chkRoOverride->blockSignals(true);
    ui->chkRoOverride->setChecked(overrideState);
    ui->chkRoOverride->blockSignals(false);

    ui->chkRoPump->blockSignals(true);
    ui->chkRoPump->setChecked(pump);
    ui->chkRoPump->blockSignals(false);

    ui->lblDashRoIp->setText("IP: " + ip);
    ui->lblDashRoStatus->setText(QString("Water: %1 | Pump: %2").arg(level ? "FULL" : "Normal").arg(pump ? "ON" : "OFF"));
    addHistoryRow("tblRoHistory", now, "RO Pump", QString("Pump: %1").arg(pump ? "ON" : "OFF"));
}

void MainWindow::on_btnRoSync_clicked() { fetchNodeStatus("ro_pump"); }
void MainWindow::on_chkRoOverride_toggled(bool checked) { sendNodeCommand("ro_pump", checked ? "MANUAL_ON" : "MANUAL_OFF"); }
void MainWindow::on_chkRoPump_toggled(bool checked) { sendNodeCommand("ro_pump", checked ? "PUMP_ON" : "PUMP_OFF"); }

// ---------------- Main Door ----------------
void MainWindow::updateMainDoorUi(const QJsonObject &json)
{
    bool pir = json.value("pir").toBool() || (json.value("pir").toInt() == 1) || (json.value("pir").toString() == "Motion");
    doorRelayState = json.value("relay").toBool() || (json.value("relay").toInt() == 1) || (json.value("relay").toString() == "UNLOCKED") || (json.value("relay").toString() == "ON");
    doorBuzzerState = json.value("buzzer").toBool() || (json.value("buzzer").toInt() == 1) || (json.value("buzzer").toString() == "ON");
    doorFlashState = json.value("flash").toBool() || (json.value("flash").toInt() == 1) || (json.value("flash").toString() == "ON");

    QString ip = json.value("ip").toString();
    if (ip.isEmpty()) ip = getIpForNode("door");

    QString now = QDateTime::currentDateTime().toString("HH:mm:ss");

    ui->lblDoorPir->setText(pir ? "Motion" : "Clear");
    ui->lblDoorRelay->setText(doorRelayState ? "UNLOCKED" : "Locked");
    ui->lblDoorBuzzer->setText(doorBuzzerState ? "ON" : "OFF");
    ui->lblDoorFlash->setText(doorFlashState ? "ON" : "OFF");
    ui->lblDoorNodeIp->setText(ip);
    ui->lblDoorLastSync->setText(now);

    ui->btnDoorUnlock->setText(doorRelayState ? "Lock Door (Relay OFF)" : "Unlock Door (Relay ON)");
    ui->btnDoorBuzzer->setText(doorBuzzerState ? "Turn Buzzer OFF" : "Turn Buzzer ON");
    ui->btnDoorFlash->setText(doorFlashState ? "Turn Flash OFF" : "Turn Flash ON");

    ui->lblDashDoorIp->setText("IP: " + ip);
    ui->lblDashDoorStatus->setText(QString("Door: %1 | Motion: %2").arg(doorRelayState ? "UNLOCKED" : "Locked").arg(pir ? "Motion" : "Clear"));
}

void MainWindow::on_btnDoorSync_clicked() { fetchNodeStatus("door"); }

void MainWindow::on_btnDoorUnlock_clicked() {
    doorRelayState = !doorRelayState;
    sendNodeCommand("door", doorRelayState ? "RELAY_ON" : "RELAY_OFF");
    ui->btnDoorUnlock->setText(doorRelayState ? "Lock Door (Relay OFF)" : "Unlock Door (Relay ON)");
}

void MainWindow::on_btnDoorBuzzer_clicked() {
    doorBuzzerState = !doorBuzzerState;
    sendNodeCommand("door", doorBuzzerState ? "BUZZER_ON" : "BUZZER_OFF");
    ui->btnDoorBuzzer->setText(doorBuzzerState ? "Turn Buzzer OFF" : "Turn Buzzer ON");
}

void MainWindow::on_btnDoorFlash_clicked() {
    doorFlashState = !doorFlashState;
    sendNodeCommand("door", doorFlashState ? "FLASH_ON" : "FLASH_OFF");
    ui->btnDoorFlash->setText(doorFlashState ? "Turn Flash OFF" : "Turn Flash ON");
}

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
