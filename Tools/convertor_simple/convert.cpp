#include "convert.h"
#include "ui_convert.h"
#include "fmt_convert.h"
#include <QMessageBox>
#include <stdio.h>
#include <bayer.h>
#include <string.h>
#include <QPainter>
#include <QImage>
#include <QPen>
#include <QFile>
#include <QFileDialog>
#include <QSignalBlocker>
#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QPushButton>
#include <QPainterPath>
#include <QSplitter>
#include <QMouseEvent>
#include <QRubberBand>
#include <QStatusBar>
#include <QSettings>
#include <QTabWidget>
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <new>
#include <utility>
#include <vector>

struct
{
    enum pix_fmt fmt;
    char fmt_name[20];
    int32_t bpp_container;
    int32_t bpp;
    int32_t bpp_shift;
}img_pix_fmt[] = {
{ Y8, "Y8", 8, 8, 0},
{ Y16, "Y16", 16, 16, 0},

{ UYVY, "UYVY", 16, 16, 0},
{ YVYU, "YVYU", 16, 16, 0},
{ YUYV, "YUYV", 16, 16, 0},
{ VYUY, "VYUY", 16, 16, 0},

{ YUV422P_YUV, "YUV422P_YUV", 16, 16, 0},
{ YUV420, "YUV420", 16, 16, 0},
{ NV12, "NV12", 16, 16, 0},

{ RGB444_BGGR, "RGB444_BGGR", 16, 16, 0},
{ RGB555_RGGB, "RGB555_RGGB", 16, 16, 0},
{ RGB565_GBRG, "RGB565_GBRG", 16, 16, 0},
{ RGB565_RGGB, "RGB565_RGGB", 16, 16, 0},
{ RGB565_BGGR, "RGB565_BGGR", 16, 16, 0},
{ RGB565_GRBG, "RGB565_GRBG", 16, 16, 0},

{ BAYER_BGGR, "BAYER_BGGR", 8, 8, 0},
{ BAYER_GBRG, "BAYER_GBRG", 8, 8, 0},
{ BAYER_GRBG, "BAYER_GRBG", 8, 8, 0},
{ BAYER_RGGB, "BAYER_RGGB", 8, 8, 0},
{ RCCG, "RCCG", 16, 16, 8},

{ BMP24_BGR, "BMP24_BGR", 24, 24, 0},
{ BMP24_RGB, "BMP24_RGB", 24, 24, 0},

{ ABMP32_RGB, "BMP32_RGB", 32, 32, 0},
{ BAYER10_PACKED, "BAYER10_PACKED", 16, 16, 0},
{ RGBIR16, "RGBIR", 16, 16, 0},
{ BGGR16, "BGGR16", 16, 16, 0}

};

namespace {

double luminanceAt(const uint8_t *rgb, size_t pixelOffset)
{
    return 0.299 * rgb[pixelOffset] + 0.587 * rgb[pixelOffset + 1] +
            0.114 * rgb[pixelOffset + 2];
}

bool normalizeBayerSamples(const uint8_t *source, uint8_t *destination, size_t pixelCount,
        unsigned int containerBits, unsigned int validBits, unsigned int shift)
{
    if ((containerBits != 8 && containerBits != 16) || validBits == 0 ||
            validBits > containerBits || shift > containerBits ||
            validBits > containerBits - shift)
        return false;

    const uint32_t sampleMask = validBits == 16 ? 0xffffU : (1U << validBits) - 1U;
    const uint32_t sampleMaximum = sampleMask;
    for (size_t index = 0; index < pixelCount; ++index)
    {
        uint32_t sample = source[index];
        if (containerBits == 16)
            sample = static_cast<uint32_t>(source[index * 2]) |
                    (static_cast<uint32_t>(source[index * 2 + 1]) << 8);
        sample = (sample >> shift) & sampleMask;
        destination[index] = static_cast<uint8_t>((sample * 255U + sampleMaximum / 2U) /
                sampleMaximum);
    }
    return true;
}

bool isBayer8Or16Format(enum pix_fmt format)
{
    return format == BAYER_BGGR || format == BAYER_GBRG ||
            format == BAYER_GRBG || format == BAYER_RGGB;
}

class LensShadingPlotWidget : public QWidget
{
public:
    LensShadingPlotWidget(const uint8_t *rgb, unsigned int imageWidth, unsigned int imageHeight,
            QWidget *parent = nullptr) : QWidget(parent), grid_(32 * 32),
            minimum_(std::numeric_limits<double>::max()), maximum_(0), center_(0)
    {
        constexpr int gridSize = 32;
        for (int gridY = 0; gridY < gridSize; ++gridY)
            for (int gridX = 0; gridX < gridSize; ++gridX)
            {
                const unsigned int x0 = static_cast<unsigned int>(
                        static_cast<uint64_t>(gridX) * imageWidth / gridSize);
                const unsigned int x1 = static_cast<unsigned int>(
                        static_cast<uint64_t>(gridX + 1) * imageWidth / gridSize);
                const unsigned int y0 = static_cast<unsigned int>(
                        static_cast<uint64_t>(gridY) * imageHeight / gridSize);
                const unsigned int y1 = static_cast<unsigned int>(
                        static_cast<uint64_t>(gridY + 1) * imageHeight / gridSize);
                double total = 0;
                uint64_t count = 0;
                for (unsigned int y = y0; y < qMin(imageHeight, qMax(y0 + 1, y1)); ++y)
                    for (unsigned int x = x0; x < qMin(imageWidth, qMax(x0 + 1, x1)); ++x)
                    {
                        total += luminanceAt(rgb, (static_cast<size_t>(y) * imageWidth + x) * 3);
                        ++count;
                    }
                grid_[gridY * gridSize + gridX] = count ? total / count : 0;
            }

        double centerTotal = 0;
        for (int y = 12; y < 20; ++y)
            for (int x = 12; x < 20; ++x)
                centerTotal += grid_[y * gridSize + x];
        center_ = centerTotal / 64.0;
        if (center_ > 0)
            for (double &value : grid_)
            {
                value /= center_;
                minimum_ = qMin(minimum_, value);
                maximum_ = qMax(maximum_, value);
            }
        setMinimumSize(480, 360);
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.fillRect(rect(), QColor("#f4f3ef"));
        painter.setPen(QColor("#202628"));
        painter.setFont(QFont("Segoe UI", 14, QFont::DemiBold));
        painter.drawText(QRect(24, 12, width() - 48, 30), Qt::AlignLeft | Qt::AlignVCenter,
                "Lens shading | normalized luminance contours");

        const QRectF plot(64, 58, width() - 104, height() - 170);
        painter.fillRect(plot, QColor("#101516"));
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1));
        for (int index = 1; index < 8; ++index)
        {
            const qreal x = plot.left() + plot.width() * index / 8.0;
            const qreal y = plot.top() + plot.height() * index / 8.0;
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
        }
        if (center_ <= 0)
        {
            painter.setPen(Qt::white);
            painter.drawText(plot, Qt::AlignCenter, "No luminance data to normalize");
            return;
        }

        for (int levelIndex = 1; levelIndex <= 8; ++levelIndex)
        {
            if (maximum_ - minimum_ < 0.005)
                break;
            const double level = minimum_ + (maximum_ - minimum_) * levelIndex / 9.0;
            painter.setPen(QPen(QColor(220, 232, 228, 190), 1.2));
            for (int gridY = 0; gridY < 31; ++gridY)
                for (int gridX = 0; gridX < 31; ++gridX)
                {
                    const QPointF points[4] = {
                        gridPoint(plot, gridX, gridY), gridPoint(plot, gridX + 1, gridY),
                        gridPoint(plot, gridX + 1, gridY + 1), gridPoint(plot, gridX, gridY + 1)
                    };
                    const double values[4] = {
                        grid_[gridY * 32 + gridX], grid_[gridY * 32 + gridX + 1],
                        grid_[(gridY + 1) * 32 + gridX + 1], grid_[(gridY + 1) * 32 + gridX]
                    };
                    QPointF intersections[4];
                    int count = 0;
                    for (int edge = 0; edge < 4; ++edge)
                    {
                        const int next = (edge + 1) % 4;
                        if ((values[edge] < level && values[next] >= level) ||
                                (values[next] < level && values[edge] >= level))
                        {
                            const double portion = (level - values[edge]) /
                                    (values[next] - values[edge]);
                            intersections[count++] = points[edge] +
                                    (points[next] - points[edge]) * portion;
                        }
                    }
                    for (int point = 0; point + 1 < count; point += 2)
                        painter.drawLine(intersections[point], intersections[point + 1]);
                }
        }

        painter.setPen(QColor("#202628"));
        painter.setFont(QFont("Segoe UI", 10));
        painter.drawText(QRectF(24, plot.bottom() + 12, width() - 48, 36),
                Qt::AlignLeft | Qt::AlignVCenter,
                QString("Center reference: %1 DN   |   Relative range: %2 to %3")
                .arg(center_, 0, 'f', 1).arg(minimum_, 0, 'f', 3).arg(maximum_, 0, 'f', 3));
        painter.drawText(QRectF(24, height() - 30, width() - 48, 20),
                Qt::AlignLeft | Qt::AlignVCenter,
                "For lens-shading measurement, use a uniform-field capture.");
    }

private:
    static QPointF gridPoint(const QRectF &plot, int x, int y)
    {
        return QPointF(plot.left() + plot.width() * x / 31.0,
                plot.top() + plot.height() * y / 31.0);
    }

    std::vector<double> grid_;
    double minimum_;
    double maximum_;
    double center_;
};

class AETuningPlotWidget : public QWidget
{
public:
    AETuningPlotWidget(const uint8_t *rgb, unsigned int width, unsigned int height,
            QWidget *parent = nullptr) : QWidget(parent), histogram_(256, 0),
            rowRed_(height, 0), rowGreen_(height, 0), rowBlue_(height, 0),
            rowNoise_(height, 0), columnLuminance_(width, 0), meanLuminance_(0)
    {
        uint64_t pixelCount = 0;
        for (unsigned int y = 0; y < height; ++y)
        {
            double rowY = 0;
            double rowY2 = 0;
            for (unsigned int x = 0; x < width; ++x)
            {
                const size_t offset = (static_cast<size_t>(y) * width + x) * 3;
                const double red = rgb[offset];
                const double green = rgb[offset + 1];
                const double blue = rgb[offset + 2];
                const double luminance = 0.299 * red + 0.587 * green + 0.114 * blue;
                ++histogram_[qBound(0, static_cast<int>(luminance), 255)];
                rowRed_[y] += red;
                rowGreen_[y] += green;
                rowBlue_[y] += blue;
                rowY += luminance;
                rowY2 += luminance * luminance;
                columnLuminance_[x] += luminance;
                meanLuminance_ += luminance;
                ++pixelCount;
            }
            const double rowMean = rowY / width;
            rowNoise_[y] = std::sqrt(qMax(0.0, rowY2 / width - rowMean * rowMean));
            rowRed_[y] /= width;
            rowGreen_[y] /= width;
            rowBlue_[y] /= width;
        }
        for (double &value : columnLuminance_)
            value /= height;
        meanLuminance_ /= qMax<uint64_t>(pixelCount, 1);
        setMinimumSize(480, 360);
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.fillRect(rect(), QColor("#f4f3ef"));
        painter.setPen(QColor("#202628"));
        painter.setFont(QFont("Segoe UI", 14, QFont::DemiBold));
        painter.drawText(QRect(24, 10, width() - 48, 28), Qt::AlignLeft | Qt::AlignVCenter,
                "AE tuning | single-frame image statistics");
        painter.setFont(QFont("Segoe UI", 9));
        painter.drawText(QRect(24, 38, width() - 48, 22), Qt::AlignLeft | Qt::AlignVCenter,
                QString("Mean luminance: %1 / 255   |   No exposure sweep supplied")
                .arg(meanLuminance_, 0, 'f', 1));

        const qreal gap = 20;
        const qreal chartWidth = (width() - 3 * gap) / 2.0;
        const qreal chartHeight = (height() - 4 * gap - 64) / 2.0;
        QRectF charts[4] = {
            QRectF(gap, 64, chartWidth, chartHeight),
            QRectF(2 * gap + chartWidth, 64, chartWidth, chartHeight),
            QRectF(gap, 2 * gap + 64 + chartHeight, chartWidth, chartHeight),
            QRectF(2 * gap + chartWidth, 2 * gap + 64 + chartHeight, chartWidth, chartHeight)
        };
        drawChart(painter, charts[0], "Luminance histogram", histogram_, QColor("#247c79"), 0, -1);
        drawChart(painter, charts[1], "RGB mean by image row", rowRed_, QColor("#d65f53"), 0, 255,
            &rowGreen_, QColor("#39936c"), &rowBlue_, QColor("#4b78ae"));
        drawChart(painter, charts[2], "Luminance variation by image row", rowNoise_, QColor("#c17835"), 0, -1);
        drawChart(painter, charts[3], "Mean luminance by image column", columnLuminance_, QColor("#247c79"), 0, 255);
    }

private:
    static void drawChart(QPainter &painter, QRectF bounds, const QString &title,
            const std::vector<double> &values, const QColor &color, double minimum, double maximum,
            const std::vector<double> *secondValues = nullptr, const QColor &secondColor = QColor(),
            const std::vector<double> *thirdValues = nullptr, const QColor &thirdColor = QColor())
    {
        painter.fillRect(bounds, QColor("#ffffff"));
        painter.setPen(QColor("#202628"));
        painter.setFont(QFont("Segoe UI", 9, QFont::DemiBold));
        if (!title.isEmpty())
            painter.drawText(bounds.adjusted(12, 7, -8, -8), Qt::AlignTop | Qt::AlignLeft, title);
        const QRectF plot = bounds.adjusted(38, 30, -12, -27);
        painter.setPen(QPen(QColor("#dfe3df"), 1));
        for (int line = 0; line <= 4; ++line)
        {
            const qreal y = plot.top() + plot.height() * line / 4.0;
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
        }

        const auto drawSeries = [&painter, &plot](const std::vector<double> &series,
                const QColor &seriesColor, double low, double high) {
            if (series.empty())
                return;
            QPainterPath path;
            for (size_t index = 0; index < series.size(); ++index)
            {
                const double x = series.size() == 1 ? 0 :
                        static_cast<double>(index) / (series.size() - 1);
                const double normalized = qBound(0.0, (series[index] - low) / (high - low), 1.0);
                const QPointF point(plot.left() + x * plot.width(), plot.bottom() - normalized * plot.height());
                if (index == 0)
                    path.moveTo(point);
                else
                    path.lineTo(point);
            }
            painter.setPen(QPen(seriesColor, 1.8));
            painter.drawPath(path);
        };
        if (maximum <= minimum)
        {
            maximum = *std::max_element(values.begin(), values.end());
            minimum = 0;
            if (maximum <= minimum)
                maximum = 1;
        }
        drawSeries(values, color, minimum, maximum);
        if (secondValues)
            drawSeries(*secondValues, secondColor, minimum, maximum);
        if (thirdValues)
            drawSeries(*thirdValues, thirdColor, minimum, maximum);
    }

    std::vector<double> histogram_;
    std::vector<double> rowRed_;
    std::vector<double> rowGreen_;
    std::vector<double> rowBlue_;
    std::vector<double> rowNoise_;
    std::vector<double> columnLuminance_;
    double meanLuminance_;
};

class DynamicRangePlotWidget : public QWidget
{
public:
    DynamicRangePlotWidget(const uint8_t *rgb, unsigned int width, unsigned int height,
            QWidget *parent = nullptr) : QWidget(parent), histogram_(256, 0), pixelCount_(0),
            mean_(0), lowPercentile_(0), highPercentile_(0)
    {
        for (size_t index = 0; index < static_cast<size_t>(width) * height; ++index)
        {
            const int luminance = qBound(0, static_cast<int>(luminanceAt(rgb, index * 3)), 255);
            ++histogram_[luminance];
            mean_ += luminance;
            ++pixelCount_;
        }
        if (pixelCount_ != 0)
        {
            mean_ /= pixelCount_;
            lowPercentile_ = percentile(0.01);
            highPercentile_ = percentile(0.99);
        }
        setMinimumSize(480, 360);
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.fillRect(rect(), QColor("#f4f3ef"));
        painter.setPen(QColor("#202628"));
        painter.setFont(QFont("Segoe UI", 14, QFont::DemiBold));
        painter.drawText(QRect(24, 12, width() - 48, 30), Qt::AlignLeft | Qt::AlignVCenter,
                "Dynamic range | single-frame luminance distribution");
        painter.setFont(QFont("Segoe UI", 9));
        painter.drawText(QRect(24, 42, width() - 48, 34), Qt::AlignLeft | Qt::AlignVCenter,
                "Encoded tonal span only. Sensor dynamic range requires exposure and noise measurements.");

        const QRectF plot(66, 92, width() - 110, height() - 184);
        painter.fillRect(plot, Qt::white);
        painter.setPen(QPen(QColor("#dfe3df"), 1));
        for (int line = 0; line <= 4; ++line)
        {
            const qreal y = plot.top() + plot.height() * line / 4.0;
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
        }
        const uint64_t maximum = *std::max_element(histogram_.begin(), histogram_.end());
        if (maximum != 0)
        {
            const qreal barWidth = plot.width() / 256.0;
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor("#247c79"));
            for (int value = 0; value < 256; ++value)
            {
                const qreal barHeight = plot.height() * histogram_[value] / maximum;
                painter.drawRect(QRectF(plot.left() + value * barWidth, plot.bottom() - barHeight,
                        qMax<qreal>(1, barWidth), barHeight));
            }
            drawMarker(painter, plot, lowPercentile_, QColor("#d65f53"));
            drawMarker(painter, plot, highPercentile_, QColor("#c17835"));
        }

        const double clippedBlack = pixelCount_ ? 100.0 * histogram_[0] / pixelCount_ : 0;
        const double clippedWhite = pixelCount_ ? 100.0 * histogram_[255] / pixelCount_ : 0;
        painter.setPen(QColor("#202628"));
        painter.setFont(QFont("Segoe UI", 10));
        painter.drawText(QRectF(24, plot.bottom() + 10, width() - 48, 68),
            Qt::AlignLeft | Qt::AlignVCenter,
            QString("Mean Y: %1 / 255   |   Encoded P1-P99: %2-%3 codes\nClipped black: %4%   |   Clipped white: %5%")
            .arg(mean_, 0, 'f', 1).arg(lowPercentile_).arg(highPercentile_)
            .arg(clippedBlack, 0, 'f', 2).arg(clippedWhite, 0, 'f', 2));
    }

private:
    int percentile(double fraction) const
    {
        const uint64_t target = static_cast<uint64_t>(pixelCount_ * fraction);
        uint64_t cumulative = 0;
        for (int value = 0; value < 256; ++value)
        {
            cumulative += histogram_[value];
            if (cumulative >= target)
                return value;
        }
        return 255;
    }

    static void drawMarker(QPainter &painter, const QRectF &plot, int value, const QColor &color)
    {
        const qreal x = plot.left() + plot.width() * value / 255.0;
        painter.setPen(QPen(color, 2, Qt::DashLine));
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
    }

    std::vector<uint64_t> histogram_;
    uint64_t pixelCount_;
    double mean_;
    int lowPercentile_;
    int highPercentile_;
};

struct ColorLab
{
    double l;
    double a;
    double b;
};

struct ColorPatchSample
{
    int index;
    ColorLab target;
    ColorLab measured;
    QColor expectedColor;
    QColor measuredColor;
    double deltaE;
};

QColor samplePatchColor(const uint8_t *rgb, unsigned int imageWidth, unsigned int imageHeight,
        unsigned int left, unsigned int top, unsigned int right, unsigned int bottom)
{
    const unsigned int cellWidth = right - left;
    const unsigned int cellHeight = bottom - top;
    const unsigned int x0 = left + cellWidth * 28 / 100;
    const unsigned int x1 = qMin(imageWidth, qMax(x0 + 1, left + cellWidth * 72 / 100));
    const unsigned int y0 = top + cellHeight * 28 / 100;
    const unsigned int y1 = qMin(imageHeight, qMax(y0 + 1, top + cellHeight * 72 / 100));
    uint64_t red = 0;
    uint64_t green = 0;
    uint64_t blue = 0;
    uint64_t count = 0;
    for (unsigned int y = y0; y < y1; ++y)
        for (unsigned int x = x0; x < x1; ++x)
        {
            const size_t offset = (static_cast<size_t>(y) * imageWidth + x) * 3;
            red += rgb[offset];
            green += rgb[offset + 1];
            blue += rgb[offset + 2];
            ++count;
        }
    if (count == 0)
        return QColor();
    return QColor(static_cast<int>((red + count / 2) / count),
            static_cast<int>((green + count / 2) / count),
            static_cast<int>((blue + count / 2) / count));
}

class AWBCCMPlotWidget : public QWidget
{
public:
    AWBCCMPlotWidget(const uint8_t *rgb, unsigned int width, unsigned int height,
            unsigned int roiLeft, unsigned int roiTop, unsigned int roiWidth, unsigned int roiHeight,
            QWidget *parent = nullptr) : QWidget(parent), meanDeltaE_(0), maximumDeltaE_(0),
            gainRed_(1), gainGreen_(1), gainBlue_(1)
    {
        static const int referenceRgb[24][3] = {
            {115, 82, 68}, {194, 150, 130}, {98, 122, 157}, {87, 108, 67},
            {133, 128, 177}, {103, 189, 170}, {214, 126, 44}, {80, 91, 166},
            {193, 90, 99}, {94, 60, 108}, {157, 188, 64}, {224, 163, 46},
            {56, 61, 150}, {70, 148, 73}, {175, 54, 60}, {231, 199, 31},
            {187, 86, 149}, {8, 133, 161}, {243, 243, 242}, {200, 200, 200},
            {160, 160, 160}, {122, 122, 121}, {85, 85, 85}, {52, 52, 52}
        };

        double neutralRed = 0;
        double neutralGreen = 0;
        double neutralBlue = 0;
        for (int patch = 0; patch < 24; ++patch)
        {
            const unsigned int cellLeft = roiLeft + (patch % 6) * roiWidth / 6;
            const unsigned int cellTop = roiTop + (patch / 6) * roiHeight / 4;
            const unsigned int cellRight = roiLeft + (patch % 6 + 1) * roiWidth / 6;
            const unsigned int cellBottom = roiTop + (patch / 6 + 1) * roiHeight / 4;
            const QColor measuredColor = samplePatchColor(rgb, width, height,
                    cellLeft, cellTop, cellRight, cellBottom);
            if (!measuredColor.isValid())
                continue;

            const double red = measuredColor.red();
            const double green = measuredColor.green();
            const double blue = measuredColor.blue();
            const ColorLab target = toLab(referenceRgb[patch][0] / 255.0,
                    referenceRgb[patch][1] / 255.0, referenceRgb[patch][2] / 255.0);
            const ColorLab measured = toLab(red / 255.0, green / 255.0, blue / 255.0);
            const double deltaL = target.l - measured.l;
            const double deltaA = target.a - measured.a;
            const double deltaB = target.b - measured.b;
            const double deltaE = std::sqrt(deltaL * deltaL + deltaA * deltaA + deltaB * deltaB);
                samples_.push_back({patch + 1, target, measured,
                    QColor(referenceRgb[patch][0], referenceRgb[patch][1], referenceRgb[patch][2]),
                    measuredColor, deltaE});
            meanDeltaE_ += deltaE;
            maximumDeltaE_ = qMax(maximumDeltaE_, deltaE);
            if (patch >= 18 && patch < 23)
            {
                neutralRed += red;
                neutralGreen += green;
                neutralBlue += blue;
            }
        }
        if (!samples_.empty())
        {
            meanDeltaE_ /= samples_.size();
            neutralRed /= 5.0;
            neutralGreen /= 5.0;
            neutralBlue /= 5.0;
            const double neutralMean = (neutralRed + neutralGreen + neutralBlue) / 3.0;
            gainRed_ = neutralRed > 0 ? neutralMean / neutralRed : 0;
            gainGreen_ = neutralGreen > 0 ? neutralMean / neutralGreen : 0;
            gainBlue_ = neutralBlue > 0 ? neutralMean / neutralBlue : 0;
        }
        setMinimumSize(480, 360);
    }

    const std::vector<ColorPatchSample> &samples() const
    {
        return samples_;
    }

protected:
    void paintEvent(QPaintEvent *) override
    {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.fillRect(rect(), QColor("#f4f3ef"));
        painter.setPen(QColor("#202628"));
        painter.setFont(QFont("Segoe UI", 14, QFont::DemiBold));
        painter.drawText(QRect(24, 12, width() - 48, 30), Qt::AlignLeft | Qt::AlignVCenter,
            "AWB & CCM | ColorChecker patch comparison");
        painter.setFont(QFont("Segoe UI", 9));
        painter.drawText(QRect(24, 42, width() - 48, 34), Qt::AlignLeft | Qt::AlignVCenter,
            "Set X/Y to chart center and Width/Height to its bounds; reference squares, measured circles.");

        const QRectF plot(88, 96, width() - 136, height() - 194);
        painter.fillRect(plot, Qt::white);
        painter.setPen(QPen(QColor("#dfe3df"), 1));
        for (int tick = -80; tick <= 80; tick += 40)
        {
            const qreal x = plot.left() + plot.width() * (tick + 100) / 200.0;
            const qreal y = plot.top() + plot.height() * (100 - tick) / 200.0;
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
        }
        painter.setPen(QColor("#202628"));
        painter.setFont(QFont("Segoe UI", 9));
        painter.drawText(QRectF(plot.left(), plot.bottom() + 2, plot.width(), 22), Qt::AlignCenter, "a* (green to red)");
        painter.save();
        painter.translate(26, plot.center().y());
        painter.rotate(-90);
        painter.drawText(QRectF(-plot.height() / 2, -10, plot.height(), 20), Qt::AlignCenter, "b* (blue to yellow)");
        painter.restore();
        painter.setPen(QPen(QColor("#202628"), 1));
        painter.drawRect(plot);

        painter.save();
        painter.setClipRect(plot);
        for (const ColorPatchSample &sample : samples_)
        {
            const QPointF target = mapLab(plot, sample.target);
            const QPointF measured = mapLab(plot, sample.measured);
            painter.setPen(QPen(QColor(90, 98, 100, 150), 1));
            painter.drawLine(target, measured);
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor("#247c79"));
            painter.drawRect(QRectF(target.x() - 3.5, target.y() - 3.5, 7, 7));
            painter.setBrush(sample.measuredColor);
            painter.drawEllipse(measured, 4, 4);
        }
        painter.restore();

        painter.setPen(QColor("#202628"));
        painter.setFont(QFont("Segoe UI", 10));
        painter.drawText(QRectF(24, plot.bottom() + 24, width() - 48, 58),
                Qt::AlignLeft | Qt::AlignVCenter,
                QString("Patches: %1/24   |   DeltaE76 mean: %2   max: %3\nNeutral-patch AWB gain estimate: R %4  G %5  B %6")
                .arg(samples_.size()).arg(meanDeltaE_, 0, 'f', 1).arg(maximumDeltaE_, 0, 'f', 1)
                .arg(gainRed_, 0, 'f', 3).arg(gainGreen_, 0, 'f', 3).arg(gainBlue_, 0, 'f', 3));
        painter.setBrush(QColor("#247c79"));
        painter.setPen(Qt::NoPen);
        painter.drawRect(QRectF(plot.right() - 164, plot.top() + 8, 8, 8));
        painter.setPen(QColor("#202628"));
        painter.drawText(QPointF(plot.right() - 150, plot.top() + 16), "Reference");
        painter.setBrush(QColor("#b45d45"));
        painter.drawEllipse(QPointF(plot.right() - 92, plot.top() + 12), 4, 4);
        painter.setPen(QColor("#202628"));
        painter.drawText(QPointF(plot.right() - 82, plot.top() + 16), "Camera");
    }

private:
    static double linearize(double value)
    {
        return value <= 0.04045 ? value / 12.92 : std::pow((value + 0.055) / 1.055, 2.4);
    }

    static double labCurve(double value)
    {
        return value > 0.008856 ? std::cbrt(value) : 7.787 * value + 16.0 / 116.0;
    }

    static QPointF mapLab(const QRectF &plot, const ColorLab &lab)
    {
        return QPointF(plot.left() + plot.width() * (lab.a + 100) / 200.0,
                plot.top() + plot.height() * (100 - lab.b) / 200.0);
    }

    static ColorLab toLab(double red, double green, double blue)
    {
        red = linearize(red);
        green = linearize(green);
        blue = linearize(blue);
        const double x = labCurve((0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047);
        const double y = labCurve(0.2126729 * red + 0.7151522 * green + 0.0721750 * blue);
        const double z = labCurve((0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883);
        return {116 * y - 16, 500 * (x - y), 200 * (y - z)};
    }

    std::vector<ColorPatchSample> samples_;
    double meanDeltaE_;
    double maximumDeltaE_;
    double gainRed_;
    double gainGreen_;
    double gainBlue_;
};

    class ColorPatchRgbWidget : public QWidget
    {
    public:
        explicit ColorPatchRgbWidget(const std::vector<ColorPatchSample> &samples,
            QWidget *parent = nullptr) : QWidget(parent), samples_(samples)
        {
        setMinimumSize(480, 360);
        }

    protected:
        void paintEvent(QPaintEvent *) override
        {
        QPainter painter(this);
        painter.fillRect(rect(), QColor("#f4f3ef"));
        painter.setPen(QColor("#202628"));
        painter.setFont(QFont("Segoe UI", 11, QFont::DemiBold));
        painter.drawText(QRect(12, 6, width() - 24, 24), Qt::AlignLeft | Qt::AlignVCenter,
            "ColorChecker RGB | expected vs measured");
        painter.setFont(QFont("Segoe UI", 8));
        painter.drawText(QRect(12, 29, width() - 24, 18), Qt::AlignLeft | Qt::AlignVCenter,
            "E = reference RGB    M = measured camera RGB    |    patch numbers follow chart order");

        const qreal gap = 4;
        const qreal cellWidth = (width() - 14 - 5 * gap) / 6.0;
        const qreal cellHeight = (height() - 55 - 3 * gap) / 4.0;
        painter.setFont(QFont("Segoe UI", 7));
        for (const ColorPatchSample &sample : samples_)
        {
            const int index = sample.index - 1;
            const QRectF cell(7 + (index % 6) * (cellWidth + gap),
                50 + (index / 6) * (cellHeight + gap), cellWidth, cellHeight);
            painter.fillRect(cell, Qt::white);
            painter.setPen(QPen(QColor("#c7cdca"), 1));
            painter.drawRect(cell);
            painter.setPen(QColor("#202628"));
            painter.drawText(QRectF(cell.left() + 3, cell.top() + 2, cell.width() - 6, 12),
                Qt::AlignLeft | Qt::AlignVCenter, QString("Patch %1").arg(sample.index));

            const QRectF expectedSwatch(cell.left() + 3, cell.top() + 18, 15, 13);
            painter.fillRect(expectedSwatch, sample.expectedColor);
            painter.setPen(QPen(QColor("#6c7371"), 1));
            painter.drawRect(expectedSwatch);
            painter.setPen(QColor("#202628"));
            painter.drawText(QRectF(cell.left() + 21, cell.top() + 17,
                cell.width() - 23, 15), Qt::AlignLeft | Qt::AlignVCenter,
                QString("E %1,%2,%3").arg(sample.expectedColor.red())
                .arg(sample.expectedColor.green()).arg(sample.expectedColor.blue()));

            const QRectF measuredSwatch(cell.left() + 3, cell.top() + 37, 15, 13);
            painter.fillRect(measuredSwatch, sample.measuredColor);
            painter.setPen(QPen(QColor("#6c7371"), 1));
            painter.drawRect(measuredSwatch);
            painter.setPen(QColor("#202628"));
            painter.drawText(QRectF(cell.left() + 21, cell.top() + 36,
                cell.width() - 23, 15), Qt::AlignLeft | Qt::AlignVCenter,
                QString("M %1,%2,%3").arg(sample.measuredColor.red())
                .arg(sample.measuredColor.green()).arg(sample.measuredColor.blue()));
            painter.drawText(QRectF(cell.left() + 3, cell.top() + 53,
                cell.width() - 6, 14), Qt::AlignLeft | Qt::AlignVCenter,
                QString("DeltaE %1").arg(sample.deltaE, 0, 'f', 1));
        }
        }

    private:
        std::vector<ColorPatchSample> samples_;
    };

}

convert::convert(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::convert),
    rightPane(nullptr),
    analysisPage(nullptr),
    chartSelectionBand(nullptr),
    chartSelectionArmed(false),
    chartSelectionDragging(false),
    src_buffer(NULL),
    des_buffer(NULL)
{
    ui->setupUi(this);
    ui->draw_window->installEventFilter(this);
    ui->draw_window->setMouseTracking(true);
    rightPane = new QStackedWidget(ui->centralWidget);
        rightPane->setMinimumSize(320, 240);
    rightPane->setSizePolicy(ui->modified_draw_window->sizePolicy());
    QLayoutItem *rightPaneSlot = ui->horizontalLayout_5->replaceWidget(ui->modified_draw_window, rightPane);
    delete rightPaneSlot;
    rightPane->addWidget(ui->modified_draw_window);

        QHBoxLayout *imageToolsLayout = ui->horizontalLayout_5;
        imageToolsLayout->setSizeConstraint(QLayout::SetNoConstraint);
        imageToolsLayout->removeWidget(ui->draw_window);
        imageToolsLayout->removeWidget(ui->line_8);
        imageToolsLayout->removeItem(ui->gridLayout);
        imageToolsLayout->removeWidget(ui->line_7);
        imageToolsLayout->removeWidget(rightPane);
        delete ui->line_8;
        delete ui->line_7;

        QWidget *toolsPane = new QWidget(ui->centralWidget);
        ui->gridLayout->setSizeConstraint(QLayout::SetNoConstraint);
        for (int itemIndex = 0; itemIndex < ui->gridLayout->count(); ++itemIndex)
        {
            QWidget *itemWidget = ui->gridLayout->itemAt(itemIndex)->widget();
            if (itemWidget)
                itemWidget->setParent(toolsPane);
        }
        ui->gridLayout->setParent(nullptr);
        toolsPane->setLayout(ui->gridLayout);
        toolsPane->setMinimumWidth(230);
        toolsPane->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        ui->draw_window->setMinimumSize(320, 240);
        ui->modified_draw_window->setMinimumSize(320, 240);

        QSplitter *paneSplitter = new QSplitter(Qt::Horizontal, ui->centralWidget);
        paneSplitter->setObjectName(QStringLiteral("paneSplitter"));
        paneSplitter->setHandleWidth(8);
        paneSplitter->setChildrenCollapsible(false);
        paneSplitter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        paneSplitter->setStyleSheet(QStringLiteral(
            "QSplitter::handle:horizontal { background: #454b4d; }"
            "QSplitter::handle:horizontal:hover { background: #54a89f; }"));
        paneSplitter->addWidget(ui->draw_window);
        paneSplitter->addWidget(toolsPane);
        paneSplitter->addWidget(rightPane);
        for (int handleIndex = 1; handleIndex < paneSplitter->count(); ++handleIndex)
        {
            paneSplitter->handle(handleIndex)->setCursor(Qt::SplitHCursor);
            paneSplitter->handle(handleIndex)->setToolTip(tr("Drag to resize panes"));
        }
        paneSplitter->setStretchFactor(0, 3);
        paneSplitter->setStretchFactor(1, 1);
        paneSplitter->setStretchFactor(2, 2);
        paneSplitter->setSizes(QList<int>() << 700 << 220 << 480);
        imageToolsLayout->addWidget(paneSplitter);
        ui->verticalLayout_7->setStretch(2, 1);
    QPushButton *lensShadingButton = new QPushButton(tr("Lens Shading"), ui->centralWidget);
    QPushButton *aeTuningButton = new QPushButton(tr("AE Tuning"), ui->centralWidget);
    QPushButton *lensDistortionButton = new QPushButton(tr("LDC (Fisheye)"), ui->centralWidget);
    QPushButton *dynamicRangeButton = new QPushButton(tr("Dynamic Range"), ui->centralWidget);
    QPushButton *awbCcmButton = new QPushButton(tr("AWB & CCM"), ui->centralWidget);
    QPushButton *selectChartButton = new QPushButton(tr("Select chart"), ui->centralWidget);
    ui->gridLayout->addWidget(lensShadingButton, 1, 0, 1, 3);
    ui->gridLayout->addWidget(aeTuningButton, 2, 0, 1, 3);
    ui->gridLayout->addWidget(dynamicRangeButton, 4, 0, 1, 3);
    ui->gridLayout->addWidget(awbCcmButton, 5, 0, 1, 3);
    ui->gridLayout->addWidget(lensDistortionButton, 3, 0, 1, 3);
    ui->gridLayout->removeWidget(ui->line_13);
    ui->gridLayout->removeWidget(ui->line_14);
    ui->gridLayout->removeWidget(ui->line_15);
    delete ui->line_13;
    delete ui->line_14;
    delete ui->line_15;
    ui->gridLayout->removeWidget(ui->equalize);
    ui->gridLayout->addWidget(ui->equalize, 0, 0, 1, 3);
    ui->gridLayout->removeWidget(ui->Crop);
    ui->gridLayout->removeWidget(ui->label_12);
    ui->gridLayout->removeWidget(ui->crop_x);
    ui->gridLayout->removeWidget(ui->label_13);
    ui->gridLayout->removeWidget(ui->crop_y);
    ui->gridLayout->removeWidget(ui->label_14);
    ui->gridLayout->removeWidget(ui->crop_width);
    ui->gridLayout->removeWidget(ui->label_15);
    ui->gridLayout->removeWidget(ui->crop_height);
    ui->gridLayout->addWidget(selectChartButton, 6, 0, 1, 3);
    ui->gridLayout->addWidget(ui->Crop, 7, 0);
    ui->gridLayout->addWidget(ui->label_12, 7, 1);
    ui->gridLayout->addWidget(ui->crop_x, 7, 2);
    ui->gridLayout->addWidget(ui->label_13, 8, 1);
    ui->gridLayout->addWidget(ui->crop_y, 8, 2);
    ui->gridLayout->addWidget(ui->label_14, 9, 1);
    ui->gridLayout->addWidget(ui->crop_width, 9, 2);
    ui->gridLayout->addWidget(ui->label_15, 10, 1);
    ui->gridLayout->addWidget(ui->crop_height, 10, 2);
    ui->gridLayout->removeWidget(ui->line_10);
    ui->gridLayout->removeWidget(ui->line_11);
    ui->gridLayout->removeWidget(ui->line_12);
    ui->gridLayout->addWidget(ui->line_11, 11, 0);
    ui->gridLayout->addWidget(ui->line_10, 11, 1);
    ui->gridLayout->addWidget(ui->line_12, 11, 2);
    ui->gridLayout->removeWidget(ui->Save);
    ui->gridLayout->addWidget(ui->Save, 12, 0, 1, 3);
    ui->gridLayout->removeItem(ui->verticalSpacer_2);
    ui->gridLayout->addItem(ui->verticalSpacer_2, 13, 0, 1, 3);
    connect(lensShadingButton, &QPushButton::clicked, this, &convert::on_lens_shading_clicked);
    connect(aeTuningButton, &QPushButton::clicked, this, &convert::on_ae_tuning_clicked);
    connect(lensDistortionButton, &QPushButton::clicked, this, &convert::on_lens_distortion_clicked);
    connect(dynamicRangeButton, &QPushButton::clicked, this, &convert::on_dynamic_range_clicked);
    connect(awbCcmButton, &QPushButton::clicked, this, &convert::on_awb_ccm_clicked);
    connect(selectChartButton, &QPushButton::clicked, this, &convert::on_select_chart_clicked);
    {
        const QSignalBlocker blocker(ui->pixel_fmt);
        ui->pixel_fmt->clear();
        for (size_t i = 0; i < (sizeof(img_pix_fmt) / sizeof(img_pix_fmt[0])); ++i)
            ui->pixel_fmt->addItem(img_pix_fmt[i].fmt_name);
    }
    restoreSettings();
    width = atoi(ui->width->text().toLocal8Bit().data());
    height = atoi(ui->height->text().toLocal8Bit().data());
    frame_stride = atoi(ui->frame_stride->text().toLocal8Bit().data());
    pix_fmt = (enum pix_fmt)ui->pixel_fmt->currentIndex();
    bpp = atoi(ui->bpp->text().toLocal8Bit().data());
    bpp_container = atoi(ui->bpp_container->text().toLocal8Bit().data());
    bpp_shift = atoi(ui->bpp_shift->text().toLocal8Bit().data());
    if (QFile::exists(ui->file_path->text()))
        paintimage();

}

convert::~convert()
{
    saveSettings();
    free(src_buffer);
    free(des_buffer);
    delete ui;
}

void convert::restoreSettings()
{
    QSettings settings(QSettings::IniFormat, QSettings::UserScope,
            QStringLiteral("Ananth"), QStringLiteral("convertor_simple"));
    ui->file_path->setText(settings.value("input/filePath", ui->file_path->text()).toString());
    ui->width->setText(settings.value("input/width", ui->width->text()).toString());
    ui->height->setText(settings.value("input/height", ui->height->text()).toString());
    ui->frame_stride->setText(settings.value("input/frameStride", ui->frame_stride->text()).toString());
    ui->bpp->setText(settings.value("format/bpp", ui->bpp->text()).toString());
    ui->bpp_container->setText(settings.value("format/bppContainer", ui->bpp_container->text()).toString());
    ui->bpp_shift->setText(settings.value("format/bppShift", ui->bpp_shift->text()).toString());
    ui->crop_x->setText(settings.value("crop/x", ui->crop_x->text()).toString());
    ui->crop_y->setText(settings.value("crop/y", ui->crop_y->text()).toString());
    ui->crop_width->setText(settings.value("crop/width", ui->crop_width->text()).toString());
    ui->crop_height->setText(settings.value("crop/height", ui->crop_height->text()).toString());

    const QSignalBlocker formatBlocker(ui->pixel_fmt);
    const QSignalBlocker frameBlocker(ui->src_img_count);
    const QString savedFormat = settings.value("format/pixelFormat",
            ui->pixel_fmt->currentText()).toString();
    const int formatIndex = ui->pixel_fmt->findText(savedFormat);
    if (formatIndex >= 0)
        ui->pixel_fmt->setCurrentIndex(formatIndex);
    ui->src_img_count->setRange(0, INT_MAX);
    ui->src_img_count->setValue(qMax(0, settings.value("input/frameIndex", 0).toInt()));
}

void convert::saveSettings() const
{
    QSettings settings(QSettings::IniFormat, QSettings::UserScope,
            QStringLiteral("Ananth"), QStringLiteral("convertor_simple"));
    settings.setValue("input/filePath", ui->file_path->text());
    settings.setValue("input/width", ui->width->text());
    settings.setValue("input/height", ui->height->text());
    settings.setValue("input/frameStride", ui->frame_stride->text());
    settings.setValue("input/frameIndex", ui->src_img_count->value());
    settings.setValue("format/pixelFormat", ui->pixel_fmt->currentText());
    settings.setValue("format/bpp", ui->bpp->text());
    settings.setValue("format/bppContainer", ui->bpp_container->text());
    settings.setValue("format/bppShift", ui->bpp_shift->text());
    settings.setValue("crop/x", ui->crop_x->text());
    settings.setValue("crop/y", ui->crop_y->text());
    settings.setValue("crop/width", ui->crop_width->text());
    settings.setValue("crop/height", ui->crop_height->text());
    settings.sync();
}

void convert::changeEvent(QEvent *e)
{
    QMainWindow::changeEvent(e);
    switch (e->type()) {
    case QEvent::LanguageChange:
        ui->retranslateUi(this);
        break;
    default:
        break;
    }
}

bool convert::eventFilter(QObject *watched, QEvent *event)
{
    if (watched != ui->draw_window || !chartSelectionArmed)
        return QMainWindow::eventFilter(watched, event);

    const QRect imageBounds = ui->draw_window->contentsRect();
    if (event->type() == QEvent::MouseButtonPress)
    {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent *>(event);
        const QPoint position = mouseEvent->position().toPoint();
        if (mouseEvent->button() != Qt::LeftButton || !imageBounds.contains(position))
            return true;
        chartSelectionDragging = true;
        chartSelectionStart = position;
        if (!chartSelectionBand)
            chartSelectionBand = new QRubberBand(QRubberBand::Rectangle, ui->draw_window);
        chartSelectionBand->setGeometry(QRect(chartSelectionStart, chartSelectionStart));
        chartSelectionBand->show();
        return true;
    }

    if (event->type() == QEvent::MouseMove && chartSelectionDragging)
    {
        const QPoint position = static_cast<QMouseEvent *>(event)->position().toPoint();
        chartSelectionBand->setGeometry(QRect(chartSelectionStart, position).normalized()
                .intersected(imageBounds));
        return true;
    }

    if (event->type() == QEvent::MouseButtonRelease && chartSelectionDragging)
    {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent *>(event);
        const QRect selection = QRect(chartSelectionStart, mouseEvent->position().toPoint())
                .normalized().intersected(imageBounds);
        chartSelectionDragging = false;
        chartSelectionArmed = false;
        ui->draw_window->unsetCursor();
        if (chartSelectionBand)
            chartSelectionBand->hide();

        if (selection.width() < 6 || selection.height() < 4 ||
                imageBounds.width() <= 0 || imageBounds.height() <= 0)
        {
            statusBar()->showMessage(tr("Chart selection is too small; drag around the full chart."), 5000);
            return true;
        }

        const qint64 left = static_cast<qint64>(selection.left() - imageBounds.left()) * width /
                imageBounds.width();
        const qint64 top = static_cast<qint64>(selection.top() - imageBounds.top()) * height /
                imageBounds.height();
        const qint64 right = (static_cast<qint64>(selection.right() - imageBounds.left() + 1) * width +
                imageBounds.width() - 1) / imageBounds.width();
        const qint64 bottom = (static_cast<qint64>(selection.bottom() - imageBounds.top() + 1) * height +
                imageBounds.height() - 1) / imageBounds.height();
        const int roiWidth = static_cast<int>(right - left);
        const int roiHeight = static_cast<int>(bottom - top);
        ui->crop_x->setText(QString::number(left + roiWidth / 2));
        ui->crop_y->setText(QString::number(top + roiHeight / 2));
        ui->crop_width->setText(QString::number(roiWidth));
        ui->crop_height->setText(QString::number(roiHeight));

        QImage selectedImage(static_cast<int>(width), static_cast<int>(height), QImage::Format_RGB888);
        if (!selectedImage.isNull())
        {
            for (unsigned int row = 0; row < height; ++row)
                memcpy(selectedImage.scanLine(static_cast<int>(row)),
                        des_buffer + static_cast<size_t>(row) * width * 3, width * 3);
            QPainter painter(&selectedImage);
                const int lineWidth = qMax(2, static_cast<int>(width / 960));
                painter.setPen(QPen(QColor("#d6ff4f"), lineWidth));
            for (int gridY = 0; gridY < 4; ++gridY)
            {
                for (int gridX = 0; gridX < 6; ++gridX)
                {
                    const int cellLeft = static_cast<int>(left + gridX * roiWidth / 6);
                    const int cellTop = static_cast<int>(top + gridY * roiHeight / 4);
                    const int cellRight = static_cast<int>(left + (gridX + 1) * roiWidth / 6);
                    const int cellBottom = static_cast<int>(top + (gridY + 1) * roiHeight / 4);
                    painter.drawRect(QRect(cellLeft, cellTop,
                            cellRight - cellLeft, cellBottom - cellTop));

                        const QColor measuredColor = samplePatchColor(des_buffer, width, height,
                            static_cast<unsigned int>(cellLeft), static_cast<unsigned int>(cellTop),
                            static_cast<unsigned int>(cellRight), static_cast<unsigned int>(cellBottom));

                    const int sampleLeft = cellLeft + (cellRight - cellLeft) * 28 / 100;
                    const int sampleRight = cellLeft + (cellRight - cellLeft) * 72 / 100;
                    const int sampleTop = cellTop + (cellBottom - cellTop) * 28 / 100;
                    const int sampleBottom = cellTop + (cellBottom - cellTop) * 72 / 100;
                    painter.setPen(QPen(QColor("#00d9ee"), lineWidth));
                    painter.drawRect(QRect(sampleLeft, sampleTop,
                        sampleRight - sampleLeft, sampleBottom - sampleTop));

                    const QPoint center((sampleLeft + sampleRight) / 2,
                        (sampleTop + sampleBottom) / 2);
                    const int markerRadius = qMax(6, static_cast<int>(width / 240));
                    painter.setPen(QPen(Qt::black, lineWidth));
                        painter.setBrush(measuredColor);
                    painter.drawEllipse(center, markerRadius, markerRadius);
                        painter.setBrush(Qt::NoBrush);
                    painter.setPen(QPen(QColor("#d6ff4f"), lineWidth));
                }
                }

                const QRect legendBounds(16, 16, static_cast<int>(width * 0.55),
                    qMax(42, static_cast<int>(height / 32)));
                painter.fillRect(legendBounds, QColor(0, 0, 0, 185));
                QFont legendFont("Segoe UI");
                legendFont.setPixelSize(qMax(18, static_cast<int>(width / 60)));
                painter.setFont(legendFont);
                painter.setPen(Qt::white);
                painter.drawText(legendBounds.adjusted(10, 2, -8, -2), Qt::AlignVCenter,
                        "Yellow: patch   Cyan: sample area   Center dot: measured RGB");
            ui->draw_window->setPixmap(QPixmap::fromImage(selectedImage));
            ui->draw_window->update();
        }
        statusBar()->showMessage(tr("Chart ROI set: center (%1, %2), size %3 x %4 pixels.")
                .arg(left + roiWidth / 2).arg(top + roiHeight / 2)
                .arg(roiWidth).arg(roiHeight), 8000);
        return true;
    }

    return QMainWindow::eventFilter(watched, event);
}

void convert::on_select_chart_clicked()
{
    if (!des_buffer || width == 0 || height == 0)
    {
        QMessageBox::information(this, tr("Select chart"), tr("Load an image first."));
        return;
    }

    chartSelectionArmed = true;
    chartSelectionDragging = false;
    ui->draw_window->setCursor(Qt::CrossCursor);
    statusBar()->showMessage(tr("Drag a rectangle around the full 6 x 4 color chart in the image."));
}

void convert::paintimage()
{
    bool widthOk = false;
    bool heightOk = false;
    bool strideOk = false;
    const int parsedWidth = ui->width->text().toInt(&widthOk);
    const int parsedHeight = ui->height->text().toInt(&heightOk);
    const int parsedStride = ui->frame_stride->text().toInt(&strideOk);
    const int formatIndex = ui->pixel_fmt->currentIndex();
    if (!widthOk || !heightOk || !strideOk || parsedWidth <= 0 ||
            parsedHeight <= 0 || parsedStride < 0 || formatIndex < 0 ||
            formatIndex >= static_cast<int>(sizeof(img_pix_fmt) / sizeof(img_pix_fmt[0])))
    {
        ui->Source_img_integrity->setText("Fail");
        return;
    }

    width = static_cast<unsigned int>(parsedWidth);
    height = static_cast<unsigned int>(parsedHeight);
    frame_stride = parsedStride;
    pix_fmt = static_cast<enum pix_fmt>(formatIndex);
    bpp = ui->bpp->text().toUInt();
    bpp_container = ui->bpp_container->text().toUInt();
    bpp_shift = ui->bpp_shift->text().toUInt();

    const quint64 pixelCount = static_cast<quint64>(width) * height;
    quint64 frameSize = 0;
    switch (pix_fmt)
    {
        case YUV420:
        case NV12:
            if ((width & 1U) || (height & 1U))
            {
                ui->Source_img_integrity->setText("Fail");
                return;
            }
            frameSize = pixelCount * 3 / 2;
            break;
        case UYVY:
        case YVYU:
        case YUYV:
        case VYUY:
        case YUV422P_YUV:
            if (width & 1U)
            {
                ui->Source_img_integrity->setText("Fail");
                return;
            }
            frameSize = pixelCount * 2;
            break;
        case BMP24_BGR:
        case BMP24_RGB:
            frameSize = pixelCount * 3;
            break;
        case ABMP32_RGB:
            frameSize = pixelCount * 4;
            break;
        case BAYER10_PACKED:
        case RGBIR16:
            if ((width % 4U) != 0 || (height & 1U))
            {
                ui->Source_img_integrity->setText("Fail");
                return;
            }
            frameSize = (pix_fmt == BAYER10_PACKED) ? pixelCount * 5 / 4 : pixelCount * 2;
            break;
        case BAYER_BGGR:
        case BAYER_GBRG:
        case BAYER_GRBG:
        case BAYER_RGGB:
            if (width < 2 || height < 2 ||
                    (bpp_container != 8 && bpp_container != 16) || bpp == 0 ||
                    bpp > bpp_container || bpp_shift > bpp_container ||
                    bpp > bpp_container - bpp_shift)
            {
                ui->Source_img_integrity->setText("Fail");
                return;
            }
            frameSize = pixelCount * (bpp_container / 8);
            break;
        default:
            if (bpp_container == 0 || (bpp_container % 8) != 0)
            {
                ui->Source_img_integrity->setText("Fail");
                return;
            }
            frameSize = pixelCount * (bpp_container / 8);
            break;
    }
    if (frameSize == 0 || frameSize > static_cast<quint64>(INT_MAX) ||
            pixelCount > static_cast<quint64>(SIZE_MAX / 3))
    {
        ui->Source_img_integrity->setText("Fail");
        return;
    }

    QFile input(ui->file_path->text());
    if (!input.open(QIODevice::ReadOnly))
    {
        QMessageBox::warning(this, "Input file", "Cannot open the input file for reading.");
        ui->Source_img_integrity->setText("Fail");
        return;
    }

    const quint64 fileLength = static_cast<quint64>(input.size());
    const quint64 stride = frame_stride ? static_cast<quint64>(frame_stride) : frameSize;
    if (stride < frameSize)
    {
        ui->Source_img_integrity->setText("Fail");
        return;
    }
    const quint64 frameCount = fileLength / stride;
    ui->file_size->setText(QString::number(fileLength));
    ui->num_frames->setText(QString::number(frameCount));
    ui->Source_img_integrity->setText((fileLength % stride) == 0 ? "Pass" : "Fail");
    const QSignalBlocker blocker(ui->src_img_count);
    ui->src_img_count->setRange(0, frameCount ? static_cast<int>(qMin<quint64>(frameCount - 1, INT_MAX)) : 0);
    if (frameCount == 0)
        return;

    const quint64 frameIndex = static_cast<quint64>(ui->src_img_count->value());
    if (!input.seek(static_cast<qint64>(frameIndex * stride)))
    {
        ui->Source_img_integrity->setText("Fail");
        return;
    }
    free(src_buffer);
    src_buffer = static_cast<uint8_t *>(malloc(static_cast<size_t>(frameSize)));
    if (!src_buffer)
    {
        QMessageBox::warning(this, "Memory", "Unable to allocate the source frame buffer.");
        return;
    }
    size = static_cast<int>(input.read(reinterpret_cast<char *>(src_buffer), static_cast<qint64>(frameSize)));
    if (size != static_cast<int>(frameSize))
    {
        ui->Source_img_integrity->setText("Fail");
        return;
    }
    free(des_buffer);
    des_buffer = static_cast<uint8_t *>(calloc(static_cast<size_t>(pixelCount), 3));
    if (!des_buffer)
    {
        QMessageBox::warning(this, "Memory", "Unable to allocate the output image buffer.");
        return;
    }
    std::vector<uint8_t> normalizedBayer;
    if (isBayer8Or16Format(pix_fmt))
    {
        try
        {
            normalizedBayer.resize(static_cast<size_t>(pixelCount));
        }
        catch (const std::bad_alloc &)
        {
            QMessageBox::warning(this, "Memory", "Unable to allocate the Bayer conversion buffer.");
            return;
        }
        if (!normalizeBayerSamples(src_buffer, normalizedBayer.data(),
                static_cast<size_t>(pixelCount), bpp_container, bpp, bpp_shift))
        {
            ui->Source_img_integrity->setText("Fail");
            QMessageBox::warning(this, "Bayer format",
                    "BPP, container size, and shift do not describe a valid sample layout.");
            return;
        }
    }
    switch(pix_fmt)
    {
        case Y8:
            {
                convert_y8_rgb888(src_buffer, des_buffer, width, height);
            }break;

        case Y16:
            {
                convert_y16_rgb888((uint16_t *)src_buffer, des_buffer, width, height);
            }break;

        case UYVY:
            {
                printf("Width %d Height %d \n", width, height);
                convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 2);
                save_asyuv(des_buffer,width, height);
            }break;

        case YVYU:
            {
                convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 1);
                save_buffer(des_buffer, width*height*3);
            }break;

        case YUYV:
            {
                convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 0);
            }break;

        case VYUY:
            {
                convert_yuyv_rgb888(src_buffer, des_buffer, width, height, 3);
            }break;

        case YUV422P_YUV:
            {
                convert_yuy422p_rgb888(src_buffer, des_buffer, width, height);
            }break;

        case YUV420:
            {
                convert_yuy420p_rgb888(src_buffer, des_buffer, width, height);
            }break;

        case NV12:
            {
                convert_nv12_rgb888(src_buffer, des_buffer, width, height);
            }break;

        case RGB444_BGGR:
            {
                convert_rgb444_888(src_buffer, des_buffer, width, height);
            }break;

        case RGB555_RGGB:
            {
                convert_rgb555_888(src_buffer, des_buffer, width, height, 0);
            }break;

        case RGB565_RGGB:
            {
                convert_rgb565_888(src_buffer, des_buffer, width, height, 0);
            }break;

        case RGB565_GBRG:
            {
                convert_rgb565_888(src_buffer, des_buffer, width, height, 1);
            }break;

        case RGB565_BGGR:
            {
                convert_rgb565_888(src_buffer, des_buffer, width, height, 2);
            }break;

        case RGB565_GRBG:
            {
                convert_rgb565_888(src_buffer, des_buffer, width, height, 3);
            }break;

        case BAYER_BGGR:
            {
                dc1394_bayer_decoding_8bit(normalizedBayer.data(), des_buffer, width, height,
                    DC1394_COLOR_FILTER_BGGR, DC1394_BAYER_METHOD_BILINEAR);
            }break;

        case BAYER_GBRG:
            {
                dc1394_bayer_decoding_8bit(normalizedBayer.data(), des_buffer, width, height,
                    DC1394_COLOR_FILTER_GBRG, DC1394_BAYER_METHOD_BILINEAR);
            }break;

        case BAYER_RGGB:
            {
                dc1394_bayer_decoding_8bit(normalizedBayer.data(), des_buffer, width, height,
                    DC1394_COLOR_FILTER_RGGB, DC1394_BAYER_METHOD_BILINEAR);
            }break;

        case BAYER_GRBG:
            {
                dc1394_bayer_decoding_8bit(normalizedBayer.data(), des_buffer, width, height,
                    DC1394_COLOR_FILTER_GRBG, DC1394_BAYER_METHOD_BILINEAR);
            }break;

        case RCCG:
            {
                convert_rccg_rgb24(src_buffer, des_buffer, width, height, 0, bpp_container, bpp_shift);
            }break;

        case BMP24_BGR:
            {
                convert_bgr888_rgb888(src_buffer, des_buffer, width, height);
            }break;

        case BMP24_RGB:
            {
                memcpy(des_buffer, src_buffer, width * height * 3);
            }break;

        case ABMP32_RGB:
            {
                convert_argb32_rgb(src_buffer, des_buffer, width, height);
            }break;

        case BAYER10_PACKED:
            {
                uint8_t *src_buffer1 = (uint8_t *)calloc(width * height, 1);
                if (!src_buffer1)
                    return;
                convert_bayer10_packed_rgbir(src_buffer, src_buffer1, width, height);
                dc1394_bayer_decoding_8bit(src_buffer1, des_buffer, width, height,
                    DC1394_COLOR_FILTER_GRBG, DC1394_BAYER_METHOD_BILINEAR);
                //dc1394_bayer_decoding_8bit(src_buffer1, des_buffer,  width, height,
                //   DC1394_COLOR_FILTER_BGGR, DC1394_BAYER_METHOD_SIMPLE);
                save_asyuv(des_buffer,width, height);
                free(src_buffer1);

                uint8_t *src_ir = (uint8_t *)calloc((width * height)/4, 1);
                if (!src_ir)
                    return;
                extract_bayer10_packed_ir(src_buffer, src_ir, width, height);
                save_ir_asyuv(src_ir,width/2, height/2);
                free(src_ir);
                break;
            }

        case RGBIR16:
            {
                uint8_t *src_buffer1 = (uint8_t *)calloc(width * height, 1);
                if (!src_buffer1)
                    return;
                convert_RGBIR16_bayer8(src_buffer, src_buffer1, width, height);
                dc1394_bayer_decoding_8bit(src_buffer1, des_buffer, width, height,
                    DC1394_COLOR_FILTER_GBRG, DC1394_BAYER_METHOD_BILINEAR);
                perform_equalize_rgb24 (des_buffer, width, height);
                save_asyuv(des_buffer,width, height);
                uint8_t *src_ir = (uint8_t *)calloc((width * height)/4, 1);
                if (!src_ir)
                {
                    free(src_buffer1);
                    return;
                }
                extract_RGBIR16_IR8(src_buffer, src_ir, width, height);
                save_ir_asyuv(src_ir,width/2, height/2);
                free(src_ir);
                free(src_buffer1);
                break;
            }
        case BGGR16:
            {
                uint8_t *src_buffer1 = (uint8_t *)calloc(width * height, 1);
                convert_bit16_bit8((uint16_t *)src_buffer, src_buffer1, width, height);
                dc1394_bayer_decoding_8bit(src_buffer1, des_buffer, width, height,
                    DC1394_COLOR_FILTER_BGGR, DC1394_BAYER_METHOD_BILINEAR);
                save_asyuv(des_buffer,width, height);
                free(src_buffer1);
                break;
            }
    }

    QImage image(static_cast<int>(width), static_cast<int>(height), QImage::Format_RGB888);
    if (image.isNull())
        return;
    for (unsigned int row = 0; row < height; ++row)
        memcpy(image.scanLine(static_cast<int>(row)), des_buffer + row * width * 3, width * 3);
    ui->draw_window->setPixmap(QPixmap::fromImage(image));
    ui->draw_window->update();
    ui->modified_draw_window->clear();
    showImagePane();
}

void convert::on_pixel_fmt_currentIndexChanged(int32_t index)
{
    char tmp_buffer[50];

    if (index < 0 || index >= static_cast<int>(sizeof(img_pix_fmt) / sizeof(img_pix_fmt[0])))
        return;
    pix_fmt = static_cast<enum pix_fmt>(index);
    memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
    sprintf(tmp_buffer, "%d", img_pix_fmt[pix_fmt].bpp_container);
    ui->bpp_container->setText(QString::fromStdString(tmp_buffer));

    memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
    sprintf(tmp_buffer, "%d", img_pix_fmt[pix_fmt].bpp);
    ui->bpp->setText(QString::fromStdString(tmp_buffer));

    memset(tmp_buffer, 0x00, sizeof(tmp_buffer));
    sprintf(tmp_buffer, "%d", img_pix_fmt[pix_fmt].bpp_shift);
    ui->bpp_shift->setText(QString::fromStdString(tmp_buffer));

    width = atoi(ui->width->text().toLocal8Bit().data());
    height = atoi(ui->height->text().toLocal8Bit().data());
    frame_stride = atoi(ui->frame_stride->text().toLocal8Bit().data());
    bpp = atoi(ui->bpp->text().toLocal8Bit().data());
    bpp_container = atoi(ui->bpp_container->text().toLocal8Bit().data());
    bpp_shift = atoi(ui->bpp_shift->text().toLocal8Bit().data());
    paintimage();
}

void convert::on_src_img_count_valueChanged(int32_t)
{
    paintimage();
}

void convert::on_width_editingFinished()
{
    paintimage();
}

void convert::on_height_editingFinished()
{
    paintimage();
}

void convert::on_frame_stride_editingFinished()
{
    paintimage();
}

void convert::on_file_path_editingFinished()
{
    paintimage();
}

void convert::on_equalize_clicked()
{
    if (!des_buffer || width == 0 || height == 0)
        return;

    const size_t rowSize = static_cast<size_t>(width) * 3;
    std::vector<uint8_t> modifiedBuffer(rowSize * height);
    QImage image(static_cast<int>(width), static_cast<int>(height), QImage::Format_RGB888);
    if (image.isNull())
        return;
    for (unsigned int row = 0; row < height; ++row)
        memcpy(modifiedBuffer.data() + row * rowSize, des_buffer + row * rowSize, rowSize);
    if (perform_equalize_rgb24(modifiedBuffer.data(), width, height) < 0)
        return;
    for (unsigned int row = 0; row < height; ++row)
        memcpy(image.scanLine(static_cast<int>(row)), modifiedBuffer.data() + row * rowSize, rowSize);

    ui->modified_draw_window->setPixmap(QPixmap::fromImage(image));
    ui->modified_draw_window->update();
    showImagePane();
}

void convert::showImagePane()
{
    if (analysisPage)
    {
        rightPane->removeWidget(analysisPage);
        analysisPage->deleteLater();
        analysisPage = nullptr;
    }
    rightPane->setCurrentWidget(ui->modified_draw_window);
}

void convert::showAnalysisPane(QWidget *content, const QString &title)
{
    showImagePane();
    QWidget *page = new QWidget(rightPane);
    QVBoxLayout *pageLayout = new QVBoxLayout(page);
    pageLayout->setContentsMargins(4, 4, 4, 4);
    QHBoxLayout *toolbar = new QHBoxLayout();
    QLabel *pageTitle = new QLabel(title, page);
    QPushButton *imageButton = new QPushButton(tr("Back"), page);
    toolbar->addWidget(pageTitle);
    toolbar->addStretch();
    toolbar->addWidget(imageButton);
    pageLayout->addLayout(toolbar);
    content->setParent(page);
    pageLayout->addWidget(content, 1);
    connect(imageButton, &QPushButton::clicked, this, [this]() { showImagePane(); });
    analysisPage = page;
    rightPane->addWidget(page);
    rightPane->setCurrentWidget(page);
}

void convert::on_lens_shading_clicked()
{
    if (!des_buffer || width == 0 || height == 0)
    {
        QMessageBox::information(this, tr("Lens shading"), tr("Load an image first."));
        return;
    }

    showAnalysisPane(new LensShadingPlotWidget(des_buffer, width, height), tr("Lens Shading Contour"));
}

void convert::on_ae_tuning_clicked()
{
    if (!des_buffer || width == 0 || height == 0)
    {
        QMessageBox::information(this, tr("AE tuning"), tr("Load an image first."));
        return;
    }

    showAnalysisPane(new AETuningPlotWidget(des_buffer, width, height), tr("AE Tuning Plots"));
}

void convert::on_dynamic_range_clicked()
{
    if (!des_buffer || width == 0 || height == 0)
    {
        QMessageBox::information(this, tr("Dynamic range"), tr("Load an image first."));
        return;
    }

    showAnalysisPane(new DynamicRangePlotWidget(des_buffer, width, height), tr("Dynamic Range Analysis"));
}

void convert::on_awb_ccm_clicked()
{
    if (!des_buffer || width == 0 || height == 0)
    {
        QMessageBox::information(this, tr("AWB & CCM"), tr("Load an image first."));
        return;
    }

    bool centerXOk = false;
    bool centerYOk = false;
    bool roiWidthOk = false;
    bool roiHeightOk = false;
    const int centerX = ui->crop_x->text().toInt(&centerXOk);
    const int centerY = ui->crop_y->text().toInt(&centerYOk);
    const int roiWidth = ui->crop_width->text().toInt(&roiWidthOk);
    const int roiHeight = ui->crop_height->text().toInt(&roiHeightOk);
    if (!centerXOk || !centerYOk || !roiWidthOk || !roiHeightOk ||
            roiWidth < 6 || roiHeight < 4)
    {
        QMessageBox::information(this, tr("AWB & CCM"),
                tr("Set X and Y to the chart center and Width and Height to the full 6 x 4 chart area, then run AWB & CCM."));
        return;
    }

    const qint64 roiLeft = static_cast<qint64>(centerX) - roiWidth / 2;
    const qint64 roiTop = static_cast<qint64>(centerY) - roiHeight / 2;
    if (roiLeft < 0 || roiTop < 0 || roiLeft + roiWidth > width || roiTop + roiHeight > height)
    {
        QMessageBox::warning(this, tr("AWB & CCM"),
                tr("The chart ROI must fit completely inside the image."));
        return;
    }

        AWBCCMPlotWidget *plot = new AWBCCMPlotWidget(des_buffer, width, height,
            static_cast<unsigned int>(roiLeft), static_cast<unsigned int>(roiTop),
            static_cast<unsigned int>(roiWidth), static_cast<unsigned int>(roiHeight));
        QTabWidget *analysisTabs = new QTabWidget();
        analysisTabs->addTab(plot, tr("Lab Comparison"));
        analysisTabs->addTab(new ColorPatchRgbWidget(plot->samples()), tr("RGB Values"));
        showAnalysisPane(analysisTabs, tr("AWB & CCM Analysis"));
}


void convert::on_lens_distortion_clicked()
{
    if (!des_buffer || width == 0 || height == 0)
    {
        QMessageBox::information(this, tr("Lens distortion correction"), tr("Load an image first."));
        return;
    }

    QSettings settings(QSettings::IniFormat, QSettings::UserScope,
            QStringLiteral("Ananth"), QStringLiteral("convertor_simple"));
    QWidget *page = new QWidget();
    QVBoxLayout *layout = new QVBoxLayout(page);
    QLabel *description = new QLabel(tr("OpenCV-compatible fisheye calibration model. Intrinsics are in pixels; use coefficients from camera calibration."), page);
    description->setWordWrap(true);
    layout->addWidget(description);

    QFormLayout *form = new QFormLayout();
    auto createParameter = [page, &settings, form](const QString &label, const QString &key, double fallback) {
        QLineEdit *edit = new QLineEdit(settings.value(key, fallback).toString(), page);
        edit->setObjectName(key);
        QObject::connect(edit, &QLineEdit::editingFinished, edit, [key, edit]() {
            QSettings saved(QSettings::IniFormat, QSettings::UserScope,
                    QStringLiteral("Ananth"), QStringLiteral("convertor_simple"));
            saved.setValue(key, edit->text());
            saved.sync();
        });
        form->addRow(label, edit);
        return edit;
    };

    QLineEdit *fxEdit = createParameter(tr("fx"), QStringLiteral("ldc/fx"), width * 0.5);
    QLineEdit *fyEdit = createParameter(tr("fy"), QStringLiteral("ldc/fy"), width * 0.5);
    QLineEdit *cxEdit = createParameter(tr("cx"), QStringLiteral("ldc/cx"), width * 0.5);
    QLineEdit *cyEdit = createParameter(tr("cy"), QStringLiteral("ldc/cy"), height * 0.5);
    QLineEdit *k1Edit = createParameter(tr("k1"), QStringLiteral("ldc/k1"), 0.0);
    QLineEdit *k2Edit = createParameter(tr("k2"), QStringLiteral("ldc/k2"), 0.0);
    QLineEdit *k3Edit = createParameter(tr("k3"), QStringLiteral("ldc/k3"), 0.0);
    QLineEdit *k4Edit = createParameter(tr("k4"), QStringLiteral("ldc/k4"), 0.0);
    layout->addLayout(form);

    QPushButton *applyButton = new QPushButton(tr("Apply LDC"), page);
    layout->addWidget(applyButton);
    QLabel *correctedView = new QLabel(page);
    correctedView->setAlignment(Qt::AlignCenter);
    correctedView->setScaledContents(true);
    correctedView->setMinimumSize(320, 220);
    correctedView->setText(tr("Enter calibrated intrinsics and distortion coefficients, then apply."));
    layout->addWidget(correctedView, 1);

    connect(applyButton, &QPushButton::clicked, this,
            [this, fxEdit, fyEdit, cxEdit, cyEdit, k1Edit, k2Edit, k3Edit, k4Edit, correctedView]() {
        bool valid = true;
        auto readValue = [&valid](QLineEdit *edit) {
            bool parsed = false;
            const double value = edit->text().toDouble(&parsed);
            valid = valid && parsed && std::isfinite(value);
            return value;
        };
        const double fx = readValue(fxEdit);
        const double fy = readValue(fyEdit);
        const double cx = readValue(cxEdit);
        const double cy = readValue(cyEdit);
        const double k1 = readValue(k1Edit);
        const double k2 = readValue(k2Edit);
        const double k3 = readValue(k3Edit);
        const double k4 = readValue(k4Edit);
        if (!valid || fx <= 0 || fy <= 0)
        {
            QMessageBox::warning(this, tr("Lens distortion correction"),
                    tr("Enter finite numeric values and positive fx/fy focal lengths."));
            return;
        }

        const size_t pixelCount = static_cast<size_t>(width) * height;
        std::vector<uint8_t> corrected;
        try
        {
            corrected.resize(pixelCount * 3);
        }
        catch (const std::bad_alloc &)
        {
            QMessageBox::warning(this, tr("Lens distortion correction"),
                    tr("Unable to allocate the corrected image buffer."));
            return;
        }
        if (convert_fisheye_undistort_rgb24(des_buffer, corrected.data(), width, height,
                fx, fy, cx, cy, k1, k2, k3, k4) != 0)
        {
            QMessageBox::warning(this, tr("Lens distortion correction"),
                    tr("The fisheye remap parameters are invalid."));
            return;
        }

        QImage correctedImage(static_cast<int>(width), static_cast<int>(height), QImage::Format_RGB888);
        if (correctedImage.isNull())
            return;
        for (unsigned int row = 0; row < height; ++row)
            memcpy(correctedImage.scanLine(static_cast<int>(row)),
                    corrected.data() + static_cast<size_t>(row) * width * 3, width * 3);
        correctedView->setPixmap(QPixmap::fromImage(correctedImage));

        QSettings saved(QSettings::IniFormat, QSettings::UserScope,
                QStringLiteral("Ananth"), QStringLiteral("convertor_simple"));
        saved.setValue("ldc/fx", fxEdit->text());
        saved.setValue("ldc/fy", fyEdit->text());
        saved.setValue("ldc/cx", cxEdit->text());
        saved.setValue("ldc/cy", cyEdit->text());
        saved.setValue("ldc/k1", k1Edit->text());
        saved.setValue("ldc/k2", k2Edit->text());
        saved.setValue("ldc/k3", k3Edit->text());
        saved.setValue("ldc/k4", k4Edit->text());
        saved.sync();
    });

    showAnalysisPane(page, tr("Lens Distortion Correction"));
}

void convert::on_bpp_editingFinished()
{
    bpp = atoi(ui->bpp->text().toLocal8Bit().data());
    bpp_container = atoi(ui->bpp_container->text().toLocal8Bit().data());
    bpp_shift = atoi(ui->bpp_shift->text().toLocal8Bit().data());
    paintimage();
}

void convert::on_bpp_container_editingFinished()
{
    bpp = atoi(ui->bpp->text().toLocal8Bit().data());
    bpp_container = atoi(ui->bpp_container->text().toLocal8Bit().data());
    bpp_shift = atoi(ui->bpp_shift->text().toLocal8Bit().data());
    paintimage();
}

void convert::on_Crop_clicked()
{
    bool xOk = false;
    bool yOk = false;
    bool cropWidthOk = false;
    bool cropHeightOk = false;
    const int x = ui->crop_x->text().toInt(&xOk);
    const int y = ui->crop_y->text().toInt(&yOk);
    const int cropWidth = ui->crop_width->text().toInt(&cropWidthOk);
    const int cropHeight = ui->crop_height->text().toInt(&cropHeightOk);
    if (!des_buffer || !xOk || !yOk || !cropWidthOk || !cropHeightOk ||
            cropWidth <= 0 || cropHeight <= 0)
    {
        QMessageBox::warning(this, "Crop", "Enter a valid crop center and positive dimensions.");
        return;
    }

    const size_t rowSize = static_cast<size_t>(cropWidth) * 3;
    if (static_cast<size_t>(cropHeight) > SIZE_MAX / rowSize)
        return;
    std::vector<uint8_t> cropBuffer(rowSize * cropHeight);
    if (perform_crop(cropBuffer.data(), des_buffer, static_cast<uint32_t>(x),
            static_cast<uint32_t>(y), static_cast<uint32_t>(cropWidth),
            static_cast<uint32_t>(cropHeight), 3, width, height) < 0)
    {
        QMessageBox::warning(this, "Crop", "The requested crop is outside the image.");
        return;
    }

    QImage sourceImage(static_cast<int>(width), static_cast<int>(height), QImage::Format_RGB888);
    QImage croppedImage(cropWidth, cropHeight, QImage::Format_RGB888);
    if (sourceImage.isNull() || croppedImage.isNull())
        return;
    for (unsigned int row = 0; row < height; ++row)
        memcpy(sourceImage.scanLine(static_cast<int>(row)), des_buffer + row * width * 3, width * 3);

    QPainter painter(&sourceImage);
    QPen pen;
    pen.setWidth(1);
    pen.setColor(Qt::red);
    painter.setPen(pen);
    painter.drawRect(x - cropWidth / 2, y - cropHeight / 2, cropWidth, cropHeight);
    ui->draw_window->setPixmap(QPixmap::fromImage(sourceImage));
    ui->draw_window->update();

    perform_equalize_rgb24(cropBuffer.data(), cropWidth, cropHeight);
    for (int row = 0; row < cropHeight; ++row)
        memcpy(croppedImage.scanLine(row), cropBuffer.data() + row * rowSize, rowSize);

    ui->modified_draw_window->setPixmap(QPixmap::fromImage(croppedImage));
    ui->modified_draw_window->update();
    showImagePane();
}

void convert::on_bpp_shift_editingFinished()
{
    bpp = atoi(ui->bpp->text().toLocal8Bit().data());
    bpp_container = atoi(ui->bpp_container->text().toLocal8Bit().data());
    bpp_shift = atoi(ui->bpp_shift->text().toLocal8Bit().data());
    paintimage();
}


void convert::on_Save_clicked()
{
    if (!des_buffer || width == 0 || height == 0)
        return;

    const QString path = QFileDialog::getSaveFileName(this, "Save image", "Sample.bmp",
            "Bitmap images (*.bmp)");
    if (path.isEmpty())
        return;

    QImage image(static_cast<int>(width), static_cast<int>(height), QImage::Format_RGB888);
    if (image.isNull())
        return;
    for (unsigned int row = 0; row < height; ++row)
        memcpy(image.scanLine(static_cast<int>(row)), des_buffer + row * width * 3, width * 3);
    if (!image.save(path, "BMP"))
        QMessageBox::warning(this, "Save image", "The image could not be saved.");
}

