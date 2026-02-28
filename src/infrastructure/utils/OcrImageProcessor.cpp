#include "OcrImageProcessor.hpp"
#include <cmath>
#include <algorithm>

namespace ocr::infrastructure {

cv::Mat OcrImageProcessor::preprocessDet(const cv::Mat& src, const OcrConfig& config, float& out_ratio_h, float& out_ratio_w) {
    int h = src.rows;
    int w = src.cols;
    if (h <= 0 || w <= 0) return cv::Mat();

    float ratio = 1.0f;
    if (std::max(h, w) > config.maxSideLen) {
        ratio = (float)config.maxSideLen / (float)std::max(h, w);
    }

    int resize_h = (int)std::max(32.0, std::round((double)h * ratio / 32.0) * 32.0);
    int resize_w = (int)std::max(32.0, std::round((double)w * ratio / 32.0) * 32.0);

    out_ratio_h = (float)resize_h / (float)h;
    out_ratio_w = (float)resize_w / (float)w;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resize_w, resize_h));
    
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);
    
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std_dev(0.229, 0.224, 0.225);
    floatImg -= mean;
    floatImg /= std_dev;
    return floatImg;
}

std::vector<domain::BoundingBox> OcrImageProcessor::postprocessDet(const cv::Mat& heatmap, const cv::Mat& src, const OcrConfig& config, float r_h, float r_w) {
    cv::Mat binary;
    cv::threshold(heatmap, binary, config.detThreshold, 1.0, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8UC1, 255);
    cv::dilate(binary, binary, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<domain::BoundingBox> results;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) < 8) continue;
        cv::RotatedRect rect = cv::minAreaRect(contour);
        float w = rect.size.width;
        float h = rect.size.height;
        if (w <= 0 || h <= 0) continue;

        float offset = (w * h) * config.unclipRatio / (2 * (w + h));
        rect.size.width += 2 * offset;
        rect.size.height += 2 * offset;
        if (rect.size.width < 3 || rect.size.height < 3) continue;

        cv::Point2f pts[4];
        rect.points(pts);
        std::vector<domain::Point> points;
        bool ok = true;
        for (int i = 0; i < 4; ++i) {
            float px = pts[i].x / r_w;
            float py = pts[i].y / r_h;
            if (!std::isfinite(px) || !std::isfinite(py)) { ok = false; break; }
            px = std::max(0.0f, std::min(px, (float)src.cols - 1));
            py = std::max(0.0f, std::min(py, (float)src.rows - 1));
            points.push_back({px, py});
        }
        if (ok && points.size() == 4) results.emplace_back(points, 1.0f);
    }
    return results;
}

cv::Mat OcrImageProcessor::preprocessRec(const cv::Mat& boxImg, const OcrConfig& config) {
    if (boxImg.empty() || boxImg.cols <= 0 || boxImg.rows <= 0) return cv::Mat();

    int targetH = config.recTargetH;
    float scale = (float)targetH / (float)boxImg.rows;
    int targetW = (int)std::max(1.0f, (float)boxImg.cols * scale);
    if (targetW > 8000) targetW = 8000;

    cv::Mat resized;
    cv::resize(boxImg, resized, cv::Size(targetW, targetH), 0, 0, cv::INTER_CUBIC);

    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);
    cv::Scalar mean(0.5, 0.5, 0.5), std_v(0.5, 0.5, 0.5);
    floatImg -= mean;
    floatImg /= std_v;
    return floatImg;
}

cv::Mat OcrImageProcessor::enhanceCrop(const cv::Mat& crop, const OcrConfig& config) {
    if (crop.empty()) return crop;

    cv::Mat lab;
    cv::cvtColor(crop, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labChannels;
    cv::split(lab, labChannels);

    if (config.enableClahe) {
        auto clahe = cv::createCLAHE(3.0, cv::Size(4, 4));
        clahe->apply(labChannels[0], labChannels[0]);
    }

    cv::Mat blurred;
    cv::GaussianBlur(labChannels[0], blurred, cv::Size(0, 0), 1.5);
    cv::addWeighted(labChannels[0], 1.8, blurred, -0.8, 0, labChannels[0]);

    cv::merge(labChannels, lab);
    cv::Mat result;
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    return result;
}

cv::Mat OcrImageProcessor::deskewCrop(const cv::Mat& crop, const OcrConfig& config) {
    if (crop.empty() || crop.cols < 10 || crop.rows < 10) return crop;

    cv::Mat gray;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    cv::Moments m   = cv::moments(binary, true);
    double angle    = 0.0;
    if (std::abs(m.mu11) > 1e-6 || std::abs(m.mu20 - m.mu02) > 1e-6) {
        angle = 0.5 * std::atan2(2.0 * m.mu11, m.mu20 - m.mu02) * 180.0 / CV_PI;
    }

    if (std::abs(angle) < 0.5 || std::abs(angle) > 15.0) return crop;

    cv::Point2f center((float)crop.cols / 2.0f, (float)crop.rows / 2.0f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat deskewed;
    cv::warpAffine(crop, deskewed, rotMat, crop.size(),
                   cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return deskewed;
}

void OcrImageProcessor::applyGlobalEnhancements(cv::Mat& src, const OcrConfig& config) {
    if (config.enableClahe) {
        cv::Mat gray, enhanced;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray, enhanced);
        cv::cvtColor(enhanced, src, cv::COLOR_GRAY2BGR);
    }
}

} // namespace ocr::infrastructure
