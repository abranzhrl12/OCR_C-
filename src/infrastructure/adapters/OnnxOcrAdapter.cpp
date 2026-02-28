#include "OnnxOcrAdapter.hpp"
#include "../utils/OcrImageProcessor.hpp"
#include "../utils/PlatformUtils.hpp"
#include "../utils/OnnxSessionHelper.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <thread>
#include <cmath>
#include <cstring>

namespace ocr::infrastructure {
    
Ort::Env* OnnxOcrAdapter::env_ = nullptr;

OnnxOcrAdapter::OnnxOcrAdapter(const OcrConfig& config)
    : config_(config), 
      detSession_(nullptr),
      recSession_(nullptr) {
    
    // El ScopeLogSilencer restaurará los logs automáticamente al salir de este constructor
    ScopedLogSilencer silencer;
    
    try {
        if (!env_) {
            env_ = new Ort::Env(ORT_LOGGING_LEVEL_FATAL, "OcrLib");
        }

        // Inicializar sesiones usando el Helper centralizado
        detSession_ = OnnxSessionHelper::createSession(*env_, config.detModelPath);
        recSession_ = OnnxSessionHelper::createSession(*env_, config.recModelPath);

    } catch (const std::exception& e) {
        std::cerr << "[OCR_ADAPTER] Error fatídico en inicialización: " << e.what() << std::endl;
        throw;
    }

    loadDictionary(config.dictPath);
}


void OnnxOcrAdapter::loadDictionary(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    dict_.push_back("blank");
    while (std::getline(file, line)) {
        dict_.push_back(line);
    }
    dict_.push_back(" ");
}

domain::Result<std::vector<domain::TextBlock>> OnnxOcrAdapter::detectAndRecognize(const std::string& imagePath) {
    try {
        cv::Mat src = cv::imread(imagePath);
        if (src.empty()) return domain::Result<std::vector<domain::TextBlock>>::Fail("Could not load image: " + imagePath);
        return domain::Result<std::vector<domain::TextBlock>>::Ok(processMat(src));
    } catch (const std::exception& e) {
        return domain::Result<std::vector<domain::TextBlock>>::Fail(e.what());
    }
}

domain::Result<std::vector<domain::TextBlock>> OnnxOcrAdapter::detectAndRecognize(const unsigned char* buffer, size_t size) {
    try {
        if (!buffer || size == 0) return domain::Result<std::vector<domain::TextBlock>>::Fail("Invalid buffer or size zero");
        cv::Mat rawData(1, static_cast<int>(size), CV_8UC1, const_cast<unsigned char*>(buffer));
        cv::Mat src = cv::imdecode(rawData, cv::IMREAD_COLOR);
        if (src.empty()) return domain::Result<std::vector<domain::TextBlock>>::Fail("Could not decode buffer");
        return domain::Result<std::vector<domain::TextBlock>>::Ok(processMat(src));
    } catch (const std::exception& e) {
        return domain::Result<std::vector<domain::TextBlock>>::Fail(e.what());
    }
}

domain::Result<std::vector<domain::TextBlock>> OnnxOcrAdapter::detectAndRecognize(const cv::Mat& src) {

    try {
        if (src.empty()) return domain::Result<std::vector<domain::TextBlock>>::Fail("Empty OpenCV matrix");
        return domain::Result<std::vector<domain::TextBlock>>::Ok(processMat(src));
    } catch (const std::exception& e) {
        return domain::Result<std::vector<domain::TextBlock>>::Fail(e.what());
    }
}


std::vector<domain::TextBlock> OnnxOcrAdapter::processMat(const cv::Mat& src) {
    // Si necesitamos modificar la imagen globalmente, trabajamos sobre una copia local
    // para no alterar la imagen original del llamante. Si no, usamos la referencia.
    cv::Mat processingImg;
    if (config_.enableClahe) {
        processingImg = src.clone();
        OcrImageProcessor::applyGlobalEnhancements(processingImg, config_);
    } else {
        processingImg = src; // Copia superficial (puntero)
    }

    auto boxes = detect(processingImg);
    std::vector<domain::TextBlock> results;
    results.reserve(boxes.size());

    for (const auto& box : boxes) {
        auto points = box.getPoints();
        std::vector<cv::Point> cvPoints;
        for (auto& p : points) cvPoints.push_back({(int)p.x, (int)p.y});
        cv::Rect rect = cv::boundingRect(cvPoints);
        rect &= cv::Rect(0, 0, processingImg.cols, processingImg.rows);
        if (rect.width <= 3 || rect.height <= 3) continue;

        cv::Mat crop = processingImg(rect).clone();
        if (crop.empty()) continue;

        if (crop.rows > crop.cols * 1.5) cv::rotate(crop, crop, cv::ROTATE_90_CLOCKWISE);

        if (config_.enableDeskew) crop = OcrImageProcessor::deskewCrop(crop, config_);
        crop = OcrImageProcessor::enhanceCrop(crop, config_);

        float confidence = 0.0f;
        std::string text = recognize(crop, confidence);
        if (!text.empty() && confidence > config_.minConfidence) {
            results.emplace_back(box, text, confidence);
        }
    }
    return results;
}


std::vector<domain::BoundingBox> OnnxOcrAdapter::detect(const cv::Mat& src) {
    float r_h = 1.0f, r_w = 1.0f;
    cv::Mat inputImg = OcrImageProcessor::preprocessDet(src, config_, r_h, r_w);
    if (inputImg.empty()) return {};

    std::vector<int64_t> inputShape = {1, 3, inputImg.rows, inputImg.cols};
    size_t vecSize = 1 * 3 * inputImg.rows * inputImg.cols;
    std::vector<float> inputValues(vecSize);
    
    std::vector<cv::Mat> channels(3);
    cv::split(inputImg, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(inputValues.data() + (c * inputImg.rows * inputImg.cols), 
                    channels[c].data, inputImg.rows * inputImg.cols * sizeof(float));
    }

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputValues.data(), inputValues.size(), inputShape.data(), inputShape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto inputName = detSession_.GetInputNameAllocated(0, allocator);
    auto outputName = detSession_.GetOutputNameAllocated(0, allocator);
    const char* inNames[] = { inputName.get() };
    const char* outNames[] = { outputName.get() };

    auto outputTensors = detSession_.Run(Ort::RunOptions{nullptr}, inNames, &inputTensor, 1, outNames, 1);
    float* floatData = outputTensors[0].GetTensorMutableData<float>();
    cv::Mat heatmap(inputImg.rows, inputImg.cols, CV_32FC1, floatData);
    return OcrImageProcessor::postprocessDet(heatmap, src, config_, r_h, r_w);
}

std::string OnnxOcrAdapter::recognize(const cv::Mat& boxImg, float& confidence) {
    cv::Mat inputImg = OcrImageProcessor::preprocessRec(boxImg, config_);
    if (inputImg.empty()) return "";

    std::vector<int64_t> shape = {1, 3, inputImg.rows, inputImg.cols};
    std::vector<float> values(1 * 3 * inputImg.rows * inputImg.cols);
    std::vector<cv::Mat> channels(3);
    cv::split(inputImg, channels);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(values.data() + (c * inputImg.rows * inputImg.cols),
                    channels[c].data,
                    inputImg.rows * inputImg.cols * sizeof(float));
    }

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, values.data(), values.size(), shape.data(), shape.size());
    Ort::AllocatorWithDefaultOptions allocator;
    auto inName  = recSession_.GetInputNameAllocated(0, allocator);
    auto outName = recSession_.GetOutputNameAllocated(0, allocator);
    const char* inNames[]  = { inName.get() };
    const char* outNames[] = { outName.get() };

    auto outputTensors = recSession_.Run(
        Ort::RunOptions{nullptr}, inNames, &inputTensor, 1, outNames, 1);
    float* data     = outputTensors[0].GetTensorMutableData<float>();
    auto   outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t T = outShape[1], C = outShape[2];

    struct FramePred { int idx; float prob; };
    std::vector<FramePred> preds;
    preds.reserve((size_t)T);

    for (int t = 0; t < T; ++t) {
        float* row = data + t * C;
        float maxLogit = *std::max_element(row, row + C);
        float sumExp   = 0.0f;
        for (int c = 0; c < C; ++c) sumExp += std::exp(row[c] - maxLogit);
        int   bestIdx  = (int)(std::max_element(row, row + C) - row);
        float bestProb = std::exp(row[bestIdx] - maxLogit) / sumExp;
        preds.push_back({bestIdx, bestProb});
    }

    std::vector<int64_t> indices;
    indices.reserve((size_t)T);
    float totalProb = 0.0f;
    int   count     = 0;
    for (auto& p : preds) {
        indices.push_back(p.idx);
        if (p.idx > 0) { totalProb += p.prob; count++; }
    }
    confidence = count > 0 ? totalProb / count : 0.0f;
    return ctcDecode(indices, confidence);
}

std::string OnnxOcrAdapter::ctcDecode(const std::vector<int64_t>& indices, float& confidence) {
    std::string text;
    int lastIdx = -1;
    for (auto idx : indices) {
        if (idx == 0) {
            lastIdx = 0;
            continue;
        }
        if (idx != lastIdx && idx < (int64_t)dict_.size()) {
            text += dict_[idx];
        }
        lastIdx = (int)idx;
    }
    return text;
}

} // namespace ocr::infrastructure

