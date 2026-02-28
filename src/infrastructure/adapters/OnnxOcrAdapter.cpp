#include "OnnxOcrAdapter.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <thread>
#include <cmath>
#include <cstring>

#ifdef _WIN32
    #include <io.h>
    #include <fcntl.h>
    #define DUP _dup
    #define DUP2 _dup2
    #define FILENO _fileno
    #define CLOSE _close
    #define DEV_NULL "NUL"
#else
    #include <unistd.h>
    #include <fcntl.h>
    #define DUP dup
    #define DUP2 dup2
    #define FILENO fileno
    #define CLOSE close
    #define DEV_NULL "/dev/null"
#endif

namespace ocr::infrastructure {
    
Ort::Env* OnnxOcrAdapter::env_ = nullptr;

OnnxOcrAdapter::OnnxOcrAdapter(const OcrConfig& config)
    : config_(config), 
      detSession_(nullptr),
      recSession_(nullptr) {
    
    int old_stderr = DUP(FILENO(stderr));
    
#ifdef _WIN32
    FILE* nul = nullptr;
    bool silencerOp = (freopen_s(&nul, DEV_NULL, "w", stderr) == 0);
#else
    FILE* nul = freopen(DEV_NULL, "w", stderr);
    bool silencerOp = (nul != nullptr);
#endif

    if (silencerOp) {
        try {
            if (!env_) {
                env_ = new Ort::Env(ORT_LOGGING_LEVEL_FATAL, "OcrLib");
            }
            Ort::SessionOptions sessionOptions;
            sessionOptions.SetLogSeverityLevel(4);
            unsigned int cpuCores = std::thread::hardware_concurrency();
            sessionOptions.SetIntraOpNumThreads(std::max(2, (int)std::sqrt((float)cpuCores)));
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
            std::wstring wDetPath(config.detModelPath.begin(), config.detModelPath.end());
            std::wstring wRecPath(config.recModelPath.begin(), config.recModelPath.end());
            detSession_ = Ort::Session(*env_, wDetPath.c_str(), sessionOptions);
            recSession_ = Ort::Session(*env_, wRecPath.c_str(), sessionOptions);
#else
            detSession_ = Ort::Session(*env_, config.detModelPath.c_str(), sessionOptions);
            recSession_ = Ort::Session(*env_, config.recModelPath.c_str(), sessionOptions);
#endif
        } catch (...) {
            DUP2(old_stderr, FILENO(stderr));
            throw;
        }
        fflush(stderr);
        DUP2(old_stderr, FILENO(stderr));
        CLOSE(old_stderr);
    } else {
        if (!env_) env_ = new Ort::Env(ORT_LOGGING_LEVEL_FATAL, "OcrLib");
        Ort::SessionOptions sessionOptions;
#ifdef _WIN32
        std::wstring wDetPath(config.detModelPath.begin(), config.detModelPath.end());
        std::wstring wRecPath(config.recModelPath.begin(), config.recModelPath.end());
        detSession_ = Ort::Session(*env_, wDetPath.c_str(), sessionOptions);
        recSession_ = Ort::Session(*env_, wRecPath.c_str(), sessionOptions);
#else
        detSession_ = Ort::Session(*env_, config.detModelPath.c_str(), sessionOptions);
        recSession_ = Ort::Session(*env_, config.recModelPath.c_str(), sessionOptions);
#endif
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

std::vector<domain::TextBlock> OnnxOcrAdapter::detectAndRecognize(const std::string& imagePath) {
    cv::Mat src = cv::imread(imagePath);
    if (src.empty()) throw std::runtime_error("Could not load image");
    return processMat(src);
}

std::vector<domain::TextBlock> OnnxOcrAdapter::detectAndRecognize(const unsigned char* buffer, size_t size) {
    cv::Mat rawData(1, static_cast<int>(size), CV_8UC1, const_cast<unsigned char*>(buffer));
    cv::Mat src = cv::imdecode(rawData, cv::IMREAD_COLOR);
    if (src.empty()) throw std::runtime_error("Could not decode buffer");
    return processMat(src);
}

std::vector<domain::TextBlock> OnnxOcrAdapter::processMat(cv::Mat src) {
    // --- Mejora global: CLAHE en escala de grises sobre la imagen fuente ---
    // Esto mejora el contraste general antes de detectar regiones de texto
    if (config_.enableClahe) {
        cv::Mat gray, enhanced;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray, enhanced);
        cv::cvtColor(enhanced, src, cv::COLOR_GRAY2BGR);
    }

    auto boxes = detect(src);
    std::vector<domain::TextBlock> results;
    results.reserve(boxes.size());

    for (const auto& box : boxes) {
        auto points = box.getPoints();
        std::vector<cv::Point> cvPoints;
        for (auto& p : points) cvPoints.push_back({(int)p.x, (int)p.y});
        cv::Rect rect = cv::boundingRect(cvPoints);
        rect &= cv::Rect(0, 0, src.cols, src.rows);
        if (rect.width <= 3 || rect.height <= 3) continue;

        cv::Mat crop = src(rect).clone();
        if (crop.empty()) continue;

        // Rotar si el crop es "vertical" (texto en columna)
        if (crop.rows > crop.cols * 1.5) cv::rotate(crop, crop, cv::ROTATE_90_CLOCKWISE);

        // Corregir inclinación leve del crop
        if (config_.enableDeskew) {
            crop = deskewCrop(crop);
        }

        // Mejorar contraste y nitidez del crop
        crop = enhanceCrop(crop);

        float confidence = 0.0f;
        std::string text = recognize(crop, confidence);
        if (!text.empty() && confidence > config_.minConfidence) {
            results.emplace_back(box, text, confidence);
        }
        crop.release();
    }
    src.release();
    return results;
}

cv::Mat OnnxOcrAdapter::preprocessDet(const cv::Mat& src, float& out_ratio_h, float& out_ratio_w) {
    int h = src.rows;
    int w = src.cols;
    if (h <= 0 || w <= 0) return cv::Mat();

    float ratio = 1.0f;
    if (std::max(h, w) > config_.maxSideLen) {
        ratio = (float)config_.maxSideLen / (float)std::max(h, w);
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

std::vector<domain::BoundingBox> OnnxOcrAdapter::detect(const cv::Mat& src) {
    float r_h = 1.0f, r_w = 1.0f;
    cv::Mat inputImg = preprocessDet(src, r_h, r_w);
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
    return postprocessDet(heatmap, src, r_h, r_w);
}

std::vector<domain::BoundingBox> OnnxOcrAdapter::postprocessDet(const cv::Mat& heatmap, const cv::Mat& src, float r_h, float r_w) {
    cv::Mat binary;
    cv::threshold(heatmap, binary, config_.detThreshold, 1.0, cv::THRESH_BINARY);
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

        float offset = (w * h) * config_.unclipRatio / (2 * (w + h));
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

cv::Mat OnnxOcrAdapter::preprocessRec(const cv::Mat& boxImg) {
    if (boxImg.empty() || boxImg.cols <= 0 || boxImg.rows <= 0) return cv::Mat();

    // Usar targetH configurable (default 64px >> 48px original para más detalle)
    int targetH = config_.recTargetH;
    float scale = (float)targetH / (float)boxImg.rows;
    int targetW = (int)std::max(1.0f, (float)boxImg.cols * scale);
    if (targetW > 8000) targetW = 8000;

    cv::Mat resized;
    // INTER_CUBIC da mejor nitidez que INTER_LINEAR al escalar texto pequeño
    cv::resize(boxImg, resized, cv::Size(targetW, targetH), 0, 0, cv::INTER_CUBIC);

    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);
    cv::Scalar mean(0.5, 0.5, 0.5), std_v(0.5, 0.5, 0.5);
    floatImg -= mean;
    floatImg /= std_v;
    return floatImg;
}

std::string OnnxOcrAdapter::recognize(const cv::Mat& boxImg, float& confidence) {
    cv::Mat inputImg = preprocessRec(boxImg);
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

    // --- Softmax por paso de tiempo y guardado de (idx, prob_softmax) ---
    // Usamos softmax para tener probabilidades reales en lugar de logits crudos
    struct FramePred { int idx; float prob; };
    std::vector<FramePred> preds;
    preds.reserve((size_t)T);

    for (int t = 0; t < T; ++t) {
        float* row = data + t * C;
        // Softmax numéricamente estable
        float maxLogit = *std::max_element(row, row + C);
        float sumExp   = 0.0f;
        for (int c = 0; c < C; ++c) sumExp += std::exp(row[c] - maxLogit);
        int   bestIdx  = (int)(std::max_element(row, row + C) - row);
        float bestProb = std::exp(row[bestIdx] - maxLogit) / sumExp;
        preds.push_back({bestIdx, bestProb});
    }

    // Recopilar índices para CTC decode
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
    // CTC decode estándar: colapsar repetidos y eliminar blanks (idx==0)
    // PERO: para caracteres IGUALES consecutivos con un blank intermedio
    // debemos permitir que ambos salgan (eso es lo correcto en CTC).
    // El problema de "47144800" -> "4714800" ocurre cuando el modelo predice
    // el MISMO idx para dos cuatro (4) consecutivos SIN un blank entre ellos.
    // La solución está en el mejor preprocesado de imagen (más resolución,
    // CLAHE, deskew) para que el modelo genere blanks entre iguales.
    // Aquí implementamos el decode correcto: resetear lastIdx cuando hay blank.
    std::string text;
    int lastIdx = -1;
    for (auto idx : indices) {
        if (idx == 0) {
            // Blank token: resetear seguimiento para permitir el mismo carácter después
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

// ---------------------------------------------------------------------------
// Mejora de imagen: CLAHE local + unsharp mask para afilar texto
// ---------------------------------------------------------------------------
cv::Mat OnnxOcrAdapter::enhanceCrop(const cv::Mat& crop) const {
    if (crop.empty()) return crop;

    // Convertir a LAB para mejorar sólo el canal de luminancia
    cv::Mat lab;
    cv::cvtColor(crop, lab, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> labChannels;
    cv::split(lab, labChannels);

    // CLAHE sobre L
    if (config_.enableClahe) {
        auto clahe = cv::createCLAHE(3.0, cv::Size(4, 4));
        clahe->apply(labChannels[0], labChannels[0]);
    }

    // Unsharp mask: afilar = original + alpha*(original - blur)
    cv::Mat blurred;
    cv::GaussianBlur(labChannels[0], blurred, cv::Size(0, 0), 1.5);
    cv::addWeighted(labChannels[0], 1.8, blurred, -0.8, 0, labChannels[0]);

    cv::merge(labChannels, lab);
    cv::Mat result;
    cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
    return result;
}

// ---------------------------------------------------------------------------
// Corrección de inclinación (deskew) usando momentos de imagen binarizada
// Solo corrige ángulos pequeños (<=15°) para no distorsionar texto recto
// ---------------------------------------------------------------------------
cv::Mat OnnxOcrAdapter::deskewCrop(const cv::Mat& crop) const {
    if (crop.empty() || crop.cols < 10 || crop.rows < 10) return crop;

    cv::Mat gray;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);

    // Binarizar con Otsu para detectar ángulo de skew
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // Calcular momentos y ángulo
    cv::Moments m   = cv::moments(binary, true);
    double angle    = 0.0;
    if (std::abs(m.mu11) > 1e-6 || std::abs(m.mu20 - m.mu02) > 1e-6) {
        angle = 0.5 * std::atan2(2.0 * m.mu11, m.mu20 - m.mu02) * 180.0 / CV_PI;
    }

    // Limitar corrección a ±15° para no rotar texto que ya está bien
    if (std::abs(angle) < 0.5 || std::abs(angle) > 15.0) return crop;

    cv::Point2f center((float)crop.cols / 2.0f, (float)crop.rows / 2.0f);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat deskewed;
    cv::warpAffine(crop, deskewed, rotMat, crop.size(),
                   cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return deskewed;
}

} // namespace ocr::infrastructure
