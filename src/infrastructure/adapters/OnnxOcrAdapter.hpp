#ifndef INFRASTRUCTURE_ONNX_OCR_ADAPTER_HPP
#define INFRASTRUCTURE_ONNX_OCR_ADAPTER_HPP

#include "../../domain/ports/OcrEngine.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace ocr::infrastructure {

struct OcrConfig {
    std::string detModelPath;
    std::string recModelPath;
    std::string dictPath;
    float detThreshold   = 0.15f;  // sensible para no perder caracteres débiles
    float boxThreshold   = 0.3f;   // aceptar cajas con baja confianza
    float unclipRatio    = 2.0f;   // más expansión de caja para no recortar bordes
    int   maxSideLen     = 2560;   // soportar A4 a 300 DPI sin reducción excesiva
    int   recTargetH     = 64;     // altura de reconocimiento (más alto = más detalle)
    bool  enableDeskew   = true;   // corregir inclinación de cada crop
    bool  enableClahe    = true;   // mejorar contraste local con CLAHE
    float minConfidence  = 0.15f;  // umbral mínimo de confianza para aceptar bloque
};

class OnnxOcrAdapter : public domain::OcrEngine {
public:
    explicit OnnxOcrAdapter(const OcrConfig& config);
    ~OnnxOcrAdapter() override = default;

    std::vector<domain::TextBlock> detectAndRecognize(const std::string& imagePath) override;
    std::vector<domain::TextBlock> detectAndRecognize(const unsigned char* buffer, size_t size) override;

private:
    std::vector<domain::TextBlock> processMat(cv::Mat src);
    OcrConfig config_;
    static Ort::Env* env_;
    Ort::Session detSession_;
    Ort::Session recSession_;
    std::vector<std::string> dict_;

    // Helpers
    void loadDictionary(const std::string& path);

    std::vector<domain::BoundingBox> detect(const cv::Mat& src);
    std::string recognize(const cv::Mat& boxImg, float& confidence);

    cv::Mat preprocessDet(const cv::Mat& src, float& ratio_h, float& ratio_w);
    std::vector<domain::BoundingBox> postprocessDet(const cv::Mat& heatmap, const cv::Mat& src, float ratio_h, float ratio_w);

    cv::Mat preprocessRec(const cv::Mat& boxImg);
    std::string ctcDecode(const std::vector<int64_t>& indices, float& confidence);

    // Mejoras de imagen
    cv::Mat enhanceCrop(const cv::Mat& crop) const;  // CLAHE + afilado
    cv::Mat deskewCrop(const cv::Mat& crop) const;   // corrección inclinación
};

} // namespace ocr::infrastructure

#endif // INFRASTRUCTURE_ONNX_OCR_ADAPTER_HPP
