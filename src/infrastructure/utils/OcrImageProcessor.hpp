#ifndef INFRASTRUCTURE_OCR_IMAGE_PROCESSOR_HPP
#define INFRASTRUCTURE_OCR_IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "../../domain/entities/BoundingBox.hpp"
#include "../adapters/OnnxOcrAdapter.hpp" // For OcrConfig

namespace ocr::infrastructure {

/**
 * @brief Clase utilitaria para procesamiento de imágenes específico de OCR.
 * 
 * Desacopla la lógica de visión artificial de la lógica de inferencia de ONNX.
 */
class OcrImageProcessor {
public:
    // --- Pre-procesamiento de Detección ---
    static cv::Mat preprocessDet(const cv::Mat& src, const OcrConfig& config, float& out_ratio_h, float& out_ratio_w);
    
    // --- Post-procesamiento de Detección (Extracción de cajas) ---
    static std::vector<domain::BoundingBox> postprocessDet(const cv::Mat& heatmap, const cv::Mat& src, const OcrConfig& config, float r_h, float r_w);
    
    // --- Pre-procesamiento de Reconocimiento (Normalización de crops) ---
    static cv::Mat preprocessRec(const cv::Mat& boxImg, const OcrConfig& config);
    
    // --- Mejoras de Calidad ---
    static cv::Mat enhanceCrop(const cv::Mat& crop, const OcrConfig& config);
    static cv::Mat deskewCrop(const cv::Mat& crop, const OcrConfig& config);
    
    // --- Utilidades Globales ---
    static void applyGlobalEnhancements(cv::Mat& src, const OcrConfig& config);
};

} // namespace ocr::infrastructure

#endif
