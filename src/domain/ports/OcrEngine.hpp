#ifndef DOMAIN_OCR_ENGINE_HPP
#define DOMAIN_OCR_ENGINE_HPP

#include "../entities/TextBlock.hpp"
#include "../common/Result.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

namespace ocr::domain {

/**
 * @brief Interfaz (Puerto) para el motor de OCR.
 * 
 * Define las capacidades mínimas que cualquier implementación de OCR debe proveer.
 */
class OcrEngine {
public:
    virtual ~OcrEngine() = default;

    /**
     * @brief Procesa una imagen desde una ruta.
     */
    virtual Result<std::vector<TextBlock>> detectAndRecognize(const std::string& imagePath) = 0;

    /**
     * @brief Procesa una imagen desde un buffer en memoria.
     */
    virtual Result<std::vector<TextBlock>> detectAndRecognize(const unsigned char* buffer, size_t size) = 0;

    /**
     * @brief Procesa una imagen directamente desde una matriz de OpenCV. (Puente Ultra-Rápido)
     */
    virtual Result<std::vector<TextBlock>> detectAndRecognize(const cv::Mat& src) = 0;
};


} // namespace ocr::domain

#endif // DOMAIN_OCR_ENGINE_HPP

