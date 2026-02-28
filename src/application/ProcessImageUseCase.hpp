#ifndef APPLICATION_PROCESS_IMAGE_USE_CASE_HPP
#define APPLICATION_PROCESS_IMAGE_USE_CASE_HPP

#include "../domain/ports/OcrEngine.hpp"

#include "../domain/services/LayoutService.hpp"
#include "../domain/builders/TextBuilderFactory.hpp"
#include <nlohmann/json.hpp>
#include <memory>
#include <vector>
#include <string>

namespace ocr::application {

/**
 * @brief Caso de uso que orquesta el proceso completo de OCR.
 * 
 * Combina la inferencia del motor (OCR Engine) con el formateo de salida (Layout Service).
 */
class ProcessImageUseCase {
public:
    explicit ProcessImageUseCase(std::shared_ptr<domain::OcrEngine> ocrEngine)
        : ocrEngine_(std::move(ocrEngine)) {}

    /**
     * @brief Ejecuta el motor de OCR y devuelve los bloques detectados.
     */
    domain::Result<std::vector<domain::TextBlock>> execute(const std::string& imagePath) {
        return ocrEngine_->detectAndRecognize(imagePath);
    }

    /**
     * @brief Ejecuta el motor de OCR desde buffer.
     */
    domain::Result<std::vector<domain::TextBlock>> execute(const unsigned char* buffer, size_t size) {
        return ocrEngine_->detectAndRecognize(buffer, size);
    }

    /**
     * @brief Ejecuta el motor de OCR desde matriz (Ultra-rápido).
     */
    domain::Result<std::vector<domain::TextBlock>> execute(const cv::Mat& src) {
        return ocrEngine_->detectAndRecognize(src);
    }

private:
    std::shared_ptr<domain::OcrEngine> ocrEngine_;
};


} // namespace ocr::application

#endif // APPLICATION_PROCESS_IMAGE_USE_CASE_HPP
