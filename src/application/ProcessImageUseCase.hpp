#ifndef APPLICATION_PROCESS_IMAGE_USE_CASE_HPP
#define APPLICATION_PROCESS_IMAGE_USE_CASE_HPP

#include "../domain/ports/OcrEngine.hpp"
#include <memory>
#include <vector>

namespace ocr::application {

class ProcessImageUseCase {
public:
    explicit ProcessImageUseCase(std::shared_ptr<domain::OcrEngine> ocrEngine)
        : ocrEngine_(std::move(ocrEngine)) {}

    std::vector<domain::TextBlock> execute(const std::string& imagePath) {
        return ocrEngine_->detectAndRecognize(imagePath);
    }

private:
    std::shared_ptr<domain::OcrEngine> ocrEngine_;
};

} // namespace ocr::application

#endif // APPLICATION_PROCESS_IMAGE_USE_CASE_HPP
