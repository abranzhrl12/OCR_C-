#ifndef DOMAIN_OCR_ENGINE_HPP
#define DOMAIN_OCR_ENGINE_HPP

#include "../entities/TextBlock.hpp"
#include <vector>
#include <string>
#include <memory>

namespace ocr::domain {

class OcrEngine {
public:
    virtual ~OcrEngine() = default;
    virtual std::vector<TextBlock> detectAndRecognize(const std::string& imagePath) = 0;
    virtual std::vector<TextBlock> detectAndRecognize(const unsigned char* buffer, size_t size) = 0;
};

} // namespace ocr::domain

#endif // DOMAIN_OCR_ENGINE_HPP
