#ifndef DOMAIN_TEXT_BLOCK_HPP
#define DOMAIN_TEXT_BLOCK_HPP

#include "BoundingBox.hpp"
#include <string>

namespace ocr::domain {

class TextBlock {
public:
    TextBlock(BoundingBox box, std::string text, float confidence)
        : box_(std::move(box)), text_(std::move(text)), confidence_(confidence) {}

    const BoundingBox& getBox() const { return box_; }
    const std::string& getText() const { return text_; }
    float getConfidence() const { return confidence_; }

private:
    BoundingBox box_;
    std::string text_;
    float confidence_;
};

} // namespace ocr::domain

#endif // DOMAIN_TEXT_BLOCK_HPP
