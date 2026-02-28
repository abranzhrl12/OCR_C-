#ifndef DOMAIN_TEXT_BLOCK_HPP
#define DOMAIN_TEXT_BLOCK_HPP

#include "BoundingBox.hpp"
#include <string>
#include <algorithm>
#include <cmath>

namespace ocr::domain {

/**
 * @brief Entidad que representa un bloque de texto detectado con su geometría.
 */
class TextBlock {
public:
    TextBlock(BoundingBox box, std::string text, float confidence)
        : box_(std::move(box)), text_(std::move(text)), confidence_(confidence) {}

    const BoundingBox& getBox() const { return box_; }
    const std::string& getText() const { return text_; }
    float getConfidence() const { return confidence_; }

    // --- Helpers Geométricos ---

    float getMinX() const {
        const auto& p = box_.getPoints();
        return std::min({p[0].x, p[1].x, p[2].x, p[3].x});
    }

    float getMaxX() const {
        const auto& p = box_.getPoints();
        return std::max({p[0].x, p[1].x, p[2].x, p[3].x});
    }

    float getMinY() const {
        const auto& p = box_.getPoints();
        return std::min({p[0].y, p[1].y, p[2].y, p[3].y});
    }

    float getMaxY() const {
        const auto& p = box_.getPoints();
        return std::max({p[0].y, p[1].y, p[2].y, p[3].y});
    }

    float getCenterY() const {
        float sum = 0;
        for (const auto& p : box_.getPoints()) sum += p.y;
        return sum / 4.0f;
    }

    float getHeight() const {
        const auto& p = box_.getPoints();
        float h1 = std::sqrt(std::pow(p[0].x - p[3].x, 2) + std::pow(p[0].y - p[3].y, 2));
        float h2 = std::sqrt(std::pow(p[1].x - p[2].x, 2) + std::pow(p[1].y - p[2].y, 2));
        return (h1 + h2) / 2.0f;
    }

private:
    BoundingBox box_;
    std::string text_;
    float confidence_;
};

} // namespace ocr::domain

#endif // DOMAIN_TEXT_BLOCK_HPP

