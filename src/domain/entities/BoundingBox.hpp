#ifndef DOMAIN_BOUNDING_BOX_HPP
#define DOMAIN_BOUNDING_BOX_HPP

#include <vector>
#include <array>

namespace ocr::domain {

struct Point {
    float x;
    float y;
};

class BoundingBox {
public:
    BoundingBox(std::vector<Point> points, float score)
        : points_(std::move(points)), score_(score) {}

    const std::vector<Point>& getPoints() const { return points_; }
    float getScore() const { return score_; }

private:
    std::vector<Point> points_;
    float score_;
};

} // namespace ocr::domain

#endif // DOMAIN_BOUNDING_BOX_HPP
