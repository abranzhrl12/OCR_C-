#ifndef DOMAIN_TEXT_BUILDER_HPP
#define DOMAIN_TEXT_BUILDER_HPP

#include "../entities/TextBlock.hpp"
#include <vector>
#include <string>

namespace ocr::domain {

/**
 * @brief Estructura que representa una línea lógica de bloques.
 */
struct LineGroup {
    std::vector<size_t> blockIndices;
    float avgY;
};

/**
 * @brief Interfaz base para el patrón Builder de construcción de texto.
 */
class ITextBuilder {
public:
    virtual ~ITextBuilder() = default;

    virtual void reset() = 0;
    virtual void buildHeader(const std::vector<TextBlock>& allBlocks) = 0;
    virtual void buildLine(size_t lineIdx, const LineGroup& line, const std::vector<TextBlock>& allBlocks) = 0;
    virtual void buildFooter() = 0;
    
    virtual std::string getResult() const = 0;
};

} // namespace ocr::domain

#endif
