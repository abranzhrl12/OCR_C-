#ifndef DOMAIN_LAYOUT_SERVICE_HPP
#define DOMAIN_LAYOUT_SERVICE_HPP

#include "../entities/TextBlock.hpp"
#include "../builders/ITextBuilder.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace ocr::domain {

/**
 * @brief Servicio para agrupar bloques de texto en líneas y aplicar constructores.
 */
class LayoutService {
public:
    /**
     * @brief Agrupa los bloques de texto en líneas físicas.
     */
    static std::vector<LineGroup> groupIntoLines(const std::vector<TextBlock>& blocks) {
        if (blocks.empty()) return {};

        float avgHeight = calculateAvgHeight(blocks);
        std::vector<LineGroup> lines;
        
        for (size_t i = 0; i < blocks.size(); ++i) {
            float currentY = blocks[i].getCenterY();
            bool added = false;
            
            for (auto& line : lines) {
                if (std::abs(currentY - line.avgY) < avgHeight * 0.5f) {
                    line.blockIndices.push_back(i);
                    float sum = 0;
                    for (size_t idx : line.blockIndices) sum += blocks[idx].getCenterY();
                    line.avgY = sum / line.blockIndices.size();
                    added = true;
                    break;
                }
            }
            
            if (!added) {
                lines.push_back({{i}, currentY});
            }
        }
        
        // Ordenar verticalmente
        std::sort(lines.begin(), lines.end(), [](const LineGroup& a, const LineGroup& b) {
            return a.avgY < b.avgY;
        });
        
        // Ordenar horizontalmente dentro de cada línea
        for (auto& line : lines) {
            std::sort(line.blockIndices.begin(), line.blockIndices.end(), [&](size_t a, size_t b) {
                return blocks[a].getMinX() < blocks[b].getMinX();
            });
        }
        
        return lines;
    }

    /**
     * @brief Construye el texto final usando un constructor específico.
     */
    static std::string build(const std::vector<TextBlock>& blocks, 
                           const std::vector<LineGroup>& lines,
                           ITextBuilder& builder) {
        builder.reset();
        builder.buildHeader(blocks);
        for (size_t i = 0; i < lines.size(); ++i) {
            builder.buildLine(i, lines[i], blocks);
        }
        builder.buildFooter();
        return builder.getResult();
    }

private:
    static float calculateAvgHeight(const std::vector<TextBlock>& blocks) {
        if (blocks.empty()) return 20.0f;
        float total = 0;
        for (const auto& b : blocks) total += b.getHeight();
        return total / blocks.size();
    }
};

} // namespace ocr::domain

#endif
