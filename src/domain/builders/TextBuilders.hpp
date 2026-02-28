#ifndef DOMAIN_TEXT_BUILDERS_HPP
#define DOMAIN_TEXT_BUILDERS_HPP

#include "ITextBuilder.hpp"

#include <nlohmann/json.hpp>
#include <sstream>
#include <algorithm>

namespace ocr::domain {

using json = nlohmann::json;

/**
 * @brief Implementación que construye JSON crudo de todos los bloques (Ideal para Rust).
 */
class RawJsonTextBuilder : public ITextBuilder {
public:
    void reset() override { json_array_ = json::array(); }
    
    void buildHeader(const std::vector<TextBlock>& allBlocks) override {
        for (const auto& b : allBlocks) {
            json box = json::array();
            for (const auto& p : b.getBox().getPoints()) {
                box.push_back({{"x", p.x}, {"y", p.y}});
            }
            json_array_.push_back({
                {"text", b.getText()},
                {"confidence", b.getConfidence()},
                {"box", box}
            });
        }
    }

    void buildLine(size_t, const LineGroup&, const std::vector<TextBlock>&) override {}
    void buildFooter() override {}
    
    std::string getResult() const override { return json_array_.dump(); }

private:
    json json_array_;
};

/**
 * @brief Implementación que construye la salida en formato JSON agrupado por líneas.
 */

class JsonTextBuilder : public ITextBuilder {
public:
    void reset() override {
        json_array_ = json::array();
    }

    void buildHeader(const std::vector<TextBlock>& allBlocks) override {}

    void buildLine(size_t lineIdx, const LineGroup& line, const std::vector<TextBlock>& allBlocks) override {
        for (size_t idx : line.blockIndices) {
            const auto& b = allBlocks[idx];
            json box = json::array();
            for (const auto& p : b.getBox().getPoints()) {
                box.push_back({{"x", p.x}, {"y", p.y}});
            }
            
            json_array_.push_back({
                {"text", b.getText()},
                {"confidence", b.getConfidence()},
                {"line", lineIdx},
                {"box", box}
            });
        }
    }

    void buildFooter() override {}

    std::string getResult() const override {
        return json_array_.dump();
    }

private:
    json json_array_;
};

/**
 * @brief Implementación que construye texto respetando la disposición visual.
 */
class LayoutTextBuilder : public ITextBuilder {

public:
    void reset() override {
        result_.str("");
        result_.clear();
        globalMinX_ = 100000.0f;
        avgCharWidth_ = 10.0f;
    }

    void buildHeader(const std::vector<TextBlock>& allBlocks) override {
        if (allBlocks.empty()) return;

        float totalHeight = 0;
        for (const auto& b : allBlocks) {
            globalMinX_ = std::min(globalMinX_, b.getMinX());
            totalHeight += b.getHeight();
        }
        float avgHeight = totalHeight / allBlocks.size();
        avgCharWidth_ = avgHeight * 0.6f;
    }

    void buildLine(size_t lineIdx, const LineGroup& line, const std::vector<TextBlock>& allBlocks) override {
        if (lineIdx > 0) result_ << "\n";

        for (size_t pos = 0; pos < line.blockIndices.size(); ++pos) {
            size_t currIdx = line.blockIndices[pos];
            const auto& b = allBlocks[currIdx];

            if (pos == 0) {
                // Indentación inicial
                float rel = std::max(0.0f, b.getMinX() - globalMinX_);
                int indent = (int)(rel / avgCharWidth_);
                if (indent > 0) result_ << std::string(indent, ' ');
            } else {
                // Espacios entre palabras
                size_t prevIdx = line.blockIndices[pos - 1];
                float gap = b.getMinX() - allBlocks[prevIdx].getMaxX();
                int spaces = (gap > 0) ? std::max(1, (int)(gap / avgCharWidth_)) : 1;
                result_ << std::string(spaces, ' ');
            }
            result_ << b.getText();
        }
    }

    void buildFooter() override {}

    std::string getResult() const override {
        return result_.str();
    }

private:
    std::stringstream result_;
    float globalMinX_ = 100000.0f;
    float avgCharWidth_ = 10.0f;
};

/**
 * @brief Implementación que construye texto plano y compacto.
 */
class PlainTextBuilder : public ITextBuilder {
public:
    void reset() override {
        result_.str("");
        result_.clear();
    }

    void buildHeader(const std::vector<TextBlock>& allBlocks) override {}

    void buildLine(size_t lineIdx, const LineGroup& line, const std::vector<TextBlock>& allBlocks) override {
        if (lineIdx > 0) result_ << " ";
        for (size_t pos = 0; pos < line.blockIndices.size(); ++pos) {
            if (pos > 0) result_ << " ";
            result_ << allBlocks[line.blockIndices[pos]].getText();
        }
    }

    void buildFooter() override {}

    std::string getResult() const override {
        return result_.str();
    }

private:
    std::stringstream result_;
};

} // namespace ocr::domain

#endif
