#include "ocr_ffi.h"
#include "infrastructure/adapters/OnnxOcrAdapter.hpp"
#include "domain/entities/TextBlock.hpp"
#include "domain/entities/BoundingBox.hpp"
#include <nlohmann/json.hpp>
#include <string>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>

using json = nlohmann::json;
using namespace ocr::infrastructure;
using namespace ocr::domain;

// Helpers para coordenadas de bloques
static float getAvgY(const TextBlock& tb) {
    float sum = 0;
    for (auto& p : tb.getBox().getPoints()) sum += p.y;
    return sum / 4.0f;
}

static float getMinX(const TextBlock& tb) {
    auto p = tb.getBox().getPoints();
    return std::min({p[0].x, p[1].x, p[2].x, p[3].x});
}

static float getMaxX(const TextBlock& tb) {
    auto p = tb.getBox().getPoints();
    return std::max({p[0].x, p[1].x, p[2].x, p[3].x});
}

// Estructura interna de línea para agrupamiento
struct TextLine {
    std::vector<size_t> blockIndices;
    float avgY;
};

// Agrupar y ordenar bloques en líneas
static std::vector<TextLine> groupIntoLines(
    const std::vector<TextBlock>& results, 
    float avgHeight
) {
    std::vector<TextLine> lines;
    
    for (size_t i = 0; i < results.size(); ++i) {
        float currentY = getAvgY(results[i]);
        bool added = false;
        
        for (auto& line : lines) {
            if (std::abs(currentY - line.avgY) < avgHeight * 0.5f) {
                line.blockIndices.push_back(i);
                float sum = 0;
                for (size_t idx : line.blockIndices) sum += getAvgY(results[idx]);
                line.avgY = sum / line.blockIndices.size();
                added = true;
                break;
            }
        }
        
        if (!added) {
            TextLine newLine;
            newLine.blockIndices.push_back(i);
            newLine.avgY = currentY;
            lines.push_back(newLine);
        }
    }
    
    // Ordenar líneas de arriba a abajo
    std::sort(lines.begin(), lines.end(), [](const TextLine& a, const TextLine& b) {
        return a.avgY < b.avgY;
    });
    
    // Dentro de cada línea, ordenar de izquierda a derecha
    for (auto& line : lines) {
        std::sort(line.blockIndices.begin(), line.blockIndices.end(),
            [&](size_t a, size_t b) {
                return getMinX(results[a]) < getMinX(results[b]);
            });
    }
    
    return lines;
}

// Calcular métricas globales
static float calcAvgHeight(const std::vector<TextBlock>& results) {
    float totalHeight = 0;
    for (const auto& block : results) {
        auto p = block.getBox().getPoints();
        float h1 = std::sqrt(std::pow(p[0].x - p[3].x, 2) + std::pow(p[0].y - p[3].y, 2));
        float h2 = std::sqrt(std::pow(p[1].x - p[2].x, 2) + std::pow(p[1].y - p[2].y, 2));
        totalHeight += (h1 + h2) / 2.0f;
    }
    return results.empty() ? 20.0f : totalHeight / results.size();
}

// Generar texto en modo LAYOUT (con estructura visual de líneas)
static std::string buildLayoutText(
    const std::vector<TextBlock>& results,
    const std::vector<TextLine>& lines,
    float avgHeight
) {
    float globalMinX = 10000.0f;
    for (const auto& b : results) {
        float mx = getMinX(b);
        if (mx < globalMinX) globalMinX = mx;
    }
    float charWidth = avgHeight * 0.6f;
    
    std::string text;
    text.reserve(results.size() * 30); // Pre-allocar
    
    for (size_t lineIdx = 0; lineIdx < lines.size(); ++lineIdx) {
        if (lineIdx > 0) text += "\n";
        
        for (size_t pos = 0; pos < lines[lineIdx].blockIndices.size(); ++pos) {
            size_t i = lines[lineIdx].blockIndices[pos];
            
            if (pos == 0) {
                float rel = std::max(0.0f, getMinX(results[i]) - globalMinX);
                int indent = (int)(rel / charWidth);
                if (indent > 0) text += std::string(indent, ' ');
            } else {
                size_t prev = lines[lineIdx].blockIndices[pos - 1];
                float gap = getMinX(results[i]) - getMaxX(results[prev]);
                int spaces = (gap > 0) ? std::max(1, (int)(gap / charWidth)) : 1;
                text += std::string(spaces, ' ');
            }
            
            text += results[i].getText();
        }
    }
    return text;
}

// Generar texto en modo PLAIN (compacto, párrafo denso)
static std::string buildPlainText(
    const std::vector<TextBlock>& results,
    const std::vector<TextLine>& lines
) {
    std::string text;
    text.reserve(results.size() * 20); // Pre-allocar
    
    for (size_t lineIdx = 0; lineIdx < lines.size(); ++lineIdx) {
        // Separar líneas con solo 1 espacio (todo compacto)
        if (lineIdx > 0) text += " ";
        
        for (size_t pos = 0; pos < lines[lineIdx].blockIndices.size(); ++pos) {
            if (pos > 0) text += " ";
            text += results[lines[lineIdx].blockIndices[pos]].getText();
        }
    }
    return text;
}

// Generar JSON con coordenadas
static std::string buildJsonOutput(const std::vector<TextBlock>& results) {
    json output = json::array();
    for (const auto& block : results) {
        json box = json::array();
        for (const auto& p : block.getBox().getPoints()) {
            box.push_back({{"x", p.x}, {"y", p.y}});
        }
        output.push_back({
            {"text", block.getText()},
            {"confidence", block.getConfidence()},
            {"box", box}
        });
    }
    return output.dump();
}

// Duplicar string para retornar al caller (el caller debe liberar con ocr_free_string)
static char* duplicateString(const std::string& str) {
    char* result = new char[str.size() + 1];
    std::memcpy(result, str.c_str(), str.size() + 1);
    return result;
}

// Helper para formatear resultados de inferencia
static const char* formatResult(const std::vector<TextBlock>& results, int mode) {
    if (results.empty()) return duplicateString("");
    
    float avgHeight = calcAvgHeight(results);
    auto lines = groupIntoLines(results, avgHeight);
    
    std::string text;
    if (mode == OCR_MODE_PLAIN) {
        text = buildPlainText(results, lines);
    } else {
        text = buildLayoutText(results, lines, avgHeight);
    }
    
    return duplicateString(text);
}

// ========== API FFI ==========

extern "C" {

OCR_API void* ocr_init(const char* det_model_path,
                       const char* rec_model_path,
                       const char* dict_path) {
    try {
        OcrConfig config;
        config.detModelPath = det_model_path;
        config.recModelPath = rec_model_path;
        config.dictPath = dict_path;
        
        auto* adapter = new OnnxOcrAdapter(config);
        return static_cast<void*>(adapter);
    } catch (const std::exception& e) {
        std::cerr << "[OCR_FFI] Error inicializando: " << e.what() << std::endl;
        return nullptr;
    }
}

OCR_API const char* ocr_process(void* handle,
                                const char* image_path,
                                int mode) {
    if (!handle || !image_path) return nullptr;
    
    try {
        auto* adapter = static_cast<OnnxOcrAdapter*>(handle);
        return formatResult(adapter->detectAndRecognize(image_path), mode);
    } catch (const std::exception& e) {
        std::cerr << "[OCR_FFI] Error procesando: " << e.what() << std::endl;
        return nullptr;
    }
}

OCR_API const char* ocr_process_buffer(void* handle,
                                       const unsigned char* buffer,
                                       int size,
                                       int mode) {
    if (!handle || !buffer || size <= 0) return nullptr;
    
    try {
        auto* adapter = static_cast<OnnxOcrAdapter*>(handle);
        return formatResult(adapter->detectAndRecognize(buffer, (size_t)size), mode);
    } catch (const std::exception& e) {
        std::cerr << "[OCR_FFI] Error procesando buffer: " << e.what() << std::endl;
        return nullptr;
    }
}

OCR_API const char* ocr_process_json(void* handle,
                                     const char* image_path) {
    if (!handle || !image_path) return nullptr;
    
    try {
        auto* adapter = static_cast<OnnxOcrAdapter*>(handle);
        auto results = adapter->detectAndRecognize(image_path);
        return duplicateString(buildJsonOutput(results));
    } catch (const std::exception& e) {
        std::cerr << "[OCR_FFI] Error procesando JSON: " << e.what() << std::endl;
        return nullptr;
    }
}

OCR_API const char* ocr_process_buffer_json(void* handle,
                                            const unsigned char* buffer,
                                            int size) {
    if (!handle || !buffer || size <= 0) return nullptr;
    
    try {
        auto* adapter = static_cast<OnnxOcrAdapter*>(handle);
        auto results = adapter->detectAndRecognize(buffer, (size_t)size);
        return duplicateString(buildJsonOutput(results));
    } catch (const std::exception& e) {
        std::cerr << "[OCR_FFI] Error procesando buffer JSON: " << e.what() << std::endl;
        return nullptr;
    }
}

OCR_API void ocr_free_string(const char* str) {
    delete[] str;
}

OCR_API void ocr_destroy(void* handle) {
    if (handle) {
        delete static_cast<OnnxOcrAdapter*>(handle);
    }
}

} // extern "C"
