#ifndef DOMAIN_TEXT_BUILDER_FACTORY_HPP
#define DOMAIN_TEXT_BUILDER_FACTORY_HPP

#include "TextBuilders.hpp"
#include <memory>

namespace ocr::domain {

/**
 * @brief Tipos de formatos de texto soportados.
 */
enum class TextFormat {
    PLAIN,
    LAYOUT,
    JSON,
    RAW_JSON
};

/**
 * @brief Simple Factory para centralizar la creación de constructores de texto.
 * 
 * Desacopla la lógica de aplicación de las implementaciones concretas de builders.
 */
class TextBuilderFactory {
public:
    /**
     * @brief Crea un builder basado en el formato solicitado.
     */
    static std::unique_ptr<ITextBuilder> create(TextFormat format) {
        switch (format) {
            case TextFormat::PLAIN:    return std::make_unique<PlainTextBuilder>();
            case TextFormat::LAYOUT:   return std::make_unique<LayoutTextBuilder>();
            case TextFormat::JSON:     return std::make_unique<JsonTextBuilder>();
            case TextFormat::RAW_JSON: return std::make_unique<RawJsonTextBuilder>();
            default: return nullptr;
        }
    }
};



} // namespace ocr::domain

#endif // DOMAIN_TEXT_BUILDER_FACTORY_HPP
