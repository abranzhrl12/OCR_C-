#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "ocr_ffi.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <imagen_path> [det_model] [rec_model] [dict_path]" << std::endl;
        return 1;
    }

    const char* imagePath = argv[1];
    const char* detModel = (argc > 2) ? argv[2] : "models2/pp-ocrv5_mobile_det.onnx";
    const char* recModel = (argc > 3) ? argv[3] : "models2/pp-ocrv5_mobile_rec.onnx";
    const char* dictPath = (argc > 4) ? argv[4] : "models2/ppocrv5_dict.txt";

    // 1. Inicializar FFI
    void* handle = ocr_init(detModel, recModel, dictPath);
    if (!handle) {
        std::cerr << "Fallo al inicializar OCR" << std::endl;
        return 1;
    }

    // 2. Cargar imagen a memoria
    std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir la imagen: " << imagePath << std::endl;
        return 1;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error leyendo imagen" << std::endl;
        return 1;
    }

    // 3. Probar ocr_process_buffer (Modo LAYOUT)
    std::cout << "Procesando en modo LAYOUT..." << std::endl;
    const char* layoutText = ocr_process_buffer(handle, (unsigned char*)buffer.data(), (int)size, OCR_MODE_LAYOUT);
    
    if (layoutText) {
        std::ofstream out("last_extraction_layout.txt");
        out << layoutText;
        out.close();
        std::cout << "Resultado guardado en last_extraction_layout.txt" << std::endl;
        ocr_free_string(layoutText);
    } else {
        std::cerr << "Error en ocr_process_buffer (LAYOUT)" << std::endl;
    }

    // 4. Probar ocr_process_buffer (Modo PLAIN)
    std::cout << "Procesando en modo PLAIN..." << std::endl;
    const char* plainText = ocr_process_buffer(handle, (unsigned char*)buffer.data(), (int)size, OCR_MODE_PLAIN);
    if (plainText) {
        std::ofstream out("last_extraction_plain.txt");
        out << plainText;
        out.close();
        std::cout << "Resultado guardado en last_extraction_plain.txt" << std::endl;
        ocr_free_string(plainText);
    }

    // 5. Limpiar
    ocr_destroy(handle);
    return 0;
}
