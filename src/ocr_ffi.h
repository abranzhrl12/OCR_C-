#ifndef OCR_FFI_H
#define OCR_FFI_H

#define OCR_API 


#ifdef __cplusplus
extern "C" {
#endif

// Modos de salida de texto
// OCR_MODE_LAYOUT = 0: Texto con estructura visual (líneas y espaciado)
// OCR_MODE_PLAIN  = 1: Texto compacto en párrafo (todo junto, denso)
#define OCR_MODE_LAYOUT 0
#define OCR_MODE_PLAIN  1

// Inicializar motor OCR con rutas a modelos
// Retorna: handle opaco (void*) o NULL si falla
OCR_API void* ocr_init(const char* det_model_path,
                       const char* rec_model_path,
                       const char* dict_path);

// Procesar imagen y extraer texto
// handle: handle del motor OCR (de ocr_init)
// image_path: ruta de la imagen
// mode: OCR_MODE_LAYOUT o OCR_MODE_PLAIN
// Retorna: string con texto extraído (debe liberarse con ocr_free_string)
OCR_API const char* ocr_process(void* handle,
                                const char* image_path,
                                int mode);

// Procesar buffer de imagen en memoria (bytes)
// buffer: puntero a los bytes de la imagen (jpg, png, etc.)
// size: tamaño del buffer en bytes
// mode: OCR_MODE_LAYOUT o OCR_MODE_PLAIN
OCR_API const char* ocr_process_buffer(void* handle,
                                       const unsigned char* buffer,
                                       int size,
                                       int mode);

// Procesar imagen y obtener JSON con coordenadas agrupado por líneas (Layout)
OCR_API const char* ocr_process_json(void* handle,
                                     const char* image_path);

// Procesar buffer y obtener JSON Layout
OCR_API const char* ocr_process_buffer_json(void* handle,
                                            const unsigned char* buffer,
                                            int size);

// Procesar imagen y obtener JSON crudo de bloques (Materia prima para Rust)
OCR_API const char* ocr_process_raw(void* handle,
                                    const char* image_path);

// Procesar buffer y obtener JSON crudo
OCR_API const char* ocr_process_buffer_raw(void* handle,
                                           const unsigned char* buffer,
                                           int size);

// Procesar píxeles crudos (Alto rendimiento para Rust)
// pixels: puntero a los bytes (BGR/RGB)
// width, height: dimensiones
// channels: usualmente 3 (BGR)
OCR_API const char* ocr_process_pixels_raw(void* handle,
                                           const unsigned char* pixels,
                                           int width,
                                           int height,
                                           int channels);



// Liberar string retornado por ocr_process o ocr_process_json
OCR_API void ocr_free_string(const char* str);

// Destruir motor OCR y liberar memoria
OCR_API void ocr_destroy(void* handle);

#ifdef __cplusplus
}
#endif

#endif // OCR_FFI_H
