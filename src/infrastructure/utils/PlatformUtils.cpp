#include "PlatformUtils.hpp"
#include <cstdio>
#include <iostream>

#ifdef _WIN32
    #include <io.h>
    #include <fcntl.h>
    #define DUP _dup
    #define DUP2 _dup2
    #define FILENO _fileno
    #define CLOSE _close
    #define DEV_NULL "NUL"
#else
    #include <unistd.h>
    #include <fcntl.h>
    #define DUP dup
    #define DUP2 dup2
    #define FILENO fileno
    #define CLOSE close
    #define DEV_NULL "/dev/null"
#endif

namespace ocr::infrastructure {

ScopedLogSilencer::ScopedLogSilencer(bool enable) 
    : oldStderr_(-1), isSilenced_(false) {
    if (!enable) return;

    oldStderr_ = DUP(FILENO(stderr));
    if (oldStderr_ < 0) return;

#ifdef _WIN32
    FILE* nul = nullptr;
    if (freopen_s(&nul, DEV_NULL, "w", stderr) == 0) {
        isSilenced_ = true;
    }
#else
    FILE* nul = freopen(DEV_NULL, "w", stderr);
    if (nul != nullptr) {
        isSilenced_ = true;
    }
#endif
}

ScopedLogSilencer::~ScopedLogSilencer() {
    if (!isSilenced_) {
        if (oldStderr_ >= 0) CLOSE(oldStderr_);
        return;
    }

    std::fflush(stderr);
    DUP2(oldStderr_, FILENO(stderr));
    CLOSE(oldStderr_);
}

#ifdef _WIN32
std::wstring PathUtils::toOnnxPath(const std::string& path) {
    if (path.empty()) return std::wstring();
    // Conversión simple a wstring para rutas estándar de Windows
    return std::wstring(path.begin(), path.end());
}
#else
const char* PathUtils::toOnnxPath(const std::string& path) {
    return path.c_str();
}
#endif

} // namespace ocr::infrastructure
