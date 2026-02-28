#ifndef DOMAIN_COMMON_RESULT_HPP
#define DOMAIN_COMMON_RESULT_HPP

#include <string>
#include <variant>
#include <stdexcept>
#include <optional>

namespace ocr::domain {

/**
 * @brief Estructura que representa un error en el sistema.
 */
struct Error {
    std::string message;
    int code;

    explicit Error(std::string msg, int c = -1) : message(std::move(msg)), code(c) {}
};

/**
 * @brief Clase genérica para manejar el éxito o fallo de una operación (Similar a std::expected).
 * 
 * @tparam T Tipo del valor devuelto en caso de éxito.
 */
template <typename T>
class Result {
public:
    // Constructores para éxito
    Result(T value) : data_(std::move(value)) {}

    // Constructores para error
    Result(Error error) : data_(std::move(error)) {}

    /**
     * @brief Indica si la operación fue exitosa.
     */
    bool isOk() const { return std::holds_alternative<T>(data_); }

    /**
     * @brief Indica si hubo un error.
     */
    bool isError() const { return std::holds_alternative<Error>(data_); }

    /**
     * @brief Obtiene el valor (Lanza excepción si es un error, por seguridad).
     */
    const T& value() const {
        if (isError()) throw std::runtime_error("Attempted to access value of an Error Result: " + error().message);
        return std::get<T>(data_);
    }

    T& value() {
        if (isError()) throw std::runtime_error("Attempted to access value of an Error Result: " + error().message);
        return std::get<T>(data_);
    }

    /**
     * @brief Obtiene el error.
     */
    const Error& error() const {
        if (isOk()) throw std::runtime_error("Attempted to access error of an Ok Result");
        return std::get<Error>(data_);
    }

    // Sugar: permite usar el objeto en un if
    explicit operator bool() const { return isOk(); }

    // Helpers estáticos para construcción fluida
    static Result<T> Ok(T value) { return Result<T>(std::move(value)); }
    static Result<T> Fail(std::string msg, int code = -1) { return Result<T>(Error(std::move(msg), code)); }

private:
    std::variant<T, Error> data_;
};

/**
 * @brief Especialización para Result<void> para operaciones que solo pueden fallar pero no devuelven valor.
 */
template <>
class Result<void> {
public:
    Result() : error_(std::nullopt) {}
    Result(Error error) : error_(std::move(error)) {}

    bool isOk() const { return !error_.has_value(); }
    bool isError() const { return error_.has_value(); }

    const Error& error() const { return error_.value(); }

    explicit operator bool() const { return isOk(); }

    static Result<void> Ok() { return Result<void>(); }
    static Result<void> Fail(std::string msg, int code = -1) { return Result<void>(Error(std::move(msg), code)); }

private:
    std::optional<Error> error_;
};

} // namespace ocr::domain

#endif // DOMAIN_COMMON_RESULT_HPP
