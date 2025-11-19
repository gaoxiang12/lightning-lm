//
// Created by xiang on 25-3-12.
//

#include "io/file_io.h"
#include <filesystem>

namespace lightning {
bool PathExists(const std::string& file_path) {
    std::filesystem::path path(file_path);
    return std::filesystem::exists(path);
}

bool RemoveIfExist(const std::string& path) {
    if (PathExists(path)) {
        try {
            std::filesystem::remove(std::filesystem::path(path));
            return true;
        } catch (const std::filesystem::filesystem_error& e) {
            // LOG(WARNING) << "Failed to remove " << path << ": " << e.what();
            return false;
        }
    }
    return false;
}

bool IsDirectory(const std::string& path) { return std::filesystem::is_directory(path); }

}  // namespace lightning