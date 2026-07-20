//
// Created by gaoxiang on 2020/5/13.
//

#ifndef LIGHTNING_YAML_IO_H
#define LIGHTNING_YAML_IO_H

#include <yaml-cpp/yaml.h>
#include <cassert>
#include <string>

namespace lightning {

/// 读取yaml配置文件的相关IO
class YAML_IO {
   public:
    explicit YAML_IO(const std::string &path);

    YAML_IO() = default;
    ~YAML_IO() = default;

    inline bool IsOpened() const { return is_opened_; }

    /// 保存文件，不指明路径时，覆盖原文件
    bool Save(const std::string &path = "");

    /// 获取类型为T的参数值
    template <typename T>
    T GetValue(const std::string &key) const {
        assert(is_opened_);
        return yaml_node_[key].as<T>();
    }

    /// 获取在NODE下的key值
    // 读取两层yaml参数
    template <typename T>
    T GetValue(const std::string &node, const std::string &key) const {
        assert(is_opened_);
        T res = yaml_node_[node][key].as<T>();
        return res;
    }

    template <typename T>
    T GetValueOr(const std::string &node, const std::string &key, const T &default_value) const {
        assert(is_opened_);
        const YAML::Node value = yaml_node_[node][key];
        return value ? value.as<T>() : default_value;
    }
    // 读取三层yaml参数
    template <typename T>
    T GetValue(const std::string &node_1, const std::string &node_2, const std::string &key) const {
        assert(is_opened_);
        T res = yaml_node_[node_1][node_2][key].as<T>();
        return res;
    }

    /// 设定类型为T的参数值
    template <typename T>
    void SetValue(const std::string &key, const T &value) {
        yaml_node_[key] = value;
    }

    /// 设定NODE下的key值
    template <typename T>
    void SetValue(const std::string &node, const std::string &key, const T &value) {
        yaml_node_[node][key] = value;
    }

   private:
    std::string path_;
    bool is_opened_ = false;
    YAML::Node yaml_node_;
};

}  // namespace lightning

#endif
