//
// Created by user on 2026/3/18.
//

#ifndef LIGHTNING_IMU_FILTER_H
#define LIGHTNING_IMU_FILTER_H

#include "common/eigen_types.h"
#include "common/imu.h"

#include <deque>

namespace lightning {

class IMUFilter {
   private:
    // 配置参数
    struct Config {
        int median_window_size = 5;    // 中值滤波窗口大小
        int moving_avg_window = 3;     // 移动平均窗口
        double rate_limit = 3.0;       // 角速度变化率限制 (rad/s²)
        double spike_threshold = 3.0;  // 毛刺检测阈值 (标准差倍数)
        bool enable_adaptive = true;   // 启用自适应滤波
    } config_;

    // 数据缓冲区
    std::deque<IMU> buffer_;
    std::deque<double> gyro_x_history_;
    std::deque<double> gyro_y_history_;
    std::deque<double> gyro_z_history_;

    // 上一帧滤波后的值
    IMU prev_filtered_;

    // 统计信息
    double gyro_mean_[3] = {0};
    double gyro_std_[3] = {0};
    int sample_count_ = 0;

   public:
    IMUFilter() { prev_filtered_.timestamp = -1; }

    // 设置滤波参数
    void SetMedianWindowSize(int size) {
        if (size >= 3 && size % 2 == 1) {
            config_.median_window_size = size;
        }
    }

    void SetRateLimit(double limit) { config_.rate_limit = std::abs(limit); }

    void SetSpikeThreshold(double threshold) { config_.spike_threshold = threshold; }

    // 主滤波函数
    IMU Filter(const IMU &raw_data) {
        IMU filtered = raw_data;

        // 1. 更新缓冲区
        updateBuffer(raw_data);

        // 2. 多级滤波处理
        filtered.angular_velocity.x() = processAxis(raw_data.angular_velocity.x(), gyro_x_history_, 0);
        filtered.angular_velocity.y() = processAxis(raw_data.angular_velocity.y(), gyro_y_history_, 1);
        filtered.angular_velocity.z() = processAxis(raw_data.angular_velocity.z(), gyro_z_history_, 2);

        // 3. 速率限制（防止突变）
        if (prev_filtered_.timestamp > 0) {
            double dt = raw_data.timestamp - prev_filtered_.timestamp;
            if (dt > 0 && dt < 0.1) {  // 合理的dt范围
                filtered.angular_velocity.x() =
                    rateLimit(filtered.angular_velocity.x(), prev_filtered_.angular_velocity.x(), dt);
                filtered.angular_velocity.y() =
                    rateLimit(filtered.angular_velocity.y(), prev_filtered_.angular_velocity.y(), dt);
                filtered.angular_velocity.z() =
                    rateLimit(filtered.angular_velocity.z(), prev_filtered_.angular_velocity.z(), dt);
            }
        }

        // 4. 更新统计信息（用于自适应阈值）
        updateStatistics(filtered);

        prev_filtered_ = filtered;
        return filtered;
    }

   private:
    // 处理单个轴的数据
    double processAxis(double raw_value, std::deque<double> &history, int axis_idx) {
        double filtered = raw_value;

        // 1. 毛刺检测与替换
        if (history.size() >= config_.median_window_size && sample_count_ > 100) {
            filtered = detectAndRemoveSpike(raw_value, history, axis_idx);
        }

        // 2. 中值滤波
        filtered = medianFilter(filtered, history);

        // 3. 移动平均平滑
        filtered = movingAverage(filtered, history);

        return filtered;
    }

    // 更新缓冲区
    void updateBuffer(const IMU &data) {
        // 更新各轴历史数据
        gyro_x_history_.push_back(data.angular_velocity.x());
        gyro_y_history_.push_back(data.angular_velocity.y());
        gyro_z_history_.push_back(data.angular_velocity.z());

        // 限制历史数据长度
        int max_history = std::max({config_.median_window_size, config_.moving_avg_window, 10});
        while (gyro_x_history_.size() > max_history) {
            gyro_x_history_.pop_front();
            gyro_y_history_.pop_front();
            gyro_z_history_.pop_front();
        }
    }

    // 毛刺检测与替换
    double detectAndRemoveSpike(double value, std::deque<double> &history, int axis_idx) {
        if (history.size() < config_.median_window_size) {
            return value;
        }

        // 计算当前窗口的中位数
        std::vector<double> window(history.end() - config_.median_window_size, history.end());
        std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
        double median = window[window.size() / 2];

        // 计算与中位数的偏差
        double diff = std::abs(value - median);

        // 自适应阈值
        double threshold = config_.spike_threshold * gyro_std_[axis_idx];
        if (config_.enable_adaptive && gyro_std_[axis_idx] > 0) {
            threshold = std::max(threshold, config_.spike_threshold * 0.5);
        }

        // 如果是毛刺，用中位数替换
        if (diff > threshold) {
            LOG(INFO) << "find imu spike: " << diff << ", " << threshold;
            return median;
        }

        return value;
    }

    // 中值滤波
    double medianFilter(double value, std::deque<double> &history) {
        if (history.size() < config_.median_window_size) {
            return value;
        }

        // 取最近N个值进行中值滤波
        std::vector<double> window(history.end() - config_.median_window_size, history.end());
        std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
        return window[window.size() / 2];
    }

    // 移动平均
    double movingAverage(double value, std::deque<double> &history) {
        if (history.size() < config_.moving_avg_window) {
            return value;
        }

        double sum = 0;
        auto it = history.end() - config_.moving_avg_window;
        for (; it != history.end(); ++it) {
            sum += *it;
        }
        return sum / config_.moving_avg_window;
    }

    // 速率限制
    double rateLimit(double current, double previous, double dt) {
        double max_change = config_.rate_limit * dt;
        double diff = current - previous;

        if (std::abs(diff) > max_change) {
            return previous + (diff > 0 ? max_change : -max_change);
        }
        return current;
    }

    // 更新统计信息
    void updateStatistics(const IMU &data) {
        const double alpha = 0.01;  // 指数移动平均系数

        if (sample_count_ == 0) {
            gyro_mean_[0] = data.angular_velocity[0];
            gyro_mean_[1] = data.angular_velocity[1];
            gyro_mean_[2] = data.angular_velocity[2];
            gyro_std_[0] = 0.1;  // 初始值
            gyro_std_[1] = 0.1;
            gyro_std_[2] = 0.1;
        } else {
            // 更新均值
            double prev_mean[3] = {gyro_mean_[0], gyro_mean_[1], gyro_mean_[2]};
            gyro_mean_[0] = (1 - alpha) * gyro_mean_[0] + alpha * data.angular_velocity[0];
            gyro_mean_[1] = (1 - alpha) * gyro_mean_[1] + alpha * data.angular_velocity[1];
            gyro_mean_[2] = (1 - alpha) * gyro_mean_[2] + alpha * data.angular_velocity[2];

            // 更新标准差
            gyro_std_[0] = (1 - alpha) * gyro_std_[0] + alpha * std::abs(data.angular_velocity[0] - gyro_mean_[0]);
            gyro_std_[1] = (1 - alpha) * gyro_std_[1] + alpha * std::abs(data.angular_velocity[1] - gyro_mean_[1]);
            gyro_std_[2] = (1 - alpha) * gyro_std_[2] + alpha * std::abs(data.angular_velocity[2] - gyro_mean_[2]);
        }

        sample_count_++;
    }
};
}  // namespace lightning

#endif  // LIGHTNING_IMU_FILTER_H
