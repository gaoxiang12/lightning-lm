//
// Created by MiMo on 25-7-21.
//

#include "core/loop_closing/bevplace2_loop_detector.h"
#include "io/yaml_io.h"

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace lightning {

void BEVPlace2LoopDetector::Init(const std::string& yaml_path) {
    if (!yaml_path.empty()) {
        YAML_IO yio(yaml_path);
        match_threshold_ = yio.GetValueOr<double>("bevplace2", "descriptor_match_th", 1.5);
        min_id_gap_ = yio.GetValueOr<int>("bevplace2", "min_id_gap", 50);

        BEVGenerator::Options bev_opts;
        bev_opts.voxel_size_ = yio.GetValueOr<double>("bevplace2", "bev_resolution", 0.4);
        bev_opts.range_ = yio.GetValueOr<double>("bevplace2", "bev_range", 40.0);
        bev_opts.bev_size_ = yio.GetValueOr<int>("bevplace2", "bev_size", 200);
        bev_gen_ = BEVGenerator(bev_opts);
    }

    backend_ready_ = StartInferenceBackend();
    if (!backend_ready_) {
        LOG(ERROR) << "BEVPlace2: Python inference backend not available, detector disabled";
    }
}

bool BEVPlace2LoopDetector::StartInferenceBackend() {
    std::string script_path = __FILE__;
    auto pos = script_path.rfind('/');
    if (pos != std::string::npos) {
        script_path = script_path.substr(0, pos);
    }
    script_path += "/bevplace2_inference.py";

    FILE* check = fopen(script_path.c_str(), "r");
    if (!check) {
        LOG(ERROR) << "BEVPlace2 inference script not found: " << script_path;
        return false;
    }
    fclose(check);

    int stdin_pipe[2], stdout_pipe[2];
    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
        LOG(ERROR) << "BEVPlace2: failed to create pipes";
        return false;
    }

    python_pid_ = fork();
    if (python_pid_ == 0) {
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        execlp("python3", "python3", script_path.c_str(), (char*)nullptr);
        _exit(1);
    }

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    python_stdin_ = fdopen(stdin_pipe[1], "w");
    python_stdout_ = fdopen(stdout_pipe[0], "r");

    if (!python_stdin_ || !python_stdout_) {
        return false;
    }

    char buf[256];
    if (fgets(buf, sizeof(buf), python_stdout_)) {
        std::string line(buf);
        if (line.find("ready") != std::string::npos) {
            LOG(INFO) << "BEVPlace2 inference backend ready";
            return true;
        }
    }
    return false;
}

static std::string Base64Encode(const unsigned char* data, size_t len) {
    static const char encoding[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned int n = static_cast<unsigned int>(data[i]) << 16;
        if (i + 1 < len) n |= static_cast<unsigned int>(data[i + 1]) << 8;
        if (i + 2 < len) n |= static_cast<unsigned int>(data[i + 2]);
        result += encoding[(n >> 18) & 0x3F];
        result += encoding[(n >> 12) & 0x3F];
        result += (i + 1 < len) ? encoding[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? encoding[n & 0x3F] : '=';
    }
    return result;
}

static std::vector<float> Base64DecodeFloats(const std::string& b64) {
    static const unsigned char dtable[256] = {
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,62,64,64,64,63,
        52,53,54,55,56,57,58,59,60,61,64,64,64,64,64,64,
        64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,64,64,64,64,64,
        64,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64
    };

    std::vector<unsigned char> decoded;
    decoded.reserve(b64.size() * 3 / 4);
    unsigned int accum = 0;
    int bits = 0;
    for (char c : b64) {
        if (c == '=') break;
        unsigned char val = dtable[static_cast<unsigned char>(c)];
        if (val >= 64) continue;
        accum = (accum << 6) | val;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            decoded.push_back(static_cast<unsigned char>((accum >> bits) & 0xFF));
        }
    }

    size_t n_floats = decoded.size() / sizeof(float);
    std::vector<float> desc(n_floats);
    std::memcpy(desc.data(), decoded.data(), n_floats * sizeof(float));
    return desc;
}

std::vector<float> BEVPlace2LoopDetector::ExtractDescriptor(CloudPtr cloud) {
    if (!backend_ready_) {
        return {};  // 返回空 vector 表示失败
    }

    cv::Mat bev = bev_gen_.Generate(cloud);

    std::vector<unsigned char> raw;
    raw.push_back(static_cast<unsigned char>(bev.rows & 0xFF));
    raw.push_back(static_cast<unsigned char>((bev.rows >> 8) & 0xFF));
    raw.push_back(static_cast<unsigned char>(bev.cols & 0xFF));
    raw.push_back(static_cast<unsigned char>((bev.cols >> 8) & 0xFF));
    for (int r = 0; r < bev.rows; r++) {
        for (int c = 0; c < bev.cols; c++) {
            raw.push_back(bev.at<uchar>(r, c));
        }
    }

    std::string b64 = Base64Encode(raw.data(), raw.size());

    fprintf(python_stdin_, "{\"cmd\":\"extract\",\"data\":\"%s\"}\n", b64.c_str());
    fflush(python_stdin_);

    char resp_buf[1024 * 100];
    if (!fgets(resp_buf, sizeof(resp_buf), python_stdout_)) {
        LOG(ERROR) << "BEVPlace2: failed to read inference response";
        return {};
    }

    std::string resp(resp_buf);
    auto desc_pos = resp.find("\"desc\":\"");
    if (desc_pos == std::string::npos) {
        LOG(ERROR) << "BEVPlace2: invalid response format";
        return {};
    }
    desc_pos += 8;
    auto desc_end = resp.find("\"", desc_pos);
    std::string desc_b64 = resp.substr(desc_pos, desc_end - desc_pos);

    auto desc = Base64DecodeFloats(desc_b64);
    if (desc.size() != 8192) {
        LOG(ERROR) << "BEVPlace2: unexpected descriptor dimension: " << desc.size();
        return {};
    }

    return desc;
}

void BEVPlace2LoopDetector::AddKeyframe(Keyframe::Ptr kf) {
    if (!backend_ready_) {
        return;  // 后端不可用，跳过
    }

    // 如果是当前帧，使用缓存
    if (kf == cur_frame_kf_ && !cur_frame_desc_.empty()) {
        database_descs_.push_back(cur_frame_desc_);
        database_kfs_.push_back(kf);
        return;
    }

    auto desc = ExtractDescriptor(kf->GetCloud());
    if (desc.empty()) {
        LOG(WARNING) << "BEVPlace2: failed to extract descriptor for kf " << kf->GetID();
        return;  // 提取失败，不加入数据库
    }
    database_descs_.push_back(desc);
    database_kfs_.push_back(kf);
}

std::vector<LoopCandidate> BEVPlace2LoopDetector::Detect(Keyframe::Ptr cur_kf) {
    std::vector<LoopCandidate> candidates;

    if (!backend_ready_) {
        return candidates;
    }

    if (database_descs_.size() < 2) {
        return candidates;
    }

    // 提取当前帧描述子（只提取一次）
    auto query_desc = ExtractDescriptor(cur_kf->GetCloud());
    if (query_desc.empty()) {
        return candidates;
    }

    // 缓存当前帧描述子，供 AddKeyframe 使用
    cur_frame_desc_ = query_desc;
    cur_frame_kf_ = cur_kf;

    int skip = std::min(static_cast<int>(database_descs_.size()), min_id_gap_);

    float best_dist = std::numeric_limits<float>::max();
    int best_idx = -1;

    for (size_t i = 0; i + skip < database_descs_.size(); i++) {
        float dist = 0.0f;
        for (size_t j = 0; j < query_desc.size() && j < database_descs_[i].size(); j++) {
            float d = query_desc[j] - database_descs_[i][j];
            dist += d * d;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = static_cast<int>(i);
        }
    }

    // match_threshold_ 是欧氏距离，比较平方距离
    float th_sq = match_threshold_ * match_threshold_;
    if (best_idx >= 0 && best_dist < th_sq) {
        auto matched_kf = database_kfs_[best_idx];
        LoopCandidate c(matched_kf->GetID(), cur_kf->GetID());
        c.Tij_ = matched_kf->GetLIOPose().inverse() * cur_kf->GetLIOPose();
        c.ndt_score_ = 1.0;  // 描述子匹配成功
        candidates.push_back(c);

        LOG(INFO) << "BEVPlace2: loop candidate, dist=" << std::sqrt(best_dist)
                  << " ids=" << matched_kf->GetID() << " -> " << cur_kf->GetID();
    }

    return candidates;
}

}  // namespace lightning
