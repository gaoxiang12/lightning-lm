//
// Created by xiang on 2022/6/21.
//

#include "common/nav_state.h"

namespace lightning {
/// 矢量变量的维度
const std::vector<NavState::MetaInfo> NavState::vect_states_{
    {NavState::kPosIdx, NavState::kPosIdx, NavState::kBlockDim},  // pos
    {NavState::kVelIdx, NavState::kVelIdx, NavState::kBlockDim},  // vel
    {NavState::kBgIdx, NavState::kBgIdx, NavState::kBlockDim},    // bg
};

/// SO3 变量的维度
const std::vector<NavState::MetaInfo> NavState::SO3_states_{
    {NavState::kRotIdx, NavState::kRotIdx, NavState::kBlockDim},  // rot
};

}  // namespace lightning
