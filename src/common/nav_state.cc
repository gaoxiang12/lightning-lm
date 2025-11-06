//
// Created by xiang on 2022/6/21.
//

#include "common/nav_state.h"

namespace lightning {
/// 矢量变量的维度
const std::vector<NavState::MetaInfo> NavState::vect_states_{
    {0, 0, 3},    // pos
    {9, 9, 3},    // offset t
    {12, 12, 3},  // vel
    {15, 15, 3},  // bg
    {18, 18, 3},  // ba
    {21, 21, 3},  // grav
};

/// SO3 变量的维度
const std::vector<NavState::MetaInfo> NavState::SO3_states_{
    {3, 3, 3},  // rot
    {6, 6, 3},  // offset_R_li
};

}  // namespace lightning
