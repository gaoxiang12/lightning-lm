ament_add_gtest(test_navigation_interfaces
    test_navigation_interfaces.cc
)
target_link_libraries(test_navigation_interfaces lightning.libs)
target_include_directories(test_navigation_interfaces PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/core/miao
)
ament_target_dependencies(test_navigation_interfaces geometry_msgs nav_msgs sensor_msgs)
