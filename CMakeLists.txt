# Minimum CMake vesion: limited by CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
cmake_minimum_required(VERSION 3.4)

if("$ENV{ROS_VERSION}" STREQUAL "2")
	set(DETECTED_ROS2 TRUE)
endif()

# Tell CMake we'll use C++ for use in its tests/flags
project(mola_kiss_icp_wrapper LANGUAGES CXX)

# MOLA CMake scripts: "mola_xxx()"
find_package(mola_common REQUIRED)

# find dependencies:
find_package(mrpt-maps REQUIRED)
find_package(mrpt-tclap REQUIRED)
find_mola_package(mola_kernel)

# CLI apps
# ---------------------------------
find_package(mola_input_kitti_dataset) # optional
find_package(mola_input_rawlog) # optional
find_package(mola_input_mulran_dataset) # optional
find_package(mola_input_rosbag2) # optional
find_package(mola_input_paris_luco_dataset) # optional

# kiss_icp interface:
mola_add_executable(
	TARGET  mola-lidar-odometry-cli-kiss
	SOURCES apps/mola-lidar-odometry-cli-kiss.cpp
	LINK_LIBRARIES
		mrpt::maps
		mrpt::tclap
		mola::mola_kernel
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/submodules/kiss-icp/cpp/kiss_icp ${CMAKE_CURRENT_BINARY_DIR}/kiss_icp)
target_include_directories(mola-lidar-odometry-cli-kiss PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/submodules/kiss-icp/include)
target_link_libraries(mola-lidar-odometry-cli-kiss kiss_icp::pipeline)

if (mola_input_kitti_dataset_FOUND)
target_link_libraries(mola-lidar-odometry-cli-kiss mola::mola_input_kitti_dataset)
target_compile_definitions(mola-lidar-odometry-cli-kiss PRIVATE HAVE_MOLA_INPUT_KITTI)
endif()

if (mola_input_mulran_dataset_FOUND)
target_link_libraries(mola-lidar-odometry-cli-kiss mola::mola_input_mulran_dataset)
target_compile_definitions(mola-lidar-odometry-cli-kiss PRIVATE HAVE_MOLA_INPUT_MULRAN)
endif()

if (mola_input_rawlog_FOUND)
target_link_libraries(mola-lidar-odometry-cli-kiss mola::mola_input_rawlog)
target_compile_definitions(mola-lidar-odometry-cli-kiss PRIVATE HAVE_MOLA_INPUT_RAWLOG)
endif()

if (mola_input_rosbag2_FOUND)
target_link_libraries(mola-lidar-odometry-cli-kiss mola::mola_input_rosbag2)
target_compile_definitions(mola-lidar-odometry-cli-kiss PRIVATE HAVE_MOLA_INPUT_ROSBAG2)
endif()

if (mola_input_paris_luco_dataset_FOUND)
target_link_libraries(mola-lidar-odometry-cli-kiss mola::mola_input_paris_luco_dataset)
target_compile_definitions(mola-lidar-odometry-cli-kiss PRIVATE HAVE_MOLA_INPUT_PARIS_LUCO)
endif()

# -----------------------------------------------------------------------------
#  ROS2
# -----------------------------------------------------------------------------
if (DETECTED_ROS2)
	# find dependencies
	find_package(ament_cmake REQUIRED)

	#ament_target_dependencies(MY_TARGET
	#  rclcpp...
	#)

	ament_package()
endif()
