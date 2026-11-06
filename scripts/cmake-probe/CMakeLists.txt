# cmake-probe: a minimal project that performs EXACTLY the find_package() calls made by
# a2/CMakeLists.txt, with the same version numbers, and builds a C++23 smoke program against them.
#
# It exists so that dependency discovery can be verified without the full a2 build (which also
# needs FetchContent of nanobench and the autodiff 1.1.3 requirement, see README "Known blockers").
#
# autodiff is NOT looked up unless A2_PROBE_AUTODIFF_VERSION is set. The a2 project requests
# 1.1.3, which does not exist upstream; verify-env.sh configures this probe twice (1.1.3 expected
# to FAIL with the CMake error from find_package; 1.1.2 expected to PASS) and reports both.
cmake_minimum_required(VERSION 3.31)

project(a2_cmake_probe LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(A2_PROBE_AUTODIFF_VERSION "" CACHE STRING
    "If set (e.g. 1.1.3 or 1.1.2), also run find_package(autodiff <version> REQUIRED)")
set(A2_DATA_DIR "" CACHE PATH "Directory containing indoor/ and outdoor/ videos (read only)")

# ---- exactly the discovery calls of a2/CMakeLists.txt ----------------------------------------
# Versions/locations are captured IMMEDIATELY after each find_package(): later packages (e.g. VTK's
# own module finder) can re-run find_package(Eigen3 ...) and overwrite the variables.
function(a2_report name version location)
    message(STATUS "A2PROBE ${name} ${version} ${location}")
endfunction()

find_package(doctest REQUIRED)
a2_report(doctest "${doctest_VERSION}" "${doctest_DIR}")

find_package(Boost 1.74 REQUIRED)
a2_report(Boost "${Boost_VERSION_STRING}" "${Boost_DIR}")
a2_report(Boost_include "${Boost_INCLUDE_DIR}" "-")

find_package(Eigen3 5.0 REQUIRED NO_MODULE)
a2_report(Eigen3 "${Eigen3_VERSION}" "${Eigen3_DIR}")

find_package(OpenCV 5.0 REQUIRED)
a2_report(OpenCV "${OpenCV_VERSION}" "${OpenCV_DIR}")

find_package(VTK 9.5 REQUIRED)
a2_report(VTK "${VTK_VERSION}" "${VTK_DIR}")

find_package(onnxruntime REQUIRED)
a2_report(onnxruntime "${onnxruntime_VERSION}" "${onnxruntime_DIR}")

find_package(Doxygen QUIET)     # optional, as in a2
a2_report(Doxygen "${DOXYGEN_VERSION}" "${DOXYGEN_EXECUTABLE}")

if(A2_PROBE_AUTODIFF_VERSION)
    find_package(autodiff ${A2_PROBE_AUTODIFF_VERSION} REQUIRED)
    a2_report(autodiff "${autodiff_VERSION}" "${autodiff_DIR}")
endif()

# ---- smoke program -----------------------------------------------------------------------------
add_executable(a2_smoke src/smoke.cpp)
target_link_libraries(a2_smoke PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${VTK_LIBRARIES}
    onnxruntime::onnxruntime)
target_include_directories(a2_smoke PRIVATE ${Boost_INCLUDE_DIR})
# Same VTK module auto-initialisation as a2/CMakeLists.txt
vtk_module_autoinit(TARGETS a2_smoke MODULES ${VTK_LIBRARIES})
target_compile_definitions(a2_smoke PRIVATE A2_DATA_DIR="${A2_DATA_DIR}")

# doctest self-test (exercises the doctest package and ctest integration)
add_executable(a2_doctest_probe src/doctest_probe.cpp)
target_link_libraries(a2_doctest_probe PRIVATE doctest::doctest)
enable_testing()
add_test(NAME a2_doctest_probe COMMAND a2_doctest_probe --no-intro)

if(A2_PROBE_AUTODIFF_VERSION)
    add_executable(a2_autodiff_smoke src/autodiff_smoke.cpp)
    target_link_libraries(a2_autodiff_smoke PRIVATE autodiff::autodiff Eigen3::Eigen)
endif()
