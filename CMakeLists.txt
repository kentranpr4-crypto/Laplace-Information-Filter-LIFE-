cmake_minimum_required(VERSION 3.31)

# Suppress CMake developer warnings (e.g., from third-party find_package config files)
set(CMAKE_WARN_DEPRECATED FALSE CACHE BOOL "Warn about deprecated functionality")
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE CACHE BOOL "Suppress developer warnings")

# Set the project name
file(STRINGS project.txt PRJ_NAME LIMIT_COUNT 1)
project(${PRJ_NAME})

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# CXX flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall) # -Wextra -Wpedantic
add_compile_options("$<$<CONFIG:Debug>:-g;-O0>")
add_compile_options("$<$<CONFIG:Release>:-O3>")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    # add_compile_definitions(_GLIBCXX_DEBUG)             # Note: This needs all other dependencies to also be built in debug mode
    # add_compile_definitions(LIBCXX_ENABLE_DEBUG_MODE)
    add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_NAN)
else()
    add_compile_definitions(NDEBUG)
    add_compile_definitions(EIGEN_NO_DEBUG)
endif()

# Force coloured compiler output
set(CMAKE_COLOR_DIAGNOSTICS ON)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")          # match Clang or AppleClang
    # Support colours for ninja + clang under MSYS2
    # (see https://github.com/msys2/MINGW-packages/issues/2988#issuecomment-1425591134)
    add_compile_options(-fansi-escape-codes)
    add_link_options(-fansi-escape-codes)
endif()

# Workaround for Qt5 being unlinked under homebrew
if(NOT DEFINED ENV{Qt5_DIR} OR ENV{Qt5_DIR} STREQUAL "Qt5_DIR-NOTFOUND")
    execute_process(COMMAND brew --prefix qt@5 OUTPUT_VARIABLE PREFIX_QT5 OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(ENV{Qt5_DIR} ${PREFIX_QT5}/lib/cmake/Qt5)
endif()

include(FetchContent)

FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.3.11
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(nanobench)

find_package(doctest REQUIRED)
find_package(Boost 1.74 REQUIRED)
find_package(Eigen3 5.0 REQUIRED NO_MODULE)
find_package(autodiff 1.1.2 REQUIRED)   # User decision B1: 1.1.3 does not exist upstream; latest release is v1.1.2
find_package(OpenCV 5.0 REQUIRED)
find_package(VTK 9.5 REQUIRED)

# conda-forge's HDF5 (pulled in transitively by VTK) exports NDEBUG and _FORTIFY_SOURCE=2 through
# INTERFACE_COMPILE_DEFINITIONS. That silently disabled assert() in Debug builds of this project.
# Strip them; NDEBUG is still defined for Release builds below.
foreach(_a2_hdf5_target HDF5::HDF5 hdf5::hdf5 hdf5::hdf5-shared hdf5::hdf5_hl hdf5::hdf5_hl-shared)
    if(TARGET ${_a2_hdf5_target})
        get_target_property(_a2_defs ${_a2_hdf5_target} INTERFACE_COMPILE_DEFINITIONS)
        if(_a2_defs)
            list(REMOVE_ITEM _a2_defs NDEBUG _FORTIFY_SOURCE=2)
            set_target_properties(${_a2_hdf5_target} PROPERTIES INTERFACE_COMPILE_DEFINITIONS "${_a2_defs}")
        endif()
    endif()
endforeach()
find_package(onnxruntime REQUIRED)
find_package(Doxygen QUIET)     # optional

option(BUILD_DOCUMENTATION "Build documentation" OFF)

if(BUILD_DOCUMENTATION)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        # Set input and output files
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        # Request to configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

        # Get all source files
        file(GLOB_RECURSE DOC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*)

        # Add doxygen target
        doxygen_add_docs(docs
            ${DOC_SOURCES} # Can't be a directory or wildcards if USE_STAMP_FILE is used (https://cmake.org/cmake/help/latest/module/FindDoxygen.html)
            CONFIG_FILE ${DOXYGEN_OUT}
            COMMENT "Generating API documentation with Doxygen"
            USE_STAMP_FILE
        )
    else()
        message(STATUS "Doxygen not found, documentation will not be built")
    endif()
endif()


include_directories(${VTK_PREFIX_PATH}/include/vtk-${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION})
include_directories(${VTK_PREFIX_PATH}/include/vtk) # Support VTK 9.x under MSYS2
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# Target names
set(TARGET_LIB common)
set(TARGET_BIN ${PROJECT_NAME})
set(TARGET_TEST tests)

# Libraries
set(LIBS
    ${Boost_LIBRARIES}
    Eigen3::Eigen
    autodiff::autodiff
    ${OpenCV_LIBS}
    ${VTK_LIBRARIES}
    nanobench
    onnxruntime::onnxruntime
)

# Source files
set(DIR_SRC src)
set(DIR_TEST_SRC test/src)

# Application
file(GLOB SRC_BIN CONFIGURE_DEPENDS ${DIR_SRC}/*.cpp ${DIR_SRC}/*.h ${DIR_SRC}/*.hpp)

# Static library
set(SRC_LIB ${SRC_BIN})
list(FILTER SRC_LIB EXCLUDE REGEX ".*main.cpp$")

# Unit tests
file(GLOB SRC_TEST CONFIGURE_DEPENDS ${DIR_TEST_SRC}/*.cpp ${DIR_TEST_SRC}/*.h ${DIR_TEST_SRC}/*.hpp)

# Targets

# Static library
if(SRC_LIB)
    add_library(${TARGET_LIB} ${SRC_LIB})
    target_link_libraries(${TARGET_LIB} PUBLIC ${LIBS})
endif()

# Application
add_executable(${TARGET_BIN} ${DIR_SRC}/main.cpp)
if(SRC_LIB)
    target_link_libraries(${TARGET_BIN} PRIVATE ${TARGET_LIB})
else()
    target_link_libraries(${TARGET_BIN} PRIVATE ${LIBS})
endif()

vtk_module_autoinit(TARGETS ${TARGET_BIN} MODULES ${VTK_LIBRARIES})

# Add documentation dependency if applicable
if(BUILD_DOCUMENTATION AND DOXYGEN_FOUND)
    add_dependencies(${TARGET_BIN} docs)
endif()

# Unit tests
add_executable(${TARGET_TEST} ${SRC_TEST})
if(SRC_LIB)
    target_link_libraries(${TARGET_TEST} PRIVATE ${TARGET_LIB})
else()
    target_link_libraries(${TARGET_TEST} PRIVATE ${LIBS})
endif()
target_link_libraries(${TARGET_TEST} PRIVATE doctest::doctest)

# Make test data location independent of the working directory tests are run from
target_compile_definitions(${TARGET_TEST} PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")

# Register unit tests with CTest
enable_testing()
add_test(NAME unit_tests COMMAND ${TARGET_TEST} --no-intro --force-colors=true)

# Run unit tests after building executables
add_custom_target(run_tests ALL
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS ${TARGET_TEST}
    DEPENDS ${TARGET_BIN}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests"
)

set_property(GLOBAL PROPERTY TARGET_MESSAGES OFF)
