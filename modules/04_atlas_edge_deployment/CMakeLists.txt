cmake_minimum_required(VERSION 3.14)
project(car-asr-engine VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(CAR_ASR_BUILD_ENGINE "Build the AscendCL engine and CLI" ON)
option(CAR_ASR_BUILD_HOST_TESTS "Build hardware-independent unit tests" ON)

# ============================================================
# AscendCL — CANN 7.0.RC1
# ============================================================
if(CAR_ASR_BUILD_ENGINE)
    set(ASCEND_TOOLKIT_HOME "$ENV{ASCEND_HOME}"
        CACHE PATH "Ascend CANN toolkit root")
    if(NOT ASCEND_TOOLKIT_HOME)
        set(ASCEND_TOOLKIT_HOME /usr/local/Ascend/ascend-toolkit/latest)
    endif()

    find_path(ATLAS_INC_DIR acl/acl.h
        PATHS
            ${ASCEND_TOOLKIT_HOME}/include
            ${ASCEND_TOOLKIT_HOME}/acllib/include
        REQUIRED
    )
    find_library(ASCENDCL_LIB ascendcl
        PATHS
            ${ASCEND_TOOLKIT_HOME}/lib64
            ${ASCEND_TOOLKIT_HOME}/runtime/lib64
            ${ASCEND_TOOLKIT_HOME}/acllib/lib64
        REQUIRED
    )

    message(STATUS "AscendCL include: ${ATLAS_INC_DIR}")
    message(STATUS "AscendCL library: ${ASCENDCL_LIB}")
endif()

# ============================================================
# Sources
# ============================================================
set(ENGINE_SOURCES
    src/asr_engine.cpp
    src/ascend_inference.cpp
    src/audio_preprocess.cpp
    src/vad_detector.cpp
    src/ctc_decoder.cpp
    src/utils.cpp
)

set(ENGINE_HEADERS
    include/common.h
    include/asr_engine.h
    include/ascend_inference.h
    include/audio_preprocess.h
    include/vad_detector.h
    include/ctc_decoder.h
)

# ============================================================
# Main library — libcarasr.so
# ============================================================
if(CAR_ASR_BUILD_ENGINE)
    add_library(carasr SHARED ${ENGINE_SOURCES} ${ENGINE_HEADERS})
    target_include_directories(carasr
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ATLAS_INC_DIR}
    )
    target_link_libraries(carasr
        PUBLIC
            ${ASCENDCL_LIB}
            pthread
            dl
    )

# ============================================================
# CLI Demo executable
# ============================================================
    add_executable(car-asr-cli src/main.cpp)

# ============================================================
# ACL Hello World — NPU 链路最小验证
# ============================================================
    add_executable(acl-hello src/acl_hello.cpp)
    target_include_directories(acl-hello
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ATLAS_INC_DIR}
    )
    target_link_libraries(acl-hello
        PRIVATE
            ${ASCENDCL_LIB}
            pthread
            dl
    )
    install(TARGETS acl-hello RUNTIME DESTINATION bin)
    target_include_directories(car-asr-cli
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${ATLAS_INC_DIR}
    )
    target_link_libraries(car-asr-cli PRIVATE carasr)

# ============================================================
# Install
# ============================================================
    install(TARGETS carasr car-asr-cli
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
    )
    install(FILES ${ENGINE_HEADERS} DESTINATION include/car-asr)
endif()

if(CAR_ASR_BUILD_HOST_TESTS)
    enable_testing()
    add_executable(car-asr-host-tests
        tests/host_unit.cpp
        src/audio_preprocess.cpp
        src/vad_detector.cpp
        src/ctc_decoder.cpp
    )
    target_include_directories(car-asr-host-tests
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    add_test(NAME car-asr-host-tests COMMAND car-asr-host-tests)
endif()

message(STATUS "car-asr-engine v${PROJECT_VERSION} configured")
