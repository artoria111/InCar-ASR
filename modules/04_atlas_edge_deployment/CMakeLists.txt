cmake_minimum_required(VERSION 3.14)
project(car-asr-engine VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================
# AscendCL — CANN 7.0.RC1
# ============================================================
set(ASCEND_TOOLKIT_HOME "$ENV{ASCEND_HOME}"
    CACHE PATH "Ascend CANN toolkit root")
if(NOT ASCEND_TOOLKIT_HOME)
    set(ASCEND_TOOLKIT_HOME /usr/local/Ascend/ascend-toolkit/latest)
endif()

set(ACLLIB_DIR      ${ASCEND_TOOLKIT_HOME}/acllib)
set(ATLAS_INC_DIR   ${ACLLIB_DIR}/include)
set(ATLAS_LIB_DIR   ${ACLLIB_DIR}/lib64)
set(TOOLKIT_LIB_DIR ${ASCEND_TOOLKIT_HOME}/lib64)

# AscendCL libraries
find_library(ASCENDCL_LIB  ascendcl  PATHS ${ATLAS_LIB_DIR}   REQUIRED)
find_library(ACL_CBLAS_LIB acl_cblas PATHS ${ATLAS_LIB_DIR}   REQUIRED)

message(STATUS "AscendCL include:  ${ATLAS_INC_DIR}")
message(STATUS "AscendCL lib:      ${ATLAS_LIB_DIR}")
message(STATUS "AscendCL lib (toolkit): ${TOOLKIT_LIB_DIR}")

# ============================================================
# Optional: WebRTC VAD
# ============================================================
find_library(WEBRTC_VAD_LIB webrtc-audio-processing
    PATHS /usr/lib /usr/local/lib
    HINTS /usr/lib/aarch64-linux-gnu)
if(NOT WEBRTC_VAD_LIB)
    message(WARNING "WebRTC VAD not found — using energy-based fallback")
else()
    add_compile_definitions(HAS_WEBRTC_VAD)
    message(STATUS "WebRTC VAD:        ${WEBRTC_VAD_LIB}")
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
add_library(carasr SHARED ${ENGINE_SOURCES} ${ENGINE_HEADERS})
target_include_directories(carasr
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ATLAS_INC_DIR}
)
target_link_libraries(carasr
    PUBLIC
        ${ASCENDCL_LIB}
        ${ACL_CBLAS_LIB}
        pthread
        dl
)
if(WEBRTC_VAD_LIB)
    target_link_libraries(carasr PUBLIC ${WEBRTC_VAD_LIB})
endif()

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
        ${ACL_CBLAS_LIB}
        pthread
        dl
)
install(TARGETS acl-hello RUNTIME DESTINATION bin)
target_include_directories(car-asr-cli
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ATLAS_INC_DIR}
)
target_link_libraries(car-asr-cli
    PRIVATE
        carasr
)

# ============================================================
# Install
# ============================================================
install(TARGETS carasr car-asr-cli
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
install(FILES ${ENGINE_HEADERS} DESTINATION include/car-asr)

message(STATUS "car-asr-engine v${PROJECT_VERSION} configured")
