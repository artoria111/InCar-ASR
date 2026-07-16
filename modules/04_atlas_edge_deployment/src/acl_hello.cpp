/**
 * acl_hello.cpp — AscendCL NPU 链路最小验证
 *
 * 验证：ACL Init → SetDevice → CreateContext → Malloc(H2D/D2H) → Free
 *       → DestroyContext → ResetDevice → Finalize
 *
 * 这是 AscendCL 的 "Hello World"，确认 NPU 运行时可用。
 */

#include "acl/acl.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

// ============================================================
// 宏：带检查的 ACL 调用
// ============================================================
#define ACL_CHECK(call, msg) do {                              \
    aclError _err = (call);                                     \
    if (_err != ACL_SUCCESS) {                                  \
        fprintf(stderr, "  [FAIL] %s (code=%d)\n", msg, _err); \
        return false;                                           \
    }                                                           \
    fprintf(stdout, "  [ OK ] %s\n", msg);                     \
} while(0)

// ============================================================
// 主链路
// ============================================================
static bool RunACLHello(int device_id) {
    fprintf(stdout, "\n============================================\n");
    fprintf(stdout, "  AscendCL Hello World — NPU Link Test\n");
    fprintf(stdout, "  Device: NPU %d\n", device_id);
    fprintf(stdout, "============================================\n\n");

    // ---- Step 1: ACL 全局初始化 ----
    fprintf(stdout, "-- Step 1: ACL Init --\n");
    ACL_CHECK(aclInit(nullptr), "aclInit()");

    // ---- Step 2: 查询设备数量 ----
    uint32_t dev_count = 0;
    aclError ret = aclrtGetDeviceCount(&dev_count);
    if (ret == ACL_SUCCESS) {
        fprintf(stdout, "  [INFO] NPU device count: %u\n", dev_count);
    }

    // ---- Step 3: 设置目标设备 ----
    fprintf(stdout, "\n-- Step 2: Set Device --\n");
    ACL_CHECK(aclrtSetDevice(device_id), "aclrtSetDevice()");

    // ---- Step 4: 查询运行模式 ----
    aclrtRunMode run_mode;
    ret = aclrtGetRunMode(&run_mode);
    if (ret == ACL_SUCCESS) {
        fprintf(stdout, "  [INFO] Run mode: %s\n",
                run_mode == ACL_DEVICE ? "NPU (ACL_DEVICE)" : "CPU (ACL_HOST)");
    }

    // ---- Step 5: 创建 Context ----
    fprintf(stdout, "\n-- Step 3: Create Context --\n");
    aclrtContext context = nullptr;
    ACL_CHECK(aclrtCreateContext(&context, device_id), "aclrtCreateContext()");

    // ---- Step 6: Device 内存分配 + H2D/D2H 数据校验 ----
    fprintf(stdout, "\n-- Step 4: Device Memory R/W Test --\n");

    constexpr size_t kBytes = 4096;           // 4 KB
    constexpr size_t kElems = kBytes / sizeof(float);  // 1024 floats

    // 6a. 分配 Device 内存
    void* dev_buf = nullptr;
    ACL_CHECK(aclrtMalloc(&dev_buf, kBytes, ACL_MEM_MALLOC_NORMAL_ONLY),
              "aclrtMalloc(4KB)");

    // 6b. 准备 Host 测试数据
    float host_send[kElems];
    float host_recv[kElems];
    for (size_t i = 0; i < kElems; i++) {
        host_send[i] = static_cast<float>(i) * 0.5f;
        host_recv[i] = 0.0f;
    }

    // 6c. Host → Device 拷贝
    ACL_CHECK(aclrtMemcpy(dev_buf, kBytes, host_send, kBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy(H→D, 4KB)");

    // 6d. Device → Host 拷贝
    ACL_CHECK(aclrtMemcpy(host_recv, kBytes, dev_buf, kBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy(D→H, 4KB)");

    // 6e. 数据完整性校验
    bool mismatch = false;
    for (size_t i = 0; i < kElems; i++) {
        if (host_send[i] != host_recv[i]) {
            fprintf(stderr, "  [FAIL] Data mismatch at [%zu]: sent=%.2f recv=%.2f\n",
                    i, host_send[i], host_recv[i]);
            mismatch = true;
            break;
        }
    }
    if (!mismatch) {
        fprintf(stdout, "  [ OK ] H→D→H data integrity: %zu floats matched\n", kElems);
    } else {
        aclrtFree(dev_buf);
        return false;
    }

    // 6f. 释放 Device 内存
    ACL_CHECK(aclrtFree(dev_buf), "aclrtFree()");

    // ---- Step 7: 销毁 Context ----
    fprintf(stdout, "\n-- Step 5: Destroy Context --\n");
    ACL_CHECK(aclrtDestroyContext(context), "aclrtDestroyContext()");

    // ---- Step 8: 重置设备 ----
    fprintf(stdout, "\n-- Step 6: Reset Device --\n");
    ACL_CHECK(aclrtResetDevice(device_id), "aclrtResetDevice()");

    // ---- Step 9: ACL 反初始化 ----
    fprintf(stdout, "\n-- Step 7: ACL Finalize --\n");
    ACL_CHECK(aclFinalize(), "aclFinalize()");

    // ---- 结果 ----
    fprintf(stdout, "\n============================================\n");
    fprintf(stdout, "  ALL CHECKS PASSED\n");
    fprintf(stdout, "  Ascend NPU Link:      VERIFIED\n");
    fprintf(stdout, "  Device Memory R/W:    VERIFIED (H→D→H)\n");
    fprintf(stdout, "  CANN Runtime:         OPERATIONAL\n");
    fprintf(stdout, "============================================\n\n");

    return true;
}

// ============================================================
int main(int argc, char* argv[]) {
    int device_id = (argc > 1) ? atoi(argv[1]) : 0;

    printf("AscendCL Hello World\n");
    printf("Target: NPU device %d\n\n", device_id);

    if (RunACLHello(device_id)) {
        printf("Result: PASS — Ascend NPU runtime fully operational.\n");
        return 0;
    } else {
        fprintf(stderr, "Result: FAIL — check CANN installation and device status.\n");
        return 1;
    }
}
