// test_memory.cpp — Memory subsystem tests

#include "jllm_memory.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

int main() {
    // Initialize CUDA (required for cudaMallocHost in ScratchPool)
    cudaSetDevice(0);
    // Test 1: probe system memory
    auto budget = jllm::probe_system_memory();
    assert(budget.total_mb > 0);
    assert(budget.free_mb() > 0);
    budget.print();
    printf("PASS: probe_system_memory\n");

    // Test 2: OOM guard
    jllm::OOMGuard guard(256);
    assert(guard.real_free_mb() > 0);
    printf("Free: %ld MB\n", guard.real_free_mb());
    printf("PASS: OOMGuard\n");

    // Test 3: scratch pool
    jllm::ScratchPool scratch;
    assert(scratch.init(64 * 1024 * 1024));  // 64 MB
    void* a = scratch.get(1024);
    void* b = scratch.get(2048);
    assert(a != nullptr && b != nullptr);
    assert(a != b);
    // get() aligns to 256: 1024→1024 (already aligned), 2048→2048
    assert(scratch.used() == 1024 + 2048);
    scratch.reset();
    assert(scratch.used() == 0);
    scratch.destroy();
    printf("PASS: ScratchPool\n");

    printf("\nAll memory tests passed.\n");
    return 0;
}
