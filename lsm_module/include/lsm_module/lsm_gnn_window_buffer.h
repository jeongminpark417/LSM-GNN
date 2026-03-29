#ifndef LSM_MODULE_LSM_GNN_WINDOW_BUFFER_H
#define LSM_MODULE_LSM_GNN_WINDOW_BUFFER_H

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#include <cstdint>
#include <stdio.h>

template <typename T = float>
__forceinline__ __device__ void write_to_queue(void* src, void* dst, size_t size, uint32_t mask) {
    T* src_ptr = (T*)src;
    T* dst_ptr = (T*)dst;

    uint32_t count = __popc(mask);
    if (count == 0 || dst_ptr == nullptr || src_ptr == nullptr)
        return;
    uint32_t lane_id = threadIdx.x % 32;

    uint32_t my_id = count - (__popc(mask >> (lane_id)));
    for (; my_id < size / sizeof(T); my_id += count) {
        dst_ptr[my_id] = src_ptr[my_id];
    }
}

#endif
