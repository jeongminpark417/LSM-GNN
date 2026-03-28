#ifndef LSM_MODULE_DEVICE_PTR_CUH
#define LSM_MODULE_DEVICE_PTR_CUH

/*
 * Shims for kernels written against an extended BAM fork. Upstream bam_ptr /
 * array_d_t omit write-back and prefetch hooks used in gids_nvme.cu.
 */

#include <page_cache.h>
#include <cstdint>

template <typename T>
struct lsm_bam_ptr : bam_ptr<T> {
  __host__ __device__ lsm_bam_ptr(array_d_t<T>* a) : bam_ptr<T>(a) {}

  __device__ void set_prefetch_val(uint64_t page_byte_offset, uint8_t p_val) {
    (void)page_byte_offset;
    (void)p_val;
  }

  __device__ void set_window_buffer_counter(uint64_t page_byte_offset, uint8_t p_val) {
    (void)page_byte_offset;
    (void)p_val;
  }
};

template <typename T>
struct wb_bam_ptr : lsm_bam_ptr<T> {
  __host__ __device__ wb_bam_ptr(array_d_t<T>* a) : lsm_bam_ptr<T>(a) {}

  __device__ void set_wb(uint32_t* wb_queue_counter, uint32_t wb_depth, T* queue_ptr,
                         uint64_t* wb_id_array, uint32_t q_depth) {
    (void)wb_queue_counter;
    (void)wb_depth;
    (void)queue_ptr;
    (void)wb_id_array;
    (void)q_depth;
  }

  __device__ void set_time(uint8_t time_step, uint32_t head_ptr) {
    (void)time_step;
    (void)head_ptr;
  }

  __device__ void update_wb(uint64_t cur_node, uint32_t cur_iter, int32_t cur_batch_node) {
    (void)cur_node;
    (void)cur_iter;
    (void)cur_batch_node;
  }

  __device__ void update_wb_list(uint64_t cur_node, uint32_t cur_iter, int32_t cur_batch_node) {
    (void)cur_node;
    (void)cur_iter;
    (void)cur_batch_node;
  }

  __device__ void update_wb(uint64_t cur_node, uint64_t cur_iter) {
    (void)cur_node;
    (void)cur_iter;
  }

  __device__ void flush_wb_counter(uint64_t tid) { (void)tid; }

  __device__ void count_mask(uint64_t tid, uint64_t* counter) {
    (void)tid;
    (void)counter;
  }

  __device__ uint64_t get_page_id(uint64_t tid) {
    (void)tid;
    return 0;
  }
};

#endif
