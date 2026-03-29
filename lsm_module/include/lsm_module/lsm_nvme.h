#ifndef LSM_MODULE_LSM_NVME_H
#define LSM_MODULE_LSM_NVME_H

#include <cstdint>
#include <memory>
#include <vector>

#include <lsm_module/lsm_gnn_page_cache.h>

#define LSM_NVME_TYPE float

struct LSM_NVMeQueueMapState;
LSM_NVMeQueueMapState *lsm_nvme_queue_map_state_new();
void lsm_nvme_queue_map_state_delete(LSM_NVMeQueueMapState *p) noexcept;

struct LSM_NVMeStore {
  const char *const ctrls_paths[5] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2",
                                      "/dev/libnvm3", "/dev/libnvm4"};

  uint32_t cudaDevice = 0;
  uint32_t nvmNamespace = 1;
  uint32_t n_ctrls = 1;

  size_t blkSize = 128;
  size_t numThreads = 64;
  size_t queueDepth = 1024;
  size_t numQueues = 128;
  size_t pageSize = 4096;
  uint64_t numElems = 0;

  std::vector<Controller *> ctrls;
  page_cache_t *h_pc = nullptr;
  range_t<LSM_NVME_TYPE> *h_range = nullptr;
  std::vector<range_t<LSM_NVME_TYPE> *> vr;
  array_t<LSM_NVME_TYPE> *a = nullptr;

  /**
   * PVP (prefetch-aware eviction): page_cache_t is built with is_pvp; host allocates
   * pinned_eviction_buffer; read_feature uses read_feature_pvp_kernel when is_pvp.
   */
  bool is_pvp = false;
  lsm_gnn::pinned_eviction_buffer<LSM_NVME_TYPE> pinned_eviction_;
  /** Effective geometry after init (0 args → lsm_gnn defaults). */
  uint32_t num_pvp_buffers = 0;
  /** Same depth for pinned eviction ring and device PVP meta/loc queues. */
  uint32_t pvp_queue_depth = 0;
  uint8_t eviction_time_step = 0;
  uint32_t eviction_head_ptr = 0;

  /**
   * Device-resident cuco::static_map (lazy), owned via queue_map_. Each
   * build_node_queue_index_map clears and rebuilds it.
   * Empty key is -1, erased key sentinel is -2 (for index_map_remove). The cuco empty-payload
   * pattern is INT32_MIN internally (must differ from semantic INT32_MAX). Value: INT32_MAX if the
   * node appears in only one batch index, or only with the same batch index, the value is
   * INT32_MAX; if it appears across multiple distinct batch indices, the value is the second-smallest
   * distinct batch index (smallest is the anchor). Same batch index repeated only → INT32_MAX.
   * Valid node ids must not be -1 or -2.
   */
  std::unique_ptr<LSM_NVMeQueueMapState, void (*)(LSM_NVMeQueueMapState *)>
      queue_map_{nullptr, lsm_nvme_queue_map_state_delete};
  uint32_t queue_map_time_step_ = 0;

  LSM_NVMeStore() = default;

  ~LSM_NVMeStore();

  /** Page-cache line count: ``cache_size_gb * 2^30 / page_size``. */
  void init_controllers(int ps, uint64_t read_off, uint64_t cache_size_gb,
                        uint64_t num_ele, uint64_t num_ssd = 1, bool is_pvp_enable = false,
                        uint32_t num_pvp_buffers_arg = 0, uint32_t pvp_queue_depth_arg = 0);

  /** Uses PVP kernel when h_pc->is_pvp and pinned buffer valid; else bam_ptr path. */
  void read_feature(uint64_t tensor_ptr, uint64_t index_ptr, int64_t num_index,
                    int dim, int cache_dim);

  void update_prefetch_timestamp(uint64_t d_pages_ptr, uint64_t d_ts_ptr,
                                 uint64_t d_idx_ptr, int n);

  void read_next_reuse_for_pages(uint64_t d_pages_ptr, uint64_t d_out_ptr,
                                 int64_t n);

  /**
   * Copy one PVP ring slice (``pvp_queue_depth`` pages for head ``time_step % num_pvp_buffers``)
   * from pinned host staging to ``device_ptr`` (CUDA device pointer as ``uint64_t``).
   */
  void PVP_prefetch(uint64_t device_ptr, uint32_t time_step);

  /** Host copy of device ``page_cache_d_t::ssd_read_ops`` (NVMe read command count). */
  uint64_t ssd_read_ops_count() const;

  /**
   * Clear and rebuild the internal node reuse map on GPU. ``d_node_ids_ptr`` is device ``int64[n]``
   * (node ids; skip id < 0). ``d_batch_idx_ptr`` is device ``int32[n]``: for each row, the **batch
   * index** (0 = first batch, 1 = second, …), i.e. which PVP buffer / lookahead batch the row
   * belongs to. Concatenate batches in order so all rows from batch ``k`` use ``batch_idx == k``.
   * ``map_capacity`` lower bound on unique keys. Map value: INT32_MAX if one distinct batch index
   * or only repeats of the same index; else second-smallest distinct batch index (smallest is anchor).
   */
  void build_node_queue_index_map(uint64_t d_node_ids_ptr, uint64_t d_batch_idx_ptr,
                                  int32_t n, uint64_t map_capacity, uint32_t time_step);

  /**
   * Update the existing node reuse map in place (no clear). For each node id in the batch (device
   * int64, skip id < 0): if absent, insert value INT32_MAX; if present and value != INT32_MAX, leave
   * unchanged; if present and value == INT32_MAX, set value to time_step (second touch). Caller
   * should ensure batch ids are unique and time_step increases across calls. Requires a prior
   * successful build_node_queue_index_map that allocated the map.
   */
  void index_map_add(uint64_t d_node_ids_ptr, int32_t n, uint32_t time_step);

  /**
   * In-place removals: for each node id in the batch (device int64, skip id < 0), if the key is
   * absent do nothing; if the stored value is INT32_MAX or equals time_step (as int32), erase the
   * key; otherwise do nothing.
   */
  void index_map_remove(uint64_t d_node_ids_ptr, int32_t n, uint32_t time_step);
};

#endif /* LSM_MODULE_LSM_NVME_H */
