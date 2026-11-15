//! C header for the MORPHOS rust_trace shared library.
//!
//! Link with `rust_trace.dll` / `librust_trace.so` / `librust_trace.dylib`.

#ifndef MORPHOS_RUST_TRACE_H
#define MORPHOS_RUST_TRACE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TRACE_ABI_VERSION 5
#define TRACE_EV_RECORD_SIZE 19
#define TRACE_RING_HEAD 56
#define TRACE_RING_TAIL 64
#define TRACE_RING_HDR 128
#define TRACE_RING_SLOT 32

typedef struct {
    uint64_t tick;
    float x;
    float y;
    float z;
    float confidence;
    uint8_t flags;
} TraceEvent;

typedef struct {
    uint64_t events;
    uint64_t bytes;
    uint64_t nan_or_inf;
    uint64_t gaps;
    uint64_t first_tick;
    uint64_t last_tick;
    float min_x, max_x, min_y, max_y, min_z, max_z;
    double elapsed_s;
} TraceScanStats;

typedef struct {
    uint64_t pushed;
    uint64_t popped;
    uint64_t dropped;
    uint64_t capacity;
    uint64_t occupancy;
} TraceRingStats;

int trace_abi_version(void);
int trace_scan(const char *path, TraceScanStats *out);
ptrdiff_t trace_read(const char *path, TraceEvent *out, size_t max);
int trace_ring_new(uint64_t capacity);
uint8_t *trace_ring_ptr(void);
size_t trace_ring_bytes(void);
int trace_ring_push(uint64_t tick, float x, float y, float z, float conf, uint8_t flags);
int trace_ring_pop(TraceEvent *out);
int trace_ring_stats(TraceRingStats *out);
ptrdiff_t trace_ring_load_file(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* MORPHOS_RUST_TRACE_H */
