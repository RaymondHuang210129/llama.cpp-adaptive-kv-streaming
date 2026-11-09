#include "ggml-backend-memory.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <cstring>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
struct host_committed_range {
    size_t offset;
    size_t size;
};

struct host_reservation_state {
    void * mapping = nullptr;
    void * base = nullptr;
    size_t size = 0, mapped_size = 0, committed = 0;
    bool failed = false;
    std::mutex mutex;
    std::vector<std::shared_ptr<host_committed_range>> ranges;

    ~host_reservation_state() {
        if (!mapping) return;
#ifdef _WIN32
        if (!VirtualFree(mapping, 0, MEM_RELEASE)) GGML_LOG_ERROR("host reservation release failed\n");
#else
        if (munmap(mapping, mapped_size)) GGML_LOG_ERROR("host reservation release failed\n");
#endif
    }
};

struct host_reservation_view {
    std::shared_ptr<host_reservation_state> owner;
    std::shared_ptr<host_committed_range> range;
    void * data;
};

static void host_view_free(ggml_backend_buffer_t buffer) {
    delete static_cast<host_reservation_view *>(buffer->context);
}

static void * host_view_base(ggml_backend_buffer_t buffer) {
    return static_cast<host_reservation_view *>(buffer->context)->data;
}

static void host_view_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    std::memset(host_view_base(buffer), value, buffer->size);
}

static ggml_backend_buffer_t host_view_make(const host_reservation_view & view, size_t size);

static ggml_backend_buffer_t host_view_slice(ggml_backend_buffer_t buffer, size_t offset, size_t size) {
    auto view = *static_cast<host_reservation_view *>(buffer->context);
    view.data = static_cast<char *>(view.data) + offset;
    return host_view_make(view, size);
}

static ggml_backend_buffer_t host_view_make(const host_reservation_view & view, size_t size) {
    try {
        auto context = std::make_unique<host_reservation_view>(view);
        auto * buffer = ggml_backend_cpu_buffer_from_ptr(view.data, size);
        buffer->context = context.release();
        buffer->iface.free_buffer = host_view_free;
        buffer->iface.get_base = host_view_base;
        buffer->iface.clear = host_view_clear;
        buffer->view_buffer = host_view_slice;
        return buffer;
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}
}

struct ggml_backend_host_reservation {
    std::shared_ptr<host_reservation_state> state;
};

size_t ggml_backend_host_reservation_granularity(void) {
#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwPageSize;
#else
    const long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? size_t(page) : 0;
#endif
}

ggml_backend_host_reservation_t ggml_backend_host_reservation_new(size_t size) {
    const size_t page = ggml_backend_host_reservation_granularity();
    if (!page || (page & (page - 1)) || !size || size > SIZE_MAX - (page - 1)) return nullptr;
    const size_t padded = (size + page - 1) & ~(page - 1);
    if (page > size_t(PTRDIFF_MAX)/2 || padded > size_t(PTRDIFF_MAX) - 2*page) return nullptr;
    try {
        auto reservation = std::make_unique<ggml_backend_host_reservation>();
        reservation->state = std::make_shared<host_reservation_state>();
        auto & state = *reservation->state;
        state.mapped_size = padded + 2*page;
#ifdef _WIN32
        state.mapping = VirtualAlloc(nullptr, state.mapped_size, MEM_RESERVE, PAGE_NOACCESS);
#else
        state.mapping = mmap(nullptr, state.mapped_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (state.mapping == MAP_FAILED) state.mapping = nullptr;
#endif
        if (!state.mapping) return nullptr;
        state.base = static_cast<char *>(state.mapping) + page;
        state.size = padded;
        return reservation.release();
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

void ggml_backend_host_reservation_free(ggml_backend_host_reservation_t reservation) { delete reservation; }

void * ggml_backend_host_reservation_base(ggml_backend_host_reservation_t reservation) {
    return reservation ? reservation->state->base : nullptr;
}

size_t ggml_backend_host_reservation_size(ggml_backend_host_reservation_t reservation) {
    return reservation ? reservation->state->size : 0;
}

size_t ggml_backend_host_reservation_committed(ggml_backend_host_reservation_t reservation) {
    if (!reservation) return 0;
    std::lock_guard<std::mutex> lock(reservation->state->mutex);
    return reservation->state->committed;
}

bool ggml_backend_host_reservation_commit(ggml_backend_host_reservation_t reservation, size_t offset, size_t size) {
    const size_t page = ggml_backend_host_reservation_granularity();
    if (!reservation || !page || !size || offset%page || size%page) return false;
    auto & state = *reservation->state;
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.failed || offset > state.size || size > state.size - offset) return false;
    auto position = std::lower_bound(state.ranges.begin(), state.ranges.end(), offset,
        [](const std::shared_ptr<host_committed_range> & range, size_t first) { return range->offset < first; });
    if (position != state.ranges.end() && (*position)->offset == offset && (*position)->size == size) return true;
    if (position != state.ranges.end() && offset + size > (*position)->offset) return false;
    if (position != state.ranges.begin()) {
        const auto & before = *(position - 1);
        if (before->size > offset - before->offset) return false;
    }
    try {
        const size_t index = size_t(position - state.ranges.begin());
        auto range = std::make_shared<host_committed_range>(host_committed_range{offset, size});
        state.ranges.reserve(state.ranges.size() + 1);
        void * data = static_cast<char *>(state.base) + offset;
#ifdef _WIN32
        if (VirtualAlloc(data, size, MEM_COMMIT, PAGE_READWRITE) != data) return false;
#else
        if (mprotect(data, size, PROT_READ | PROT_WRITE)) return false;
#endif
        std::memset(data, 0, size);
        state.ranges.insert(state.ranges.begin() + index, std::move(range));
        state.committed += size;
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

bool ggml_backend_host_reservation_decommit(ggml_backend_host_reservation_t reservation, size_t offset, size_t size) {
    if (!reservation || !size) return false;
    auto & state = *reservation->state;
    std::lock_guard<std::mutex> lock(state.mutex);
    auto position = std::find_if(state.ranges.begin(), state.ranges.end(), [&](const std::shared_ptr<host_committed_range> & range) {
        return range->offset == offset && range->size == size;
    });
    if (position == state.ranges.end() || position->use_count() != 1) return false;
    void * data = static_cast<char *>(state.base) + offset;
#ifdef _WIN32
    if (!VirtualFree(data, size, MEM_DECOMMIT)) return false;
#else
    if (mprotect(data, size, PROT_NONE)) return false;
    if (madvise(data, size, MADV_DONTNEED)) {
        if (mprotect(data, size, PROT_READ | PROT_WRITE)) {
            state.failed = true;
            GGML_LOG_ERROR("host reservation protection restore failed\n");
        }
        return false;
    }
#endif
    state.committed -= size;
    state.ranges.erase(position);
    return true;
}

ggml_backend_buffer_t ggml_backend_host_reservation_view(ggml_backend_host_reservation_t reservation, size_t offset, size_t size) {
    if (!reservation || !size) return nullptr;
    auto & state = *reservation->state;
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.failed || offset > state.size || size > state.size - offset ||
            offset%ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type())) return nullptr;
    auto position = std::upper_bound(state.ranges.begin(), state.ranges.end(), offset,
        [](size_t first, const std::shared_ptr<host_committed_range> & range) { return first < range->offset; });
    if (position == state.ranges.begin()) return nullptr;
    const auto & range = *(position - 1);
    if (offset - range->offset > range->size || size > range->size - (offset - range->offset)) return nullptr;
    return host_view_make({reservation->state, range, static_cast<char *>(state.base) + offset}, size);
}

struct ggml_backend_memory_planner {
    size_t capacity;
    size_t alignment;
    uint64_t generation;
    bool building;
    std::vector<ggml_backend_memory_region> committed;
    std::vector<ggml_backend_memory_region> staged;
};

struct ggml_backend_memory_arena_view {
    ggml_backend_memory_region region;
    ggml_backend_buffer_t buffer;
    size_t active_leases;
};

struct ggml_backend_memory_arena {
    std::atomic<uint32_t> references {1};
    std::mutex mutex;
    bool accepting_leases = true;
    ggml_backend_buffer_t parent = nullptr;
    ggml_backend_memory_planner_t planner = nullptr;
    std::vector<ggml_backend_memory_arena_view> views;
};

struct ggml_backend_memory_lease {
    std::atomic<uint32_t> references {1};
    ggml_backend_memory_arena_t arena = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_memory_region region = {};
    uint64_t generation = 0;
};

static bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static bool valid_region_flags(uint32_t flags) {
    return (flags & ~GGML_BACKEND_MEMORY_REGION_PERSISTENT) == 0;
}

static bool valid_plan_flags(uint32_t flags) {
    return (flags & ~GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT) == 0;
}

static bool effective_alignment(
        const ggml_backend_memory_planner * planner, size_t requested, size_t * alignment) {
    if (requested != 0 && !is_power_of_two(requested)) {
        return false;
    }
    *alignment = std::max(planner->alignment, requested);
    return true;
}

static bool align_up(size_t value, size_t alignment, size_t * result) {
    const size_t mask = alignment - 1;
    if (value > std::numeric_limits<size_t>::max() - mask) {
        return false;
    }
    *result = (value + mask) & ~mask;
    return true;
}

bool ggml_backend_memory_plan_workspace_groups(
        const ggml_backend_buffer_type_t * bufts,
        const size_t * measurements,
        size_t n_phases,
        size_t n_slots,
        ggml_backend_memory_workspace_group * groups,
        size_t * n_groups) {
    if (n_groups == nullptr) {
        return false;
    }
    if (n_slots == 0) {
        *n_groups = 0;
        return true;
    }
    if (bufts == nullptr || measurements == nullptr || n_phases == 0 ||
            n_phases > std::numeric_limits<size_t>::max() / n_slots) {
        return false;
    }

    std::vector<ggml_backend_memory_workspace_group> planned;
    try {
        planned.reserve(n_slots);
        for (size_t slot = 0; slot < n_slots; ++slot) {
            ggml_backend_buffer_type_t buft = bufts[slot];
            if (buft == nullptr) {
                return false;
            }

            auto found = std::find_if(
                planned.begin(), planned.end(), [buft](const ggml_backend_memory_workspace_group & group) {
                    return group.buft == buft;
                });
            if (found == planned.end()) {
                planned.push_back({buft, 0, ggml_backend_buft_get_alignment(buft), slot});
                found = planned.end() - 1;
            }

            size_t size = 0;
            for (size_t phase = 0; phase < n_phases; ++phase) {
                size = std::max(size, measurements[phase*n_slots + slot]);
            }
            if (size == 0) {
                continue;
            }

            if (!is_power_of_two(found->alignment) || !align_up(size, found->alignment, &size)) {
                return false;
            }
            found->size = std::max(found->size, size);
        }
        planned.erase(std::remove_if(
            planned.begin(), planned.end(), [](const ggml_backend_memory_workspace_group & group) {
                return group.size == 0;
            }), planned.end());
    } catch (const std::bad_alloc &) {
        return false;
    }

    const size_t capacity = *n_groups;
    *n_groups = planned.size();
    if (groups == nullptr) {
        return true;
    }
    if (capacity < planned.size()) {
        return false;
    }
    std::copy(planned.begin(), planned.end(), groups);
    return true;
}

static bool has_region_id(
        const std::vector<ggml_backend_memory_region> & regions, uint64_t id) {
    return std::any_of(regions.begin(), regions.end(), [id](const ggml_backend_memory_region & region) {
        return region.id == id;
    });
}

static bool same_region(
        const ggml_backend_memory_region & lhs, const ggml_backend_memory_region & rhs) {
    return lhs.id == rhs.id && lhs.offset == rhs.offset && lhs.size == rhs.size &&
           lhs.alignment == rhs.alignment && lhs.flags == rhs.flags;
}

static ggml_backend_memory_arena_view * find_arena_view(
        ggml_backend_memory_arena * arena, uint64_t id) {
    const auto found = std::find_if(
        arena->views.begin(), arena->views.end(),
        [id](const ggml_backend_memory_arena_view & view) {
            return view.region.id == id;
        });
    return found != arena->views.end() ? &*found : nullptr;
}

static const ggml_backend_memory_region * find_staged_region(
        const ggml_backend_memory_planner * planner, uint64_t id) {
    const auto found = std::find_if(
        planner->staged.begin(), planner->staged.end(),
        [id](const ggml_backend_memory_region & region) {
            return region.id == id;
        });
    return found != planner->staged.end() ? &*found : nullptr;
}

static void free_arena_views(std::vector<ggml_backend_memory_arena_view> & views) {
    for (const auto & view : views) {
        ggml_backend_buffer_free(view.buffer);
    }
    views.clear();
}

static size_t arena_lease_count_locked(const ggml_backend_memory_arena * arena) {
    size_t count = 0;
    for (const auto & view : arena->views) {
        count += view.active_leases;
    }
    return count;
}

static bool insert_region(
        ggml_backend_memory_planner * planner,
        uint64_t id,
        size_t offset,
        size_t size,
        size_t alignment,
        uint32_t flags,
        ggml_backend_memory_region * result) {
    if (!planner->building || size == 0 || !valid_region_flags(flags) ||
            has_region_id(planner->staged, id)) {
        return false;
    }

    size_t required_alignment = 0;
    if (!effective_alignment(planner, alignment, &required_alignment) ||
            offset % required_alignment != 0 ||
            offset > planner->capacity ||
            size > planner->capacity - offset) {
        return false;
    }

    const size_t end = offset + size;
    const auto position = std::lower_bound(
        planner->staged.begin(), planner->staged.end(), offset,
        [](const ggml_backend_memory_region & region, size_t value) {
            return region.offset < value;
        });

    if (position != planner->staged.end() && end > position->offset) {
        return false;
    }
    if (position != planner->staged.begin()) {
        const auto & previous = *(position - 1);
        if (previous.offset + previous.size > offset) {
            return false;
        }
    }

    const ggml_backend_memory_region region = {
        id,
        offset,
        size,
        required_alignment,
        flags,
    };
    try {
        planner->staged.insert(position, region);
    } catch (const std::bad_alloc &) {
        return false;
    }

    if (result != nullptr) {
        *result = region;
    }
    return true;
}

ggml_backend_memory_planner_t ggml_backend_memory_planner_new(
        size_t capacity, size_t alignment) {
    if (capacity == 0 || !is_power_of_two(alignment)) {
        return nullptr;
    }

    return new (std::nothrow) ggml_backend_memory_planner {
        capacity,
        alignment,
        0,
        false,
        {},
        {},
    };
}

void ggml_backend_memory_planner_free(ggml_backend_memory_planner_t planner) {
    delete planner;
}

bool ggml_backend_memory_planner_begin(
        ggml_backend_memory_planner_t planner, uint32_t flags) {
    if (planner == nullptr || planner->building || !valid_plan_flags(flags)) {
        return false;
    }

    std::vector<ggml_backend_memory_region> staged;
    if (flags & GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT) {
        try {
            staged.reserve(planner->committed.size());
            for (const auto & region : planner->committed) {
                if (region.flags & GGML_BACKEND_MEMORY_REGION_PERSISTENT) {
                    staged.push_back(region);
                }
            }
        } catch (const std::bad_alloc &) {
            return false;
        }
    }

    planner->staged.swap(staged);
    planner->building = true;
    return true;
}

bool ggml_backend_memory_planner_reserve(
        ggml_backend_memory_planner_t planner,
        uint64_t id,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region) {
    if (planner == nullptr || !planner->building || size == 0 ||
            !valid_region_flags(flags) || has_region_id(planner->staged, id)) {
        return false;
    }

    size_t required_alignment = 0;
    if (!effective_alignment(planner, alignment, &required_alignment)) {
        return false;
    }

    size_t cursor = 0;
    for (const auto & existing : planner->staged) {
        size_t candidate = 0;
        if (align_up(cursor, required_alignment, &candidate) &&
                candidate <= existing.offset &&
                size <= existing.offset - candidate) {
            return insert_region(planner, id, candidate, size, required_alignment, flags, region);
        }
        cursor = existing.offset + existing.size;
    }

    size_t candidate = 0;
    if (!align_up(cursor, required_alignment, &candidate) ||
            candidate > planner->capacity ||
            size > planner->capacity - candidate) {
        return false;
    }
    return insert_region(planner, id, candidate, size, required_alignment, flags, region);
}

bool ggml_backend_memory_planner_reserve_at(
        ggml_backend_memory_planner_t planner,
        uint64_t id,
        size_t offset,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region) {
    if (planner == nullptr) {
        return false;
    }
    return insert_region(planner, id, offset, size, alignment, flags, region);
}

bool ggml_backend_memory_planner_commit(ggml_backend_memory_planner_t planner) {
    if (planner == nullptr || !planner->building ||
            planner->generation == std::numeric_limits<uint64_t>::max()) {
        return false;
    }

    planner->committed.swap(planner->staged);
    planner->staged.clear();
    planner->building = false;
    planner->generation++;
    return true;
}

void ggml_backend_memory_planner_rollback(ggml_backend_memory_planner_t planner) {
    if (planner == nullptr) {
        return;
    }
    planner->staged.clear();
    planner->building = false;
}

bool ggml_backend_memory_planner_get_region(
        ggml_backend_memory_planner_t planner,
        uint64_t id,
        struct ggml_backend_memory_region * region) {
    if (planner == nullptr || region == nullptr) {
        return false;
    }

    const auto found = std::find_if(
        planner->committed.begin(), planner->committed.end(),
        [id](const ggml_backend_memory_region & candidate) {
            return candidate.id == id;
        });
    if (found == planner->committed.end()) {
        return false;
    }
    *region = *found;
    return true;
}

bool ggml_backend_memory_planner_get_region_at(
        ggml_backend_memory_planner_t planner,
        size_t index,
        struct ggml_backend_memory_region * region) {
    if (planner == nullptr || region == nullptr || index >= planner->committed.size()) {
        return false;
    }
    *region = planner->committed[index];
    return true;
}

size_t ggml_backend_memory_planner_capacity(ggml_backend_memory_planner_t planner) {
    return planner != nullptr ? planner->capacity : 0;
}

size_t ggml_backend_memory_planner_alignment(ggml_backend_memory_planner_t planner) {
    return planner != nullptr ? planner->alignment : 0;
}

size_t ggml_backend_memory_planner_region_count(ggml_backend_memory_planner_t planner) {
    return planner != nullptr ? planner->committed.size() : 0;
}

size_t ggml_backend_memory_planner_used(ggml_backend_memory_planner_t planner) {
    if (planner == nullptr) {
        return 0;
    }

    size_t used = 0;
    for (const auto & region : planner->committed) {
        used += region.size;
    }
    return used;
}

size_t ggml_backend_memory_planner_high_water(ggml_backend_memory_planner_t planner) {
    if (planner == nullptr || planner->committed.empty()) {
        return 0;
    }

    const auto & region = planner->committed.back();
    return region.offset + region.size;
}

uint64_t ggml_backend_memory_planner_generation(ggml_backend_memory_planner_t planner) {
    return planner != nullptr ? planner->generation : 0;
}

ggml_backend_memory_arena_t ggml_backend_memory_arena_new(
        ggml_backend_buffer_type_t buft, size_t capacity) {
    if (buft == nullptr || capacity == 0) {
        return nullptr;
    }
    ggml_backend_buffer_t parent = ggml_backend_buft_alloc_buffer(buft, capacity);
    if (parent == nullptr) {
        return nullptr;
    }
    ggml_backend_memory_arena_t arena = ggml_backend_memory_arena_new_from_buffer(parent);
    ggml_backend_buffer_free(parent);
    return arena;
}

ggml_backend_memory_arena_t ggml_backend_memory_arena_new_from_buffer(
        ggml_backend_buffer_t buffer) {
    if (buffer == nullptr || buffer->view_buffer == nullptr ||
            ggml_backend_buffer_get_size(buffer) == 0) {
        return nullptr;
    }

    ggml_backend_memory_planner_t planner = ggml_backend_memory_planner_new(
        ggml_backend_buffer_get_size(buffer), ggml_backend_buffer_get_alignment(buffer));
    if (planner == nullptr) {
        return nullptr;
    }

    auto * arena = new (std::nothrow) ggml_backend_memory_arena;
    if (arena == nullptr) {
        ggml_backend_memory_planner_free(planner);
        return nullptr;
    }
    arena->parent = ggml_backend_buffer_retain(buffer);
    arena->planner = planner;
    return arena;
}

ggml_backend_memory_arena_t ggml_backend_memory_arena_retain(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return nullptr;
    }
    uint32_t count = arena->references.load(std::memory_order_relaxed);
    do {
        GGML_ASSERT(count > 0 && count < std::numeric_limits<uint32_t>::max());
    } while (!arena->references.compare_exchange_weak(
        count, count + 1, std::memory_order_relaxed, std::memory_order_relaxed));
    return arena;
}

void ggml_backend_memory_arena_free(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return;
    }
    const uint32_t count = arena->references.fetch_sub(1, std::memory_order_acq_rel);
    GGML_ASSERT(count > 0);
    if (count != 1) {
        return;
    }
    free_arena_views(arena->views);
    ggml_backend_memory_planner_free(arena->planner);
    ggml_backend_buffer_free(arena->parent);
    delete arena;
}

bool ggml_backend_memory_arena_begin(
        ggml_backend_memory_arena_t arena, uint32_t flags) {
    if (arena == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    return ggml_backend_memory_planner_begin(arena->planner, flags);
}

bool ggml_backend_memory_arena_reserve(
        ggml_backend_memory_arena_t arena,
        uint64_t id,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region) {
    if (arena == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    return ggml_backend_memory_planner_reserve(arena->planner, id, size, alignment, flags, region);
}

bool ggml_backend_memory_arena_reserve_at(
        ggml_backend_memory_arena_t arena,
        uint64_t id,
        size_t offset,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region) {
    if (arena == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    return ggml_backend_memory_planner_reserve_at(
        arena->planner, id, offset, size, alignment, flags, region);
}

bool ggml_backend_memory_arena_commit(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    if (!arena->planner->building) {
        return false;
    }

    for (const auto & view : arena->views) {
        if (view.active_leases == 0) {
            continue;
        }
        const ggml_backend_memory_region * staged = find_staged_region(arena->planner, view.region.id);
        if (staged == nullptr || !(staged->flags & GGML_BACKEND_MEMORY_REGION_PERSISTENT) ||
                !same_region(view.region, *staged)) {
            ggml_backend_memory_planner_rollback(arena->planner);
            return false;
        }
    }

    std::vector<ggml_backend_memory_arena_view> next_views;
    try {
        next_views.reserve(arena->planner->staged.size());
    } catch (const std::bad_alloc &) {
        ggml_backend_memory_planner_rollback(arena->planner);
        return false;
    }

    for (const auto & region : arena->planner->staged) {
        ggml_backend_buffer_t view = nullptr;
        size_t active_leases = 0;
        if (region.flags & GGML_BACKEND_MEMORY_REGION_PERSISTENT) {
            ggml_backend_memory_arena_view * existing = find_arena_view(arena, region.id);
            if (existing != nullptr && same_region(existing->region, region)) {
                view = ggml_backend_buffer_retain(existing->buffer);
                active_leases = existing->active_leases;
            }
        }
        if (view == nullptr) {
            try {
                view = ggml_backend_buffer_view(arena->parent, region.offset, region.size);
            } catch (const std::bad_alloc &) {
                free_arena_views(next_views);
                ggml_backend_memory_planner_rollback(arena->planner);
                return false;
            } catch (...) {
                free_arena_views(next_views);
                ggml_backend_memory_planner_rollback(arena->planner);
                throw;
            }
        }
        if (view == nullptr) {
            free_arena_views(next_views);
            ggml_backend_memory_planner_rollback(arena->planner);
            return false;
        }

        try {
            next_views.push_back({region, view, active_leases});
        } catch (const std::bad_alloc &) {
            ggml_backend_buffer_free(view);
            free_arena_views(next_views);
            ggml_backend_memory_planner_rollback(arena->planner);
            return false;
        }
    }

    if (!ggml_backend_memory_planner_commit(arena->planner)) {
        free_arena_views(next_views);
        ggml_backend_memory_planner_rollback(arena->planner);
        return false;
    }

    arena->views.swap(next_views);
    free_arena_views(next_views);
    return true;
}

void ggml_backend_memory_arena_rollback(ggml_backend_memory_arena_t arena) {
    if (arena != nullptr) {
        std::lock_guard<std::mutex> lock(arena->mutex);
        ggml_backend_memory_planner_rollback(arena->planner);
    }
}

ggml_backend_buffer_t ggml_backend_memory_arena_parent(ggml_backend_memory_arena_t arena) {
    return arena != nullptr ? arena->parent : nullptr;
}

ggml_backend_buffer_t ggml_backend_memory_arena_get_buffer(
        ggml_backend_memory_arena_t arena, uint64_t id) {
    if (arena == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    ggml_backend_memory_arena_view * view = find_arena_view(arena, id);
    return view != nullptr ? view->buffer : nullptr;
}

bool ggml_backend_memory_arena_get_region(
        ggml_backend_memory_arena_t arena,
        uint64_t id,
        struct ggml_backend_memory_region * region) {
    if (arena == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    return ggml_backend_memory_planner_get_region(arena->planner, id, region);
}

size_t ggml_backend_memory_arena_capacity(ggml_backend_memory_arena_t arena) {
    return arena != nullptr ? ggml_backend_memory_planner_capacity(arena->planner) : 0;
}

size_t ggml_backend_memory_arena_region_count(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    return ggml_backend_memory_planner_region_count(arena->planner);
}

size_t ggml_backend_memory_arena_used(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    return ggml_backend_memory_planner_used(arena->planner);
}

size_t ggml_backend_memory_arena_high_water(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    return ggml_backend_memory_planner_high_water(arena->planner);
}

uint64_t ggml_backend_memory_arena_generation(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    return ggml_backend_memory_planner_generation(arena->planner);
}

bool ggml_backend_memory_arena_quiesce(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    arena->accepting_leases = false;
    return true;
}

bool ggml_backend_memory_arena_resume(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    if (arena->planner->building) {
        return false;
    }
    arena->accepting_leases = true;
    return true;
}

enum ggml_backend_memory_arena_state ggml_backend_memory_arena_get_state(
        ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return GGML_BACKEND_MEMORY_ARENA_STATE_INVALID;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    if (arena->accepting_leases) {
        return GGML_BACKEND_MEMORY_ARENA_STATE_OPEN;
    }
    return arena_lease_count_locked(arena) != 0 ?
        GGML_BACKEND_MEMORY_ARENA_STATE_DRAINING :
        GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT;
}

size_t ggml_backend_memory_arena_lease_count(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    return arena_lease_count_locked(arena);
}

ggml_backend_memory_lease_t ggml_backend_memory_arena_acquire(
        ggml_backend_memory_arena_t arena, uint64_t id) {
    if (arena == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    if (!arena->accepting_leases) {
        return nullptr;
    }
    ggml_backend_memory_arena_view * view = find_arena_view(arena, id);
    if (view == nullptr || view->active_leases == std::numeric_limits<size_t>::max()) {
        return nullptr;
    }
    auto * lease = new (std::nothrow) ggml_backend_memory_lease;
    if (lease == nullptr) {
        return nullptr;
    }
    lease->arena = ggml_backend_memory_arena_retain(arena);
    lease->buffer = ggml_backend_buffer_retain(view->buffer);
    lease->region = view->region;
    lease->generation = ggml_backend_memory_planner_generation(arena->planner);
    view->active_leases++;
    return lease;
}

ggml_backend_memory_lease_t ggml_backend_memory_lease_retain(
        ggml_backend_memory_lease_t lease) {
    if (lease == nullptr) {
        return nullptr;
    }
    uint32_t count = lease->references.load(std::memory_order_relaxed);
    do {
        GGML_ASSERT(count > 0 && count < std::numeric_limits<uint32_t>::max());
    } while (!lease->references.compare_exchange_weak(
        count, count + 1, std::memory_order_relaxed, std::memory_order_relaxed));
    return lease;
}

void ggml_backend_memory_lease_free(ggml_backend_memory_lease_t lease) {
    if (lease == nullptr) {
        return;
    }
    const uint32_t count = lease->references.fetch_sub(1, std::memory_order_acq_rel);
    GGML_ASSERT(count > 0);
    if (count != 1) {
        return;
    }

    ggml_backend_memory_arena_t arena = lease->arena;
    ggml_backend_buffer_t buffer = lease->buffer;
    {
        std::lock_guard<std::mutex> lock(arena->mutex);
        ggml_backend_memory_arena_view * view = find_arena_view(arena, lease->region.id);
        GGML_ASSERT(view != nullptr && view->buffer == buffer && view->active_leases > 0);
        view->active_leases--;
    }
    delete lease;
    ggml_backend_buffer_free(buffer);
    ggml_backend_memory_arena_free(arena);
}

ggml_backend_buffer_t ggml_backend_memory_lease_buffer(ggml_backend_memory_lease_t lease) {
    return lease != nullptr ? lease->buffer : nullptr;
}

ggml_backend_buffer_t ggml_backend_memory_lease_parent(ggml_backend_memory_lease_t lease) {
    return lease != nullptr ? lease->arena->parent : nullptr;
}

bool ggml_backend_memory_lease_get_region(
        ggml_backend_memory_lease_t lease,
        struct ggml_backend_memory_region * region) {
    if (lease == nullptr || region == nullptr) {
        return false;
    }
    *region = lease->region;
    return true;
}

uint64_t ggml_backend_memory_lease_generation(ggml_backend_memory_lease_t lease) {
    return lease != nullptr ? lease->generation : 0;
}

// Copy address-ordered committed metadata under the arena lock; failure leaves the output unchanged.
bool ggml_backend_memory_arena_get_region_at(
        ggml_backend_memory_arena_t arena, size_t index, ggml_backend_memory_region * region) {
    if (arena == nullptr || region == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(arena->mutex);
    return ggml_backend_memory_planner_get_region_at(arena->planner, index, region);
}
