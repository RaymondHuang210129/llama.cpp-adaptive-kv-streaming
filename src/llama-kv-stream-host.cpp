#include "llama-kv-stream-host.h"
#include "../ggml/src/ggml-backend-memory.h"
#include "../ggml/src/ggml-backend-execution.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

// Validate byte geometry before touching an allocator.
static bool host_plan(const llama_kv_stream_host_config & c, ggml_kv_stream_layout & planes,
        size_t & stride, size_t & bytes) {
    if (!c.cache_id || !c.layers || !c.context_tokens || c.shape.page_tokens <= 0 ||
            uint64_t(c.shape.page_tokens) > SIZE_MAX) return false;
    ggml_kv_stream_execution page;
    if (ggml_kv_stream_resolve(c.shape, c.capabilities, size_t(c.shape.page_tokens), page).status != ggml_kv_stream_status::success) return false;
    const size_t page_tokens = size_t(c.shape.page_tokens);
    const size_t tail = c.context_tokens % page_tokens;
    const size_t pad = tail ? page_tokens - tail : 0;
    if (c.context_tokens > SIZE_MAX - pad) return false;
    if (ggml_kv_stream_layout_make(c.shape, c.context_tokens + pad, planes).status != ggml_kv_stream_status::success) return false;
    const size_t remainder = planes.bytes % c.shape.alignment;
    const size_t alignment_pad = remainder ? c.shape.alignment - remainder : 0;
    if (planes.bytes > SIZE_MAX - alignment_pad) return false;
    stride = planes.bytes + alignment_pad;
    if (!stride || c.layers > SIZE_MAX / stride) return false;
    bytes = stride*c.layers;
    return bytes <= SIZE_MAX - (c.shape.alignment - 1);
}

// Resolve the registry-local ordinal without assuming it equals a physical GPU index.
static int host_device_index(ggml_backend_dev_t device) {
    if (!device) return -1;
    auto * reg = ggml_backend_dev_backend_reg(device);
    if (!reg) return -1;
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg) && i <= size_t(std::numeric_limits<int>::max()); ++i) {
        if (ggml_backend_reg_dev_get(reg, i) == device) return int(i);
    }
    return -1;
}

// Missing strict pinning support must not become a pageable allocation.
ggml_backend_buffer_type_t llama_kv_stream_host_buffer_type(ggml_backend_dev_t device) {
    const int index = host_device_index(device);
    if (index < 0) return nullptr;
    auto * reg = ggml_backend_dev_backend_reg(device);
    using factory_t = ggml_backend_buffer_type_t (*)(int);
    auto fn = reinterpret_cast<factory_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_host_buffer_type"));
    if (!fn) return nullptr;
    auto * type = fn(index);
    return type && ggml_backend_buft_is_host(type) && ggml_backend_buft_get_device(type) == device ? type : nullptr;
}

// The returned wrapper owns only its new registration and a retained reference to the caller's buffer.
ggml_backend_buffer_t llama_kv_stream_host_register(ggml_backend_dev_t device, ggml_backend_buffer_t owner) {
    auto * type = llama_kv_stream_host_buffer_type(device);
    if (!type || !owner) return nullptr;
    using register_t = ggml_backend_buffer_t (*)(int, ggml_backend_buffer_t);
    auto fn = reinterpret_cast<register_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(device), "ggml_backend_cuda_kv_host_buffer_register"));
    if (!fn) return nullptr;
    auto * buffer = fn(host_device_index(device), owner);
    if (buffer && ggml_backend_buffer_get_type(buffer) != type) {
        ggml_backend_buffer_free(buffer);
        return nullptr;
    }
    return buffer;
}

// Final host ownership is independent of the device arena and its generations.
llama_kv_stream_host::~llama_kv_stream_host() { ggml_backend_buffer_free(storage); }

// Allocate only after checked sizing, and clear only the canonical storage span.
std::shared_ptr<llama_kv_stream_host> llama_kv_stream_host::create(
        const llama_kv_stream_host_config & c, ggml_backend_buffer_type_t type) {
    ggml_kv_stream_layout planes;
    size_t stride, bytes;
    if (!type || !ggml_backend_buft_is_host(type) || !host_plan(c, planes, stride, bytes)) return {};
    try {
        using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
        buffer_ptr buffer(ggml_backend_buft_alloc_buffer(type, bytes + c.shape.alignment - 1), ggml_backend_buffer_free);
        auto host = from_buffer(c, type, buffer.get());
        if (host) std::memset(host->base, 0, host->storage_bytes);
        return host;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

// Imported bytes are preserved; alignment slack is not KV payload or a device reserve.
std::shared_ptr<llama_kv_stream_host> llama_kv_stream_host::from_buffer(
        const llama_kv_stream_host_config & c, ggml_backend_buffer_type_t type, ggml_backend_buffer_t buffer) {
    if (!type || !buffer || ggml_backend_buffer_get_type(buffer) != type || !ggml_backend_buft_is_host(type)) return {};
    try {
        std::shared_ptr<llama_kv_stream_host> host(new llama_kv_stream_host);
        if (!host_plan(c, host->planes, host->layer_stride, host->storage_bytes)) return {};
        void * pointer = ggml_backend_buffer_get_base(buffer);
        const auto address = reinterpret_cast<uintptr_t>(pointer);
        if (!address) return {};
        const size_t remainder = address % c.shape.alignment;
        const size_t offset = remainder ? c.shape.alignment - remainder : 0;
        const size_t capacity = ggml_backend_buffer_get_size(buffer);
        if (offset > capacity || host->storage_bytes > capacity - offset ||
                offset > UINTPTR_MAX - address || host->storage_bytes > UINTPTR_MAX - address - offset) return {};
        host->cfg = c;
        host->base = static_cast<char *>(pointer) + offset;
        host->storage = ggml_backend_buffer_retain(buffer);
        return host;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

std::shared_ptr<llama_kv_stream_host> llama_kv_stream_host::from_reservation(
        std::shared_ptr<llama_kv_stream_host_reservation> reservation) {
    if (!reservation) return {};
    try {
        std::shared_ptr<llama_kv_stream_host> host(new llama_kv_stream_host);
        const auto & config = reservation->config();
        if (!host_plan(config, host->planes, host->layer_stride, host->storage_bytes)) return {};
        llama_kv_stream_host_layer first;
        if (!reservation->layer(0, first)) return {};
        host->cfg = config;
        host->base = first.k;
        host->demand = std::move(reservation);
        return host;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

bool llama_kv_stream_host::demand_backed() const noexcept { return bool(demand); }

bool llama_kv_stream_host::valid() const noexcept {
    if (!base) return false;
    llama_kv_stream_host_layer first;
    return !demand || demand->layer(0, first);
}

bool llama_kv_stream_host::prepare(size_t rows) {
    return demand ? demand->prepare(rows) : rows <= planes.tokens;
}

bool llama_kv_stream_host::ranges(uint32_t layer, ggml_kv_stream_operand operand, size_t offset, size_t bytes,
        std::vector<llama_kv_stream_host_range> & output) const {
    if (demand) return demand->ranges(layer, operand, offset, bytes, output);
    if (layer >= cfg.layers || !bytes || (operand != ggml_kv_stream_operand::k && operand != ggml_kv_stream_operand::v)) return false;
    const size_t capacity = operand == ggml_kv_stream_operand::k ? planes.k_bytes : planes.v_bytes;
    if (offset > capacity || bytes > capacity - offset) return false;
    try {
        const size_t leading = size_t(static_cast<const char *>(base) -
            static_cast<const char *>(ggml_backend_buffer_get_base(storage)));
        const size_t first = leading + layer*layer_stride +
            (operand == ggml_kv_stream_operand::v ? planes.v_offset : 0) + offset;
        std::vector<llama_kv_stream_host_range> ranges;
        ranges.push_back({ggml_backend_buffer_ptr(ggml_backend_buffer_retain(storage)), first, bytes});
        output = std::move(ranges);
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

bool llama_kv_stream_host::locate(const void * data, size_t offset, size_t bytes, uint32_t & layer,
        ggml_kv_stream_operand & operand, size_t & in_plane) const noexcept {
    if (!valid() || !data || !bytes) return false;
    const auto address = reinterpret_cast<uintptr_t>(data);
    const auto first = reinterpret_cast<uintptr_t>(base);
    if (address < first || address - first > storage_bytes || offset > storage_bytes - (address - first) ||
            bytes > storage_bytes - (address - first) - offset) return false;
    const size_t relative = address - first + offset;
    const size_t ordinal = relative/layer_stride;
    if (ordinal >= cfg.layers) return false;
    const size_t within = relative%layer_stride;
    size_t capacity;
    if (within < planes.k_bytes) {
        operand = ggml_kv_stream_operand::k;
        in_plane = within; capacity = planes.k_bytes;
    } else if (within >= planes.v_offset && within - planes.v_offset < planes.v_bytes) {
        operand = ggml_kv_stream_operand::v;
        in_plane = within - planes.v_offset; capacity = planes.v_bytes;
    } else {
        return false;
    }
    if (bytes > capacity - in_plane) return false;
    layer = uint32_t(ordinal);
    return true;
}

bool llama_kv_stream_host::address_ranges(const void * data, size_t bytes,
        std::vector<llama_kv_stream_host_range> & output) const {
    uint32_t layer;
    ggml_kv_stream_operand operand;
    size_t in_plane;
    return locate(data,0,bytes,layer,operand,in_plane) && ranges(layer,operand,in_plane,bytes,output);
}

bool llama_kv_stream_host::upload(ggml_backend_t backend, ggml_tensor * target, size_t offset,
        const void * data, size_t bytes, size_t & calls) const {
    if (!backend || !target || !target->data || !target->buffer || !bytes) return false;
    const size_t capacity = ggml_nbytes(target);
    if (offset > capacity || bytes > capacity-offset) return false;
    uint32_t layer;
    ggml_kv_stream_operand operand;
    size_t in_plane;
    if (!locate(data,0,bytes,layer,operand,in_plane)) return false;
    if (!demand) {
        if (calls == SIZE_MAX) return false;
        ++calls;
        ggml_backend_tensor_set_async(backend,target,data,offset,bytes);
        return true;
    }
    std::vector<llama_kv_stream_host_range> pieces;
    if (!ranges(layer,operand,in_plane,bytes,pieces) || pieces.size() > SIZE_MAX-calls) return false;
    for (const auto & piece : pieces) {
        const auto * source = static_cast<const char *>(ggml_backend_buffer_get_base(piece.buffer.get()))+piece.offset;
        ++calls;
        ggml_backend_tensor_set_async(backend,target,source,offset,piece.bytes);
        offset += piece.bytes;
    }
    return true;
}

bool llama_kv_stream_host::access(const ggml_tensor * tensor, size_t offset, size_t bytes, bool write,
        std::vector<llama_kv_stream_host_range> & output) {
    uint32_t layer;
    ggml_kv_stream_operand operand;
    size_t in_plane;
    if (!tensor || !locate(tensor->data,offset,bytes,layer,operand,in_plane)) return false;
    const size_t stride = operand == ggml_kv_stream_operand::k ? planes.k_token_bytes : planes.v_token_bytes;
    const size_t end = in_plane + bytes;
    if (write && !prepare(end/stride + (end%stride != 0))) return false;
    return ranges(layer, operand, in_plane, bytes, output);
}

ggml_backend_buffer_t llama_kv_stream_host::execution_buffer(ggml_backend_dev_t device,
        const ggml_backend_execution_ops & ops, void * context) {
    if (!demand) return ggml_backend_execution_buffer_new(device, storage, ops, context);
    ggml_backend_execution_io io{
        this,
        [](void * p, const ggml_tensor * tensor, void * data, size_t offset, size_t bytes) {
            auto & host = *static_cast<llama_kv_stream_host *>(p);
            std::vector<llama_kv_stream_host_range> ranges;
            if (!host.access(tensor, offset, bytes, false, ranges)) return false;
            size_t copied = 0;
            for (const auto & range : ranges) {
                std::memcpy(static_cast<char *>(data) + copied,
                    static_cast<const char *>(ggml_backend_buffer_get_base(range.buffer.get())) + range.offset, range.bytes);
                copied += range.bytes;
            }
            return true;
        },
        [](void * p, const ggml_tensor * tensor, const void * data, size_t offset, size_t bytes) {
            auto & host = *static_cast<llama_kv_stream_host *>(p);
            std::vector<llama_kv_stream_host_range> ranges;
            if (!host.access(tensor, offset, bytes, true, ranges)) return false;
            size_t copied = 0;
            for (const auto & range : ranges) {
                std::memcpy(static_cast<char *>(ggml_backend_buffer_get_base(range.buffer.get())) + range.offset,
                    static_cast<const char *>(data) + copied, range.bytes);
                copied += range.bytes;
            }
            return true;
        },
        [](void * p, const ggml_tensor * tensor, uint8_t value, size_t offset, size_t bytes) {
            auto & host = *static_cast<llama_kv_stream_host *>(p);
            std::vector<llama_kv_stream_host_range> ranges;
            if (!host.access(tensor, offset, bytes, true, ranges)) return false;
            for (const auto & range : ranges) std::memset(
                static_cast<char *>(ggml_backend_buffer_get_base(range.buffer.get())) + range.offset, value, range.bytes);
            return true;
        },
        [](void * p, uint8_t value) {
            auto & host = *static_cast<llama_kv_stream_host *>(p);
            return host.clear(value);
        },
    };
    return ggml_backend_execution_buffer_new_address_space(device, base, storage_bytes,
        std::max(size_t(128),cfg.shape.alignment), ops, io, context);
}

// Stable host identity does not change when a device mirror is rebound.
uint64_t llama_kv_stream_host::cache_id() const noexcept { return cfg.cache_id; }
// The validated shape and original context capacity belong to the authoritative storage.
const llama_kv_stream_host_config & llama_kv_stream_host::config() const noexcept { return cfg; }
// Each layer has the same separately packed K and V planes.
const ggml_kv_stream_layout & llama_kv_stream_host::layout() const noexcept { return planes; }
// Include alignment between layers, not just live K/V payload.
size_t llama_kv_stream_host::stride() const noexcept { return layer_stride; }
// Exclude leading/trailing allocation slack from the canonical storage size.
size_t llama_kv_stream_host::bytes() const noexcept { return storage_bytes; }
size_t llama_kv_stream_host::committed_bytes() const noexcept { return demand ? demand->committed_bytes() : storage_bytes; }
size_t llama_kv_stream_host::registered_bytes() const noexcept { return demand ? demand->registered_bytes() : 0; }
ggml_backend_buffer_type_t llama_kv_stream_host::buffer_type() const noexcept {
    return demand ? demand->buffer_type() : ggml_backend_buffer_get_type(storage);
}
bool llama_kv_stream_host::clear(uint8_t value) {
    if (!valid()) return false;
    if (demand) demand->clear(value);
    else ggml_backend_buffer_clear(storage,value);
    return true;
}
// Borrowed buffer access requires retaining this storage owner.
ggml_backend_buffer_t llama_kv_stream_host::buffer() const noexcept { return storage; }
// All products and offsets were bounded when the owner was created.
bool llama_kv_stream_host::layer(uint32_t index, llama_kv_stream_host_layer & output) const noexcept {
    if (demand) return demand->layer(index, output);
    if (index >= cfg.layers) return false;
    auto * k = static_cast<char *>(base) + size_t(index)*layer_stride;
    output = {k, k + planes.v_offset};
    return true;
}

struct llama_kv_stream_host_reservation::impl {
    struct chunk {
        size_t offset, bytes;
        ggml_backend_buffer_ptr buffer;
    };
    llama_kv_stream_host_config config;
    ggml_kv_stream_layout planes;
    size_t stride = 0, bytes = 0, base_offset = 0, chunk_bytes = 0, rows = 0, registered = 0;
    bool failed = false;
    ggml_backend_buffer_type_t type = nullptr;
    register_range registrar;
    std::unique_ptr<ggml_backend_host_reservation, decltype(&ggml_backend_host_reservation_free)>
        reservation{nullptr, ggml_backend_host_reservation_free};
    std::vector<chunk> chunks;

    auto find(size_t offset) const {
        return std::lower_bound(chunks.begin(), chunks.end(), offset,
            [](const chunk & c, size_t first) { return c.offset < first; });
    }

    void zero(size_t first, size_t bytes) {
        for (const auto & chunk : chunks) {
            const size_t begin = std::max(first, chunk.offset);
            const size_t end = std::min(first + bytes, chunk.offset + chunk.bytes);
            if (begin < end) std::memset(static_cast<char *>(ggml_backend_buffer_get_base(chunk.buffer.get())) +
                (begin - chunk.offset), 0, end - begin);
        }
    }
};

llama_kv_stream_host_reservation::llama_kv_stream_host_reservation() = default;
llama_kv_stream_host_reservation::~llama_kv_stream_host_reservation() = default;

std::unique_ptr<llama_kv_stream_host_reservation> llama_kv_stream_host_reservation::create(
        const llama_kv_stream_host_config & config, ggml_backend_dev_t device, size_t chunk_bytes) {
    return create(config, llama_kv_stream_host_buffer_type(device),
        [device](ggml_backend_buffer_t owner) { return llama_kv_stream_host_register(device, owner); }, chunk_bytes);
}

std::unique_ptr<llama_kv_stream_host_reservation> llama_kv_stream_host_reservation::create(
        const llama_kv_stream_host_config & config, ggml_backend_buffer_type_t expected_type,
        register_range registrar, size_t chunk_bytes) {
    const size_t page = ggml_backend_host_reservation_granularity();
    if (!expected_type || !ggml_backend_buft_is_host(expected_type) || !registrar || !page ||
            !chunk_bytes || chunk_bytes%page) return {};
    try {
        std::unique_ptr<llama_kv_stream_host_reservation> host(new llama_kv_stream_host_reservation);
        host->state = std::make_unique<impl>();
        auto & s = *host->state;
        if (!host_plan(config, s.planes, s.stride, s.bytes)) return {};
        s.reservation.reset(ggml_backend_host_reservation_new(s.bytes + config.shape.alignment - 1));
        if (!s.reservation) return {};
        const auto address = reinterpret_cast<uintptr_t>(ggml_backend_host_reservation_base(s.reservation.get()));
        const size_t remainder = address%config.shape.alignment;
        s.base_offset = remainder ? config.shape.alignment - remainder : 0;
        s.config = config;
        s.type = expected_type;
        s.registrar = std::move(registrar);
        s.chunk_bytes = chunk_bytes;
        return host;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

const ggml_kv_stream_layout & llama_kv_stream_host_reservation::layout() const noexcept { return state->planes; }
const llama_kv_stream_host_config & llama_kv_stream_host_reservation::config() const noexcept { return state->config; }
ggml_backend_buffer_type_t llama_kv_stream_host_reservation::buffer_type() const noexcept { return state->type; }
size_t llama_kv_stream_host_reservation::bytes() const noexcept { return state->bytes; }
size_t llama_kv_stream_host_reservation::reserved_bytes() const noexcept {
    return ggml_backend_host_reservation_size(state->reservation.get());
}
size_t llama_kv_stream_host_reservation::committed_bytes() const noexcept {
    return ggml_backend_host_reservation_committed(state->reservation.get());
}
size_t llama_kv_stream_host_reservation::registered_bytes() const noexcept { return state->registered; }

bool llama_kv_stream_host_reservation::layer(uint32_t index, llama_kv_stream_host_layer & output) const noexcept {
    const auto & s = *state;
    if (s.failed || index >= s.config.layers) return false;
    auto * k = static_cast<char *>(ggml_backend_host_reservation_base(s.reservation.get())) +
        s.base_offset + index*s.stride;
    output = {k, k + s.planes.v_offset};
    return true;
}

bool llama_kv_stream_host_reservation::prepare(size_t rows) {
    auto & s = *state;
    if (s.failed || rows > s.planes.tokens) return false;
    if (rows <= s.rows) return true;
    std::vector<size_t> missing, committed;
    std::vector<impl::chunk> pending;
    bool success = false;
    try {
        for (size_t layer = 0; layer < s.config.layers; ++layer) for (bool value : {false, true}) {
            const size_t token_bytes = value ? s.planes.v_token_bytes : s.planes.k_token_bytes;
            const size_t plane = s.base_offset + layer*s.stride + (value ? s.planes.v_offset : 0);
            const size_t first = plane + s.rows*token_bytes;
            const size_t end = plane + rows*token_bytes;
            for (size_t offset = first - first%s.chunk_bytes; offset < end;) {
                const auto found = s.find(offset);
                if (found == s.chunks.end() || found->offset != offset) missing.push_back(offset);
                const size_t remaining = reserved_bytes() - offset;
                if (s.chunk_bytes >= remaining) break;
                offset += s.chunk_bytes;
            }
        }
        std::sort(missing.begin(), missing.end());
        missing.erase(std::unique(missing.begin(), missing.end()), missing.end());
        pending.reserve(missing.size());
        committed.reserve(missing.size());
        s.chunks.reserve(s.chunks.size() + missing.size());
        for (const size_t offset : missing) {
            const size_t bytes = std::min(s.chunk_bytes, reserved_bytes() - offset);
            if (!ggml_backend_host_reservation_commit(s.reservation.get(), offset, bytes)) break;
            committed.push_back(offset);
            ggml_backend_buffer_ptr owner(ggml_backend_host_reservation_view(s.reservation.get(), offset, bytes));
            if (!owner) break;
            ggml_backend_buffer_ptr pinned(s.registrar(owner.get()));
            if (!pinned || ggml_backend_buffer_get_type(pinned.get()) != s.type ||
                    ggml_backend_buffer_get_size(pinned.get()) != bytes ||
                    ggml_backend_buffer_get_base(pinned.get()) != ggml_backend_buffer_get_base(owner.get())) break;
            pending.push_back({offset, bytes, std::move(pinned)});
        }
        success = pending.size() == missing.size();
    } catch (...) {
        success = false;
    }
    if (!success) {
        pending.clear();
        for (const size_t offset : committed) {
            const size_t bytes = std::min(s.chunk_bytes, reserved_bytes() - offset);
            if (!ggml_backend_host_reservation_decommit(s.reservation.get(), offset, bytes)) s.failed = true;
        }
        return false;
    }
    for (auto & chunk : pending) {
        s.registered += chunk.bytes;
        s.chunks.push_back(std::move(chunk));
    }
    std::sort(s.chunks.begin(), s.chunks.end(), [](const impl::chunk & a, const impl::chunk & b) { return a.offset < b.offset; });
    s.rows = rows;
    return true;
}

bool llama_kv_stream_host_reservation::truncate(size_t rows) {
    auto & s = *state;
    if (s.failed || rows > s.rows) return false;
    for (size_t layer = 0; layer < s.config.layers; ++layer) for (bool value : {false, true}) {
        const size_t token_bytes = value ? s.planes.v_token_bytes : s.planes.k_token_bytes;
        const size_t first = s.base_offset + layer*s.stride + (value ? s.planes.v_offset : 0);
        s.zero(first + rows*token_bytes, (s.rows - rows)*token_bytes);
    }
    return true;
}

void llama_kv_stream_host_reservation::clear(uint8_t value) {
    auto & s = *state;
    for (const auto & chunk : s.chunks) ggml_backend_buffer_clear(chunk.buffer.get(), value);
}

bool llama_kv_stream_host_reservation::ranges(uint32_t layer, ggml_kv_stream_operand operand,
        size_t offset, size_t bytes, std::vector<llama_kv_stream_host_range> & output) const {
    const auto & s = *state;
    if (s.failed || layer >= s.config.layers || !bytes ||
            (operand != ggml_kv_stream_operand::k && operand != ggml_kv_stream_operand::v)) return false;
    const bool value = operand == ggml_kv_stream_operand::v;
    const size_t available = s.rows*(value ? s.planes.v_token_bytes : s.planes.k_token_bytes);
    if (offset > available || bytes > available - offset) return false;
    size_t first = s.base_offset + layer*s.stride + (value ? s.planes.v_offset : 0) + offset;
    try {
        std::vector<llama_kv_stream_host_range> ranges;
        while (bytes) {
            const size_t chunk_offset = first - first%s.chunk_bytes;
            const auto chunk = s.find(chunk_offset);
            if (chunk == s.chunks.end() || chunk->offset != chunk_offset) return false;
            const size_t in_chunk = first - chunk_offset;
            const size_t count = std::min(bytes, chunk->bytes - in_chunk);
            ranges.push_back({ggml_backend_buffer_ptr(ggml_backend_buffer_retain(chunk->buffer.get())), in_chunk, count});
            first += count;
            bytes -= count;
        }
        output = std::move(ranges);
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}
