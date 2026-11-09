#include "../ggml/src/ggml-kv-stream.h"
#include "../ggml/src/ggml-kv-stream-copy.h"
#include "testing.h"
#include "../src/llama-kv-stream-selection.h"
#include <memory>
#include <cstring>
#include <algorithm>

#include <limits>

using status = ggml_kv_stream_status;
using operand = ggml_kv_stream_operand;

static ggml_kv_stream_shape shape() {
    return {GGML_TYPE_F16, GGML_TYPE_F16, 32, 32, 1, 16, 16};
}

struct allocation {
    ggml_backend_memory_arena_t arena = nullptr;
    ggml_backend_memory_lease_t lease = nullptr;

    explicit allocation(size_t bytes = 4096) {
        arena = ggml_backend_memory_arena_new(ggml_backend_cpu_buffer_type(), bytes);
        ggml_backend_memory_region region;
        if (!arena ||
                !ggml_backend_memory_arena_begin(arena, GGML_BACKEND_MEMORY_PLAN_NONE) ||
                !ggml_backend_memory_arena_reserve(arena, 1, bytes, 16, GGML_BACKEND_MEMORY_REGION_NONE, &region) ||
                !ggml_backend_memory_arena_commit(arena)) return;
        lease = ggml_backend_memory_arena_acquire(arena, 1);
    }

    ~allocation() {
        ggml_backend_memory_lease_free(lease);
        ggml_backend_memory_arena_free(arena);
    }

    bool valid() const {
        return arena && lease;
    }
};

static ggml_kv_stream_span_source span(
        ggml_backend_memory_lease_t k, ggml_backend_memory_lease_t v,
        size_t first, size_t tokens, size_t k_offset, size_t v_offset) {
    return {k, v, first, tokens, k_offset, v_offset};
}

int main() {
    testing t;

    t.test("native_column_projection_gathers_private_rows_without_reordering_attention", [](testing & t) {
        allocation a,b;
        const auto geometry=shape();
        ggml_kv_stream_layout tile;
        if (!t.assert_true(a.valid() && b.valid() && ggml_kv_stream_layout_make(geometry,256,tile).status==status::success)) return;
        allocation scratch(tile.bytes);
        if (!t.assert_true(scratch.valid())) return;
        auto * destination=ggml_backend_memory_lease_buffer(scratch.lease);
        ggml_kv_stream_span_plan_t plans[2]={};
        ggml_kv_stream_span_plan_view views[2];
        for (size_t owner=0;owner<2;++owner) {
            auto * lease=owner ? b.lease : a.lease;
            auto * buffer=ggml_backend_memory_lease_buffer(lease);
            auto * bytes=static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer));
            for (size_t row=0;row<16;++row) for (size_t plane=0;plane<2;++plane)
                std::memset(bytes+plane*1024+row*64,int(owner*61+row*7+plane*3),64);
            const auto source=span(lease,lease,0,16,0,1024);
            if (!t.assert_true(ggml_kv_stream_span_plan_make(geometry,&source,1,16,1,plans[owner]).status==status::success &&
                    ggml_kv_stream_span_plan_get_view(plans[owner],views[owner]))) return;
        }
        struct retirement {
            ggml_kv_stream_span_plan_t * plans;
            ~retirement() {for (size_t i=0;i<2;++i) ggml_kv_stream_span_plan_free(plans[i]);}
        } retained{plans};
        std::vector<ggml_kv_stream_gather_row> columns(256);
        size_t used=0;
        for (size_t i=0;i<5;++i) columns[used++]={0,i};
        for (size_t i=0;i<3;++i) columns[used++]={1,i};
        for (size_t i=0;i<8;++i) {columns[used++]={0,5+i};columns[used++]={1,3+i};}
        std::vector<ggml_kv_stream_copy_rectangle> copies;
        ggml_backend_buffer_clear(destination,0xa5);
        if (!t.assert_true(ggml_kv_stream_gather_make(views,2,columns,0,destination,copies))) return;
        t.assert_equal(size_t(8),copies.size());
        auto * output=static_cast<uint8_t *>(ggml_backend_buffer_get_base(destination));
        t.assert_true(std::all_of(output,output+tile.bytes,[](uint8_t x) {return x==0xa5;}));
        std::memset(output,0,tile.bytes);
        for (const auto & copy : copies) for (size_t row=0;row<copy.rows;++row)
            std::memcpy(output+copy.destination_offset+row*copy.destination_stride,
                static_cast<const uint8_t *>(ggml_backend_buffer_get_base(copy.source))+copy.source_offset+row*copy.source_stride,copy.row_bytes);
        std::vector<uint8_t> expected(tile.bytes,0);
        for (size_t col=0;col<used;++col) for (size_t plane=0;plane<2;++plane)
            std::memset(expected.data()+(plane ? tile.v_offset : 0)+col*64,
                int(columns[col].source*61+columns[col].token*7+plane*3),64);
        t.assert_true(!std::memcmp(expected.data(),output,tile.bytes));
        const auto saved=copies;
        for (int change=0;change<8;++change) {
            auto bad_columns=columns;auto x=views[0],y=views[1];
            if (change==0) bad_columns[0].source=2;
            if (change==1) bad_columns[0].token=16;
            if (change==2) bad_columns[255].token=1;
            if (change==3) bad_columns.pop_back();
            if (change==4) y.count=4;
            if (change==5) y.active_tokens=15;
            if (change==6) y=x;
            if (change==7) y.shape.type_v=GGML_TYPE_Q8_0;
            const ggml_kv_stream_span_plan_view invalid[]={x,y};
            t.assert_true(!ggml_kv_stream_gather_make(invalid,2,bad_columns,0,destination,copies));
            t.assert_true(copies.size()==saved.size() && copies.front().source==saved.front().source &&
                copies.back().destination_offset==saved.back().destination_offset);
        }
        t.assert_true(!ggml_kv_stream_gather_make(views,2,columns,1,destination,copies));
        auto packet=columns;
        for (auto & row : packet) if (row.source==0) row={};
        auto unloaded=views[0];unloaded.spans=nullptr;unloaded.count=0;
        const ggml_kv_stream_span_plan_view partial[]={unloaded,views[1]};
        if (!t.assert_true("a packet may omit another slot's unloaded pages",
                ggml_kv_stream_gather_make(partial,2,packet,0,destination,copies))) return;
        const size_t starts[]={0,3,10},counts[]={3,7,6},offsets[]={1664,768,0};
        ggml_kv_stream_span wrapped[2][3];
        for (size_t owner=0;owner<2;++owner) {
            auto * buffer=ggml_backend_memory_lease_buffer(owner ? b.lease : a.lease);
            auto * bytes=static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer));
            for (size_t part=0;part<3;++part) {
                wrapped[owner][part]={buffer,buffer,starts[part],counts[part],offsets[part],offsets[part]+counts[part]*64};
                for (size_t row=0;row<counts[part];++row) for (size_t plane=0;plane<2;++plane)
                    std::memset(bytes+offsets[part]+plane*counts[part]*64+row*64,int(owner*61+(starts[part]+row)*7+plane*3),64);
            }
            views[owner].spans=wrapped[owner];views[owner].count=3;
        }
        columns.resize(512);
        uint32_t random=0x512256;
        for (size_t trial=0;trial<64;++trial) {
            for (auto & row : columns) {
                random=1664525*random+1013904223;
                row=random%7 ? ggml_kv_stream_gather_row{(random>>16)%2,(random>>8)%16} : ggml_kv_stream_gather_row{};
            }
            for (size_t first : {size_t(0),size_t(256)}) {
                if (!t.assert_true(ggml_kv_stream_gather_make(views,2,columns,first,destination,copies))) return;
                std::memset(output,0,tile.bytes);std::fill(expected.begin(),expected.end(),0);
                for (const auto & copy : copies) for (size_t row=0;row<copy.rows;++row)
                    std::memcpy(output+copy.destination_offset+row*copy.destination_stride,
                        static_cast<const uint8_t *>(ggml_backend_buffer_get_base(copy.source))+copy.source_offset+row*copy.source_stride,copy.row_bytes);
                for (size_t col=0;col<256;++col) if (columns[first+col].source!=UINT32_MAX) for (size_t plane=0;plane<2;++plane)
                    std::memset(expected.data()+(plane ? tile.v_offset : 0)+col*64,
                        int(columns[first+col].source*61+columns[first+col].token*7+plane*3),64);
                if (!t.assert_true("wrapped sources retain native columns and exact padding",!std::memcmp(expected.data(),output,tile.bytes))) return;
            }
        }
    });

    t.test("exclusive_slot_pages_cover_every_resident_ring_rotation_and_tail", [](testing & t) {
        llama_kv_stream_selection selected;
        if (!t.assert_true(llama_kv_stream_select({128*1024,2},{{0,90*1024},{1,30*1024}},selected)==llama_kv_stream_selection_status::success)) return;
        llama_kv_stream_page_partition partition;
        if (!t.assert_true(llama_kv_stream_partition_pages(selected,7,{{0,2},{1,2}},partition).status==llama_kv_stream_page_partition_status::success)) return;
        size_t cases=0;
        for (auto pair : {std::pair{GGML_TYPE_F16,GGML_TYPE_F16},std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},std::pair{GGML_TYPE_Q5_1,GGML_TYPE_Q5_1}}) {
            const ggml_kv_stream_shape geometry{pair.first,pair.second,32,32,1,16,16};
            ggml_kv_stream_layout page;
            if (!t.assert_true(ggml_kv_stream_layout_make(geometry,16,page).status==status::success)) return;
            auto * type=ggml_backend_cpu_buffer_type();const size_t guard=ggml_backend_buft_get_alignment(type);
            std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)> parent(
                ggml_backend_memory_arena_new(type,7*page.bytes+2*guard),ggml_backend_memory_arena_free);
            if (!t.assert_true(parent && ggml_backend_memory_arena_begin(parent.get(),0))) return;
            for (const auto & slot : partition.slots) if (!t.assert_true(ggml_backend_memory_arena_reserve_at(parent.get(),10+slot.slot,
                    guard+size_t(slot.first_page)*page.bytes,size_t(slot.pages)*page.bytes,guard,GGML_BACKEND_MEMORY_REGION_PERSISTENT,nullptr))) return;
            if (!t.assert_true(ggml_backend_memory_arena_commit(parent.get()))) return;
            using lease_ptr=std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>;
            lease_ptr owners[]={lease_ptr(ggml_backend_memory_arena_acquire(parent.get(),10),ggml_backend_memory_lease_free),
                lease_ptr(ggml_backend_memory_arena_acquire(parent.get(),11),ggml_backend_memory_lease_free)};
            if (!t.assert_true(owners[0] && owners[1])) return;
            ggml_backend_buffer_clear(ggml_backend_memory_arena_parent(parent.get()),0xa5);
            for (size_t owner=0;owner<2;++owner) {
                auto * storage=ggml_backend_memory_lease_buffer(owners[owner].get());
                auto * other=ggml_backend_memory_lease_buffer(owners[1-owner].get());
                const auto * peer=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(other));
                const std::vector<uint8_t> saved(peer,peer+ggml_backend_buffer_get_size(other));
                auto * base=static_cast<uint8_t *>(ggml_backend_buffer_get_base(storage));
                const size_t quota=partition.slots[owner].pages;
                ggml_kv_stream_layout full;
                if (!t.assert_true(ggml_kv_stream_layout_make(geometry,quota*16,full).status==status::success)) return;
                for (size_t resident=0;resident<=quota;++resident) for (size_t ring=0;ring<=quota-resident;++ring)
                for (size_t head=0;head<std::max(size_t(1),ring);++head) for (size_t live=0;live<=ring;++live) {
                    if (!resident && !live) continue;
                    for (size_t tail=1;tail<=16;++tail) {
                        const size_t active=(resident+live-1)*16+tail;
                        ggml_backend_buffer_clear(storage,0xa5);
                        for (size_t token=0;token<active;++token) {
                            const size_t logical_page=token/16;
                            const size_t physical_page=logical_page<resident ? logical_page : resident+(head+logical_page-resident)%ring;
                            const size_t row=physical_page*16+token%16;
                            std::memset(base+row*full.k_token_bytes,int((owner*71+token*7)%251),full.k_token_bytes);
                            std::memset(base+full.v_offset+row*full.v_token_bytes,int((owner*71+token*7+3)%251),full.v_token_bytes);
                        }
                        std::vector<ggml_kv_stream_span_source> sources;
                        size_t first=0;
                        const auto append=[&](size_t count,size_t physical) {
                            if (!count) return;
                            sources.push_back({owners[owner].get(),owners[owner].get(),first,count,
                                physical*full.k_token_bytes,full.v_offset+physical*full.v_token_bytes});first+=count;
                        };
                        append(std::min(active,resident*16),0);
                        if (first<active) append(std::min(active-first,(ring-head)*16),(resident+head)*16);
                        if (first<active) append(active-first,resident*16);
                        if (!t.assert_true(first==active && !sources.empty() && sources.size()<=3)) return;
                        ggml_kv_stream_span_plan_t plan=nullptr;
                        if (!t.assert_true(ggml_kv_stream_span_plan_make(geometry,sources.data(),sources.size(),active,1,plan).status==status::success)) return;
                        ggml_kv_stream_span_plan_view view;
                        const bool valid=ggml_kv_stream_span_plan_get_view(plan,view);
                        bool exact=valid;
                        for (size_t i=0;valid && i<view.count;++i) for (size_t row=0;row<view.spans[i].tokens;++row) {
                            const auto & span=view.spans[i];const auto token=span.token_begin+row;
                            const auto * k=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(span.k_buffer))+span.k_offset+row*full.k_token_bytes;
                            const auto * v=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(span.v_buffer))+span.v_offset+row*full.v_token_bytes;
                            exact&=std::all_of(k,k+full.k_token_bytes,[&](uint8_t x) {return x==uint8_t((owner*71+token*7)%251);}) &&
                                std::all_of(v,v+full.v_token_bytes,[&](uint8_t x) {return x==uint8_t((owner*71+token*7+3)%251);});
                        }
                        ggml_kv_stream_span_plan_free(plan);
                        if (!t.assert_true("linear fragments equal the independent cyclic page oracle",exact)) return;
                        ++cases;
                    }
                }
                t.assert_true("every other slot page stays untouched",!std::memcmp(saved.data(),peer,saved.size()));
            }
            const auto * root=static_cast<const uint8_t *>(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(parent.get())));
            t.assert_true(std::all_of(root,root+guard,[](uint8_t v) {return v==0xa5;}) &&
                std::all_of(root+guard+7*page.bytes,root+2*guard+7*page.bytes,[](uint8_t v) {return v==0xa5;}));
        }
        t.assert_true(cases>1000);
    });
    t.test("one_and_multiple_spans_cover_one_logical_sequence", [](testing & t) {
        allocation storage;
        if (!t.assert_true(storage.valid())) return;

        ggml_kv_stream_span_plan_t one = nullptr;
        const auto single = span(storage.lease, storage.lease, 0, 8, 0, 1024);
        if (!t.assert_true(ggml_kv_stream_span_plan_make(shape(), &single, 1, 8, 4, one).status == status::success)) {
            return;
        }
        ggml_kv_stream_span_plan_view view;
        t.assert_true(ggml_kv_stream_span_plan_get_view(one, view));
        t.assert_equal(size_t(1), view.count);
        t.assert_equal(size_t(8), view.active_tokens);
        t.assert_equal(size_t(4), view.query_tokens);
        t.assert_equal(size_t(512), view.k_bytes);
        t.assert_equal(size_t(512), view.v_bytes);
        t.assert_equal(size_t(8), view.spans[0].tokens);
        ggml_kv_stream_span_plan_free(one);

        const ggml_kv_stream_span_source split[] = {
            span(storage.lease, storage.lease, 0, 3, 0, 1024),
            span(storage.lease, storage.lease, 3, 5, 256, 1280),
        };
        ggml_kv_stream_span_plan_t multiple = nullptr;
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), split, 2, 8, 1, multiple).status == status::success);
        t.assert_true(ggml_kv_stream_span_plan_get_view(multiple, view));
        t.assert_equal(size_t(2), view.count);
        t.assert_equal(size_t(3), view.spans[1].token_begin);
        ggml_kv_stream_span_plan_free(multiple);
    });

    t.test("rejects_gaps_overlaps_and_reordered_spans_transactionally", [](testing & t) {
        allocation storage;
        if (!t.assert_true(storage.valid())) return;
        const size_t starts[][2] = {{0, 5}, {0, 3}, {4, 0}};
        const size_t counts[][2] = {{4, 3}, {4, 5}, {4, 4}};
        for (size_t i = 0; i < 3; ++i) {
            const ggml_kv_stream_span_source spans[] = {
                span(storage.lease, storage.lease, starts[i][0], counts[i][0], 0, 1024),
                span(storage.lease, storage.lease, starts[i][1], counts[i][1], 256, 1280),
            };
            auto * sentinel = reinterpret_cast<ggml_kv_stream_span_plan_t>(uintptr_t(1));
            const auto result = ggml_kv_stream_span_plan_make(shape(), spans, 2, 8, 1, sentinel);
            t.assert_true(result.status == status::invalid_span);
            t.assert_true(sentinel == reinterpret_cast<ggml_kv_stream_span_plan_t>(uintptr_t(1)));
        }
    });

    t.test("rejects_empty_spans_and_invalid_query_width", [](testing & t) {
        allocation storage;
        if (!t.assert_true(storage.valid())) return;
        ggml_kv_stream_span_plan_t plan = nullptr;
        auto value = span(storage.lease, storage.lease, 0, 8, 0, 1024);
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), nullptr, 0, 8, 1, plan).status == status::invalid_span);
        value.tokens = 0;
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), &value, 1, 8, 1, plan).status == status::invalid_span);
        value.tokens = 8;
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), &value, 1, 8, 0, plan).status == status::invalid_shape);
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), &value, 1, 8, 9, plan).status == status::invalid_shape);
        t.assert_true(plan == nullptr);
    });

    t.test("validates_offsets_alignment_and_each_buffer_extent", [](testing & t) {
        allocation k(512);
        allocation v(512);
        if (!t.assert_true(k.valid() && v.valid())) return;

        ggml_kv_stream_span_plan_t plan = nullptr;
        auto value = span(k.lease, v.lease, 0, 4, 256, 256);
        t.assert_true(ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan).status == status::success);
        ggml_kv_stream_span_plan_free(plan);
        plan = nullptr;

        value.k_offset = 257;
        auto result = ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan);
        t.assert_true(result.status == status::invalid_alignment && result.operand == operand::k);
        value.k_offset = 272;
        result = ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan);
        t.assert_true(result.status == status::invalid_buffer && result.operand == operand::k);
        value.k_offset = 256;
        value.v_offset = 272;
        result = ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan);
        t.assert_true(result.status == status::invalid_buffer && result.operand == operand::v);
        value.v_offset = 256;
        value.k_lease = nullptr;
        result = ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan);
        t.assert_true(result.status == status::invalid_buffer && result.operand == operand::k);
        t.assert_true(plan == nullptr);
    });

    t.test("checked_arithmetic_rejects_logical_and_byte_overflow", [](testing & t) {
        allocation storage;
        if (!t.assert_true(storage.valid())) return;
        ggml_kv_stream_span_plan_t plan = nullptr;
        auto value = span(storage.lease, storage.lease, std::numeric_limits<size_t>::max() - 1, 4, 0, 0);
        t.assert_true(ggml_kv_stream_span_plan_make(
            shape(), &value, 1, std::numeric_limits<size_t>::max(), 1, plan).status == status::overflow);
        value = span(storage.lease, storage.lease, 0, 2, std::numeric_limits<size_t>::max() - 63, 0);
        const auto result = ggml_kv_stream_span_plan_make(shape(), &value, 1, 2, 1, plan);
        t.assert_true(result.status == status::overflow && result.operand == operand::k);
        t.assert_true(plan == nullptr);
    });

    t.test("plan_retains_arena_lease_until_final_release", [](testing & t) {
        allocation k(1024);
        allocation v(1024);
        if (!t.assert_true(k.valid() && v.valid())) return;
        void * const k_base = ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(k.lease));
        void * const v_base = ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(v.lease));
        const auto value = span(k.lease, v.lease, 0, 4, 0, 0);
        ggml_kv_stream_span_plan_t plan = nullptr;
        if (!t.assert_true(ggml_kv_stream_span_plan_make(shape(), &value, 1, 4, 1, plan).status == status::success)) {
            return;
        }
        auto * retained = ggml_kv_stream_span_plan_retain(plan);
        t.assert_true(retained == plan);
        ggml_backend_memory_lease_free(k.lease);
        ggml_backend_memory_lease_free(v.lease);
        k.lease = nullptr;
        v.lease = nullptr;

        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(k.arena));
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(v.arena));
        t.assert_true(ggml_backend_memory_arena_quiesce(k.arena));
        t.assert_true(ggml_backend_memory_arena_quiesce(v.arena));
        t.assert_true(ggml_backend_memory_arena_get_state(k.arena) == GGML_BACKEND_MEMORY_ARENA_STATE_DRAINING);
        t.assert_true(ggml_backend_memory_arena_get_state(v.arena) == GGML_BACKEND_MEMORY_ARENA_STATE_DRAINING);

        ggml_kv_stream_span_plan_view view;
        t.assert_true(ggml_kv_stream_span_plan_get_view(plan, view));
        t.assert_true(ggml_backend_buffer_get_base(view.spans[0].k_buffer) == k_base);
        t.assert_true(ggml_backend_buffer_get_base(view.spans[0].v_buffer) == v_base);
        ggml_backend_buffer_clear(view.spans[0].k_buffer, 0xa5);
        ggml_backend_buffer_clear(view.spans[0].v_buffer, 0x5a);
        t.assert_equal(uint8_t(0xa5), static_cast<const uint8_t *>(k_base)[0]);
        t.assert_equal(uint8_t(0x5a), static_cast<const uint8_t *>(v_base)[0]);

        ggml_kv_stream_span_plan_free(plan);
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(k.arena));
        t.assert_true(ggml_kv_stream_span_plan_get_view(retained, view));
        ggml_kv_stream_span_plan_free(retained);
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(k.arena));
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(v.arena));
        t.assert_true(ggml_backend_memory_arena_get_state(k.arena) == GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT);
        t.assert_true(ggml_backend_memory_arena_get_state(v.arena) == GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT);
    });

    t.test("invalid_plan_view_leaves_output_unchanged", [](testing & t) {
        ggml_kv_stream_span_plan_view view;
        view.count = 77;
        t.assert_true(!ggml_kv_stream_span_plan_get_view(nullptr, view));
        t.assert_equal(size_t(77), view.count);
        t.assert_true(ggml_kv_stream_span_plan_retain(nullptr) == nullptr);
        ggml_kv_stream_span_plan_free(nullptr);
    });

    return t.summary();
}
