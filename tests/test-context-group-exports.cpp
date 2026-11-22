#include "llama-context.h"
#include "llama-context-memory.h"
#include "llama-kv-stream-logical-cache.h"
#include "llama-kv-stream-selection.h"

#include <memory>

// Volatile reads keep symbol references live without executing model work.
template <typename T>
static bool linked_export(T symbol) {
    T volatile reference = symbol;
    return reference == symbol;
}

static void destroy_group(llama_context_group * group) {
    delete group;
}

static llama_context * create_selected_context(const llama_model & model, llama_context_params params,
        const std::vector<llama_seq_id> & selected, uint32_t rows, bool explicit_rows) {
    return explicit_rows ? new llama_context(model, params, selected, rows) : new llama_context(model, params, selected);
}

static void destroy_context(llama_context * context) {
    delete context;
}

static llama_batch_allocr * create_batch_allocr(uint32_t positions, bool mixed) {
    return new llama_batch_allocr(positions, mixed);
}

static llama_batch_compat * create_batch_compat(llama_context * context, const llama_batch & batch, size_t rows) {
    return new llama_batch_compat(context, batch, rows);
}

static void destroy_batch_compat(llama_batch_compat * batch) {
    delete batch;
}

int main() {
    using create_group = std::unique_ptr<llama_context_group> (*)(const llama_model &, llama_context_params);
    using create_table = std::unique_ptr<llama_context_group> (*)(const llama_model &, llama_context_params,
            std::shared_ptr<llama_kv_stream_slot_table>);

    return !(linked_export(static_cast<create_group>(&llama_context_group::create)) &&
             linked_export(static_cast<create_table>(&llama_context_group::create)) &&
             linked_export(&destroy_group) &&
             linked_export(&llama_context_group_checkpoint::payload_size) &&
             linked_export(&llama_context::group_parent) &&
             linked_export(&llama_context::get_cparams) &&
             linked_export(&llama_context_group::prepare_shared_mtp) &&
             linked_export(&llama_context_group::prepare_dflash) &&
             linked_export(&llama_context_group::selected_state_limit) &&
             linked_export(&llama_context_group::prepare_vision_parent) &&
             linked_export(&llama_context_group::quote_vision) &&
             linked_export(&llama_context_group::lend_vision) &&
             linked_export(&llama_context_group::release_vision) &&
             linked_export(&llama_context_group::activate) &&
             linked_export(&llama_context_group::park) &&
             linked_export(&llama_context_group::discard) &&
             linked_export(&llama_context_group::context) &&
             linked_export(&llama_context_group::auxiliary_context) &&
             linked_export(&llama_context_group::metadata_context) &&
             linked_export(&llama_context_group::auxiliary_metadata_context) &&
             linked_export(&llama_context_group::source_table) &&
             linked_export(&llama_context_group::source_handles) &&
             linked_export(&llama_context_group::save_request) &&
             linked_export(&llama_context_group::write_request) &&
             linked_export(&llama_context_group::read_request) &&
             linked_export(&llama_context_group::write_checkpoint) &&
             linked_export(&llama_context_group::read_checkpoint) &&
             linked_export(&llama_context_group::load_request) &&
             linked_export(&llama_context_group::reset_request) &&
             linked_export(&llama_context_group::advance_request) &&
             linked_export(&llama_context_group::parked_payload_bytes) &&
             linked_export(&create_selected_context) &&
             linked_export(&destroy_context) &&
             linked_export(&create_batch_allocr) &&
             linked_export(&create_batch_compat) &&
             linked_export(&destroy_batch_compat) &&
             linked_export(&llama_batch_allocr::init) &&
             linked_export(&llama_batch_allocr::get_batch) &&
             linked_export(&llama_context_memory_create_status_name) &&
             linked_export(&llama_kv_stream_select) &&
             linked_export(&llama_kv_stream_prefill_plan_make) &&
             linked_export(&llama_kv_stream_host::cache_id) &&
             linked_export(&llama_kv_stream_host::bytes) &&
             linked_export(&llama_kv_stream_host::committed_bytes) &&
             linked_export(&llama_kv_stream_host::registered_bytes) &&
             linked_export(&llama_kv_stream_content::generation) &&
             linked_export(&llama_kv_stream_logical_cache::host) &&
             linked_export(&llama_kv_stream_logical_cache::content) &&
             linked_export(&llama_kv_stream_logical_cache::identity) &&
             linked_export(&llama_kv_stream_logical_cache::tokens) &&
             linked_export(&llama_kv_stream_logical_cache::ready) &&
             linked_export(&llama_kv_stream_slot_table::get) &&
             linked_export(&llama_kv_stream_slot_table::limits) &&
             linked_export(&llama_kv_stream_request_group::admit) &&
             linked_export(&llama_kv_stream_request_group::phase) &&
             linked_export(&llama_kv_stream_request_group::next_preparation) &&
             linked_export(&llama_kv_stream_request_group::preparing) &&
             linked_export(&llama_kv_stream_request_group::prepared) &&
             linked_export(&llama_kv_stream_request_group::busy) &&
             linked_export(&llama_kv_stream_request_group::selected) &&
             linked_export(&llama_kv_stream_request_group::plan) &&
             linked_export(&llama_kv_stream_request_group::activate) &&
             linked_export(&llama_kv_stream_request_group::needs_reselection) &&
             linked_export(&llama_kv_stream_request_group::submission) &&
             linked_export(&llama_kv_stream_request_group::advance) &&
             linked_export(&llama_kv_stream_request_group::finish) &&
             linked_export(&llama_kv_stream_request_group::release) &&
             linked_export(&llama_context::bind_group_samplers_from) &&
             linked_export(&llama_context::quote_target_group) &&
             linked_export(&llama_context::get_model) &&
             linked_export(&llama_context::set_embeddings_nextn) &&
             linked_export(&llama_context::set_embeddings_layer_inp));
}
