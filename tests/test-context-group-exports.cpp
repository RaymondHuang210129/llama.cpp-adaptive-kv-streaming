#include "llama-context.h"

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
             linked_export(&llama_context_group::parked_payload_bytes));
}
