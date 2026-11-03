#include "../tools/server/server-common.h"
#include "../tools/server/server-task.h"
#include "ngram-map.h"
#include "testing.h"
#include <cstring>

static common_params qualified() {
    common_params p;
    p.mmproj.path = "projector.gguf";
    p.shared_device_memory_bytes = 1024*1048576;
    p.fit_params = false; p.n_parallel = 1; p.mmproj_use_gpu = true;
    p.cache_type_k = GGML_TYPE_Q8_0; p.cache_type_v = GGML_TYPE_Q4_0;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    return p;
}

int main(int argc, char ** argv) {
    testing t;
    const bool k4v_readiness = argc == 2 && !std::strcmp(argv[1], "--k4v-vision-readiness");
    if (argc > 1 && !k4v_readiness) return 2;
    if (k4v_readiness) t.set_filter("k4v_vision_readiness");
    if (k4v_readiness) t.test("k4v_vision_readiness", [](testing & t) {
        mtmd::input_chunks_ptr chunks(mtmd_test_create_input_chunks());
        const auto * image = mtmd_input_chunks_get(chunks.get(), 1);
        if (!t.assert_true(image && mtmd_input_chunk_get_type(image) == MTMD_INPUT_CHUNK_TYPE_IMAGE)) return;
        const auto rows = mtmd_input_chunk_get_n_tokens(image);
        const auto positions = mtmd_input_chunk_get_n_pos(image);
        if (!t.assert_true(rows > 0 && positions > 0)) return;
        const auto lookup = [&](const llama_tokens & prefix, const llama_tokens & suffix) {
            server_tokens prompt(prefix, true);
            prompt.push_back(image);
            for (auto id : suffix) prompt.push_back(id);
            const auto snapshot = prompt.get_speculative_text();
            const auto & text = snapshot.tokens;
            t.assert_equal(prefix.size() + suffix.size(), text.size());
            t.assert_equal(llama_pos(prefix.size() + positions + suffix.size()), prompt.pos_next());
            common_ngram_map map(2, 3, false, 1);
            common_ngram_map_begin(map, text, snapshot.ngram_begin);
            llama_tokens copied;
            common_ngram_map_draft(map, text, 20, copied);
            t.out << "K4V image lookup rows=" << prompt.size() << " text=" << text.size()
                  << " model_next=" << prompt.pos_next() << " copied=" << copied.size() << '\n';
            return copied;
        };
        t.assert_true("real post-image repetition remains usable",
            lookup({99,55}, {99,10,20,30,40,50,98,10}) == llama_tokens({30,40,50}));
        t.assert_true("first post-image occurrence is a real key, not a zero sentinel",
            lookup({}, {10,20,30,40,50,98,10}) == llama_tokens({30,40,50}));
        t.assert_true("lookup key cannot join text across an image",
            lookup({99,10}, {20,30,40,50,98,10}).empty());
        t.assert_true("copied continuation cannot cross an image",
            lookup({99,10,20,30}, {40,50,98,10}).empty());
        t.assert_true("initial suffix-only policy excludes pre-image values",
            lookup({99,10,20,30,40,50,98}, {10}).empty());
        auto p = qualified();
        p.n_batch = p.n_ubatch = 256;
        p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
        p.speculative.draft.mparams.path = "matching-dflash.gguf";
        p.speculative.draft.n_max = 3;
        p.speculative.ngram_map_k4v.size_n = 2;
        p.speculative.ngram_map_k4v.size_m = 7;
        p.speculative.ngram_map_k4v.min_hits = 1;
        t.assert_true(!common_streamed_dflash_config_error(p) && !server_vision_arena_config_error(p));
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.tokens = server_tokens(llama_tokens{1,2,3}, true);
        task.params.speculative = p.speculative;
        t.assert_true(!server_vision_arena_request_error(task, false, &p.speculative));
        p.speculative.types.insert(p.speculative.types.begin(), COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        task.params.speculative = p.speculative;
        uint32_t depth = 0;
        t.assert_true(!common_streamed_dflash_verification_error(p, depth) && depth == 7);
        t.assert_true("mixed image common admission", !common_streamed_dflash_config_error(p));
        t.assert_true("mixed image server admission", !server_vision_arena_config_error(p));
        t.assert_true("mixed image request admission", !server_vision_arena_request_error(task, false, &p.speculative));
    });
    t.test("vision_memory_report_requires_a_real_shared_projector_profile",[](testing & t) {
        auto p=qualified();p.vision_memory_report=true;
        t.assert_true(!server_vision_arena_config_error(p));
        p.mmproj.path.clear();t.assert_true(server_vision_arena_config_error(p));
        p=qualified();p.vision_memory_report=true;p.shared_device_memory_bytes=0;
        t.assert_true(server_vision_arena_config_error(p));
    });
    t.test("vision_cache_formats_are_validated_by_capabilities_not_one_pair", [](testing & t) {
        const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL};
        for (bool mtp : {false, true}) for (auto key : types) for (auto value : types) {
            auto p = qualified();
            p.cache_type_k = key;
            p.cache_type_v = value;
            if (mtp) p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            t.assert_true(server_vision_arena_config_error(p) == nullptr);
        }
    });
    t.test("serial_no_mtp_vision_arena_is_admitted_without_changing_params", [](testing & t) {
        auto p = qualified();
        t.assert_true(server_uses_vision_arena(p));
        t.assert_true(server_vision_arena_config_error(p) == nullptr);
        t.assert_true(p.shared_device_memory_bytes == 1024*1048576 && !p.fit_params);
    });
    t.test("unsupported_vision_configurations_stay_rejected", [](testing & t) {
        auto check = [&](auto change) { auto p = qualified(); change(p); t.assert_true(server_vision_arena_config_error(p) != nullptr); };
        check([](auto & p) { p.n_parallel = 2; });
        check([](auto & p) { p.fit_params = true; });
        check([](auto & p) { p.shared_device_memory_bytes = 0; p.kv_stream_pool_bytes = 1048576; });
        check([](auto & p) { p.kv_stream_pool_bytes = 1048576; });
        check([](auto & p) { p.mmproj_use_gpu = false; });
        check([](auto & p) { p.cache_type_k = GGML_TYPE_I32; });
        check([](auto & p) { p.cache_type_v = GGML_TYPE_Q8_1; });
        check([](auto & p) { p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; });
        check([](auto & p) { p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP}; p.speculative.draft.n_max = 4; });
        check([](auto & p) { p.speculative.types = {COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE}; });
        check([](auto & p) { p.kv_stream_auxiliary_layers = 1; });
        check([](auto & p) { p.embedding = true; });
        check([](auto & p) { p.no_kv_offload = true; });
        check([](auto & p) {
            p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            p.speculative.draft.cache_type_k_explicit = true;
        });
    });
    t.test("media_prefix_and_serialization_preserve_whole_chunks_and_model_positions", [](testing & t) {
        mtmd::input_chunks_ptr input(mtmd_test_create_input_chunks());
        auto * image = mtmd_input_chunks_get(input.get(),1);
        const size_t rows = mtmd_input_chunk_get_n_tokens(image);
        const auto positions = mtmd_input_chunk_get_n_pos(image);
        server_tokens a(llama_tokens{},true),b(llama_tokens{},true);
        a.push_back(1); a.push_back(image); a.push_back(2);
        b.push_back(1); b.push_back(image); b.push_back(3);
        t.assert_equal(size_t(1)+rows,a.get_common_prefix(b));
        t.assert_equal(llama_pos(2+positions),a.pos_next());
        t.assert_equal(llama_pos(1+positions),a.pos_next(1+rows));
        const auto serialized = a.serialize();
        llama_tokens packed(serialized.size()/sizeof(llama_token));
        std::memcpy(packed.data(),serialized.data(),serialized.size());
        auto restored = server_tokens::deserialize(packed,true);
        t.assert_equal(a.size(),restored.get_common_prefix(a));
        t.assert_equal(a.pos_next(),restored.pos_next());
        restored.keep_first(1+rows);
        t.assert_equal(size_t(1)+rows,restored.size());
        t.assert_equal(llama_pos(1+positions),restored.pos_next());
    });
    t.test("speculative_text_snapshot_keeps_media_boundaries_and_legacy_text", [](testing & t) {
        mtmd::input_chunks_ptr chunks(mtmd_test_create_input_chunks());
        const auto * image = mtmd_input_chunks_get(chunks.get(),1);
        for (bool placeholder : {false,true}) for (int layout=0;layout<5;++layout) {
            server_tokens prompt(llama_tokens{},true);
            if (layout!=0) { prompt.push_back(11); prompt.push_back(12); }
            if (placeholder) prompt.push_back_placeholder(image); else prompt.push_back(image);
            if (layout>=2) {
                if (layout!=3) prompt.push_back(13);
                if (placeholder) prompt.push_back_placeholder(image); else prompt.push_back(image);
            }
            if (layout!=1) { prompt.push_back(14);prompt.push_back(15); }
            const auto snapshot=prompt.get_speculative_text();
            t.assert_true(snapshot.tokens==prompt.get_text_tokens());
            t.assert_true(snapshot.has_media);
            t.assert_equal(snapshot.tokens.size()-(layout==1 ? 0u : 2u),snapshot.ngram_begin);
            const auto bytes=prompt.serialize();llama_tokens packed(bytes.size()/sizeof(llama_token));
            std::memcpy(packed.data(),bytes.data(),bytes.size());
            const auto restored=server_tokens::deserialize(packed,true).get_speculative_text();
            t.assert_true(restored.tokens==snapshot.tokens && restored.ngram_begin==snapshot.ngram_begin && restored.has_media);
        }
        for (const auto & tokens : {llama_tokens{},llama_tokens{1,2,3}}) {
            server_tokens prompt(tokens,true);const auto view=prompt.get_speculative_text();
            t.assert_true(view.tokens==tokens && !view.has_media && view.ngram_begin==0);
        }
        server_tokens malformed(llama_tokens{1,LLAMA_TOKEN_NULL,2},true);
        t.assert_true(malformed.get_text_tokens()==llama_tokens({1,2}));
        bool rejected=false;try { malformed.get_speculative_text(); } catch (const std::exception &) { rejected=true; }
        t.assert_true(rejected);
    });
    t.test("eager_multimodal_and_text_only_paths_keep_existing_policy", [](testing & t) {
        auto p = qualified(); p.shared_device_memory_bytes = 0; p.n_parallel = 4;
        t.assert_true(!server_uses_vision_arena(p) && !server_vision_arena_config_error(p));
        p = qualified(); p.mmproj.path.clear(); p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        t.assert_true(!server_uses_vision_arena(p) && !server_vision_arena_config_error(p));
    });
    t.test("serial_embedded_mtp_vision_is_admitted_only_with_bounded_drafting", [](testing & t) {
        auto p = qualified();
        p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        for (int length = 1; length <= 3; ++length) {
            p.speculative.draft.n_max = length;
            t.assert_true(server_vision_arena_config_error(p) == nullptr);
        }
        p.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        t.assert_true(server_vision_arena_config_error(p) != nullptr);
        p.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE}; p.kv_stream_auxiliary_layers = 1;
        t.assert_true(server_vision_arena_config_error(p) != nullptr);
    });
    t.test("declared_external_mtp_pair_reaches_loaded_model_validation",[](testing & t) {
        auto p=qualified();
        p.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        p.speculative.draft.n_max=3;
        p.speculative.draft.mparams.path="matching-assistant.gguf";
        t.assert_true(server_vision_arena_config_error(p)==nullptr);
    });
    t.test("request_cannot_override_the_qualified_execution_mode", [](testing & t) {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.tokens = server_tokens(llama_tokens{1,2,3},true);
        t.assert_true(!server_vision_arena_request_error(task));
        task.params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        t.assert_true(server_vision_arena_request_error(task) != nullptr);
        task.params.speculative.draft.n_max = 3;
        t.assert_true(server_vision_arena_request_error(task, true) == nullptr);
        task.params.speculative.draft.n_max = 4;
        t.assert_true(server_vision_arena_request_error(task, true) != nullptr);
        task.params.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE}; task.params.lora[0] = 1.0f;
        t.assert_true(server_vision_arena_request_error(task) != nullptr);
        task.params.lora.clear(); task.type = SERVER_TASK_TYPE_EMBEDDING;
        t.assert_true(server_vision_arena_request_error(task) != nullptr);
        task.type = SERVER_TASK_TYPE_COMPLETION; task.tokens = server_tokens(llama_tokens{LLAMA_TOKEN_NULL},true);
        t.assert_true(server_vision_arena_request_error(task) != nullptr);
    });
    t.test("configured_shared_assistant_is_not_a_request_model_override",[](testing & t) {
        auto p=qualified();p.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        p.speculative.draft.n_max=3;p.speculative.draft.mparams.path="matching-assistant.gguf";
        server_task task(SERVER_TASK_TYPE_COMPLETION);task.tokens=server_tokens(llama_tokens{1,2,3},true);
        task.params.speculative=p.speculative;
        t.assert_true(!server_vision_arena_request_error(task,true,&p.speculative));
        for (int change=0;change<8;++change) {
            server_task bad(SERVER_TASK_TYPE_COMPLETION);bad.tokens=server_tokens(llama_tokens{1,2,3},true);
            bad.params.speculative=task.params.speculative;
            switch (change) {
                case 0: bad.params.speculative.draft.mparams.path="foreign.gguf";break;
                case 1: bad.params.speculative.draft.mparams.url="https://invalid.example/model.gguf";break;
                case 2: bad.params.speculative.draft.mparams.hf_repo="foreign/repo";break;
                case 3: bad.params.speculative.draft.mparams.hf_file="foreign.gguf";break;
                case 4: bad.params.speculative.draft.mparams.docker_repo="foreign/repo";break;
                case 5: bad.params.speculative.draft.cache_type_k_explicit=true;break;
                case 6: bad.params.speculative.draft.cache_type_v_explicit=true;break;
                case 7: bad.params.speculative.draft.n_max=4;break;
            }
            t.assert_true(server_vision_arena_request_error(bad,true,&p.speculative)!=nullptr);
        }
        p.speculative.draft.n_max=1;
        t.assert_true(server_vision_arena_request_error(task,true,&p.speculative)!=nullptr);
        task.params.speculative.draft.n_max=1;
        t.assert_true(!server_vision_arena_request_error(task,true,&p.speculative));
    });
    t.test("mixed_dflash_image_admission_preserves_independent_widths_and_fixed_lookup",[](testing & t) {
        for (auto limits : {std::pair{3,7},std::pair{7,3}}) for (bool reverse : {false,true}) {
            auto p=qualified();p.n_batch=p.n_ubatch=256;
            p.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V};
            if (reverse) std::reverse(p.speculative.types.begin(),p.speculative.types.end());
            p.speculative.draft.n_max=limits.first;p.speculative.draft.mparams.path="matching-dflash.gguf";
            p.speculative.ngram_map_k4v.size_n=2;p.speculative.ngram_map_k4v.size_m=uint16_t(limits.second);p.speculative.ngram_map_k4v.min_hits=1;
            t.assert_true("common mixed image admission",!common_streamed_dflash_config_error(p));
            t.assert_true("server mixed image admission",!server_vision_arena_config_error(p));
            t.assert_equal(uint32_t(7),common_context_params_to_llama(p).n_rs_seq);
            server_task task(SERVER_TASK_TYPE_COMPLETION);task.params.speculative=p.speculative;
            task.tokens=server_tokens(llama_tokens{1,2,3},true);
            t.assert_true("configured mixed request admission",!server_vision_arena_request_error(task,false,&p.speculative));
            for (int change=0;change<10;++change) {
                auto bad=p;
                switch (change) {
                    case 0: bad.speculative.ngram_map_k4v.size_m=0;break;
                    case 1: bad.speculative.ngram_map_k4v.size_m=8;break;
                    case 2: bad.speculative.ngram_map_k4v.size_m=48;break;
                    case 3: bad.speculative.ngram_map_k4v.size_n=0;break;
                    case 4: bad.speculative.ngram_map_k4v.size_n=1025;break;
                    case 5: bad.speculative.ngram_map_k4v.min_hits=0;break;
                    case 6: bad.n_ubatch=7;break;
                    case 7: bad.speculative.types={COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V};break;
                    case 8: bad.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);break;
                    case 9: bad.speculative.draft.cache_type_k_explicit=true;break;
                }
                t.assert_true(server_vision_arena_config_error(bad));
            }
            for (int change=0;change<10;++change) {
                server_task bad(SERVER_TASK_TYPE_COMPLETION);bad.params=task.params;bad.tokens=task.tokens.clone();
                auto & spec=bad.params.speculative;
                switch (change) {
                    case 0: ++spec.ngram_map_k4v.size_m;break;
                    case 1: ++spec.ngram_map_k4v.size_n;break;
                    case 2: ++spec.ngram_map_k4v.min_hits;break;
                    case 3: spec.draft.n_max=8;break;
                    case 4: spec.draft.mparams.path="changed.gguf";break;
                    case 5: spec.draft.cache_type_v_explicit=true;break;
                    case 6: spec.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);break;
                    case 7: spec.types={COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V};spec.draft.n_max=8;break;
                    case 8: spec.types={COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V};break;
                    case 9: spec.types={COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};break;
                }
                t.assert_true(server_vision_arena_request_error(bad,false,&p.speculative));
            }
        }
    });
    t.test("pure_dflash_vision_uses_declared_serial_width_and_inherited_formats",[](testing & t) {
        for (int depth : {1,3,7}) for (auto value : {GGML_TYPE_Q4_0,GGML_TYPE_Q8_0,GGML_TYPE_Q5_1}) {
            auto p=qualified(); p.n_batch=p.n_ubatch=256;
            p.speculative.types={COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
            p.speculative.draft.n_max=depth; p.speculative.draft.mparams.path="matching-dflash.gguf";
            p.cache_type_k=value==GGML_TYPE_Q5_1 ? GGML_TYPE_Q5_1 : GGML_TYPE_Q8_0;
            p.cache_type_v=value;
            t.assert_true("pure DFlash reaches loaded-model validation",!server_vision_arena_config_error(p));
            server_task task(SERVER_TASK_TYPE_COMPLETION); task.tokens=server_tokens(llama_tokens{1,2,3},true);
            task.params.speculative=p.speculative;
            t.assert_true("configured DFlash image requests are admitted",!server_vision_arena_request_error(task,false,&p.speculative));
            for (int change=0;change<9;++change) {
                auto bad=p;
                switch (change) {
                    case 0: bad.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V); break;
                    case 1: bad.speculative.draft.n_max=8; break;
                    case 2: bad.speculative.draft.mparams.path.clear(); break;
                    case 3: bad.kv_stream_auxiliary_layers=1; break;
                    case 4: bad.speculative.draft.cache_type_k_explicit=true; break;
                    case 5: bad.n_ubatch=depth; break;
                    case 6: bad.n_parallel=2; break;
                    case 7: bad.fit_params=true; break;
                    case 8: bad.mmproj_use_gpu=false; break;
                }
                t.assert_true(server_vision_arena_config_error(bad));
            }
            for (int change=0;change<6;++change) {
                auto bad=task.params.speculative;
                switch (change) {
                    case 0: bad.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V); break;
                    case 1: bad.draft.n_max=depth+1; break;
                    case 2: bad.draft.n_max=0; break;
                    case 3: bad.draft.mparams.path="foreign.gguf"; break;
                    case 4: bad.draft.cache_type_v_explicit=true; break;
                    case 5: bad.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP}; break;
                }
                task.params.speculative=bad;
                t.assert_true(server_vision_arena_request_error(task,false,&p.speculative));
                task.params.speculative=p.speculative;
            }
            task.params.speculative.types={COMMON_SPECULATIVE_TYPE_NONE};
            t.assert_true(!server_vision_arena_request_error(task,false,&p.speculative));
            task.params.speculative=p.speculative;
            t.assert_true(server_vision_arena_request_error(task));
        }
    });
    return t.summary();
}
