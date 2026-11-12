#pragma once

static void test_dflash_k4v_routes(testing & t,llama_model * model,const char * path,const char * target_reference,
        const char * draft_reference,bool target_export,bool streamed) {
    if (!t.assert_true(model && model->arch==LLM_ARCH_QWEN35 && path && target_reference)) return;
    const llama_model_tensor_buft_override cpu[]={{".*",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    auto mp=llama_model_default_params();mp.n_gpu_layers=999;if (target_export) mp.tensor_buft_overrides=cpu;
    model_ptr draft(llama_model_load_from_file(path,mp),llama_model_free);
    if (!t.assert_true(draft && draft->arch==LLM_ARCH_DFLASH)) return;
    std::array<llama_tokens,2> histories,proposals;
    std::array<int32_t,24> tokens{},positions{},sequences{};
    for (llama_seq_id slot=0;slot<2;++slot) {
        histories[slot]={99,10,20};for (int32_t i=0;i<7;++i) histories[slot].push_back(30+slot*10+i);
        histories[slot].insert(histories[slot].end(),{98,10});
        for (size_t row=0;row<12;++row) {const auto index=size_t(slot)*12+row;tokens[index]=histories[slot][row];positions[index]=int32_t(row);sequences[index]=slot;}
    }
    auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=256;cp.n_seq_max=2;
    cp.kv_unified=true;cp.swa_full=false;cp.n_rs_seq=7;cp.n_outputs_max=32;cp.n_outputs_max_per_seq=16;
    cp.n_threads=cp.n_threads_batch=4;cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q4_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto profile=native_dflash_stage_profile(*model,*draft);profile[0]=13;profile[5]=profile[6]=12;
    kv_stream_bit_oracle prefix;std::unique_ptr<FILE,kv_stream_exact_file_free> prefix_file;
    if (!target_export) {
        prefix_file.reset(std::fopen(target_reference,"rb"));if (!t.assert_true(prefix_file!=nullptr)) return;
        prefix.load_control(prefix_file.get(),profile);prefix.compare();
    }
    std::unique_ptr<llama_context> native,child;std::unique_ptr<llama_context_group> group;
    llama_context * target=nullptr,* decoder=nullptr;
    auto input=llama_batch_init(24,0,1);
    struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{input};input.n_tokens=24;
    for (size_t row=0;row<24;++row) {
        input.token[row]=tokens[row];input.pos[row]=positions[row];input.n_seq_id[row]=1;input.seq_id[row][0]=sequences[row];input.logits[row]=true;
    }
    if (streamed) {
        cp.shared_device_memory_bytes=size_t(1024)*1024*1024;
        group=llama_context_group::create(*model,cp);if (!t.assert_true(group!=nullptr)) return;
        auto dp=cp;dp.shared_device_memory_bytes=0;dp.n_rs_seq=0;
        if (!t.assert_true(group->prepare_dflash(*draft,dp,4))) return;
        llama_kv_stream_selection selected;
        if (!t.assert_true(llama_kv_stream_select({512,2},{{0,0},{1,0}},selected)==llama_kv_stream_selection_status::success &&
                group->activate(selected,group->source_handles(),input,llama_memory_work_phase::target_prefill,1,1))) return;
        target=group->context();decoder=group->auxiliary_context();
    } else {
        native=target_export ? std::make_unique<llama_context>(*model,cp) :
            std::make_unique<llama_context>(*model,cp,llama_context_mode::measurement);target=native.get();
        if (!target_export) {auto dp=cp;dp.n_rs_seq=0;dp.ctx_other=target;child=std::make_unique<llama_context>(*draft,dp);decoder=child.get();}
    }
    if (target_export || streamed) {
        for (auto layer : draft->target_layer_ids) target->set_embeddings_layer_inp(uint32_t(layer),true);
        if (!t.assert_equal(0,target->decode(input))) return;
        target->synchronize();prefix.bytes("tokens",tokens.data(),sizeof(tokens));prefix.bytes("positions",positions.data(),sizeof(positions));prefix.bytes("sequences",sequences.data(),sizeof(sequences));
    } else {
        std::array<int32_t,24> actual{};
        prefix.input("tokens",actual.data(),sizeof(actual));if (!t.assert_true(actual==tokens)) return;
        prefix.input("positions",actual.data(),sizeof(actual));if (!t.assert_true(actual==positions)) return;
        prefix.input("sequences",actual.data(),sizeof(actual));if (!t.assert_true(actual==sequences)) return;
    }
    const auto borrowed=native_dflash_borrowed_profile(*model);
    prefix.bytes("borrowed_weights",borrowed.data(),sizeof(borrowed));
    std::map<int32_t,std::vector<float>> features;
    for (auto layer : draft->target_layer_ids) {
        auto & values=features[layer];values.resize(size_t(24)*model->hparams.n_embd);
        if (target_export || streamed) {
            const auto * data=target->get_embeddings_layer_inp(uint32_t(layer));if (!t.assert_true(data!=nullptr)) return;
            std::copy(data,data+values.size(),values.begin());prefix.bytes("features/"+std::to_string(layer),data,values.size()*sizeof(float));
        } else prefix.input("features/"+std::to_string(layer),values.data(),values.size()*sizeof(float));
        if (!t.assert_true(std::all_of(values.begin(),values.end(),[](float x) {return std::isfinite(x);}))) return;
    }
    for (int32_t row=0;row<24;++row) {
        std::vector<float> values(model->vocab.n_tokens());
        if (target_export || streamed) {
            const auto * data=target->get_logits_ith(row);if (!t.assert_true(data!=nullptr)) return;
            prefix.bytes("logits/"+std::to_string(row),data,values.size()*sizeof(float));
        } else prefix.input("logits/"+std::to_string(row),values.data(),values.size()*sizeof(float));
    }
    common_speculative * driver_for_verify=nullptr;
    const char * verify_path=std::getenv("LLAMA_TEST_DFLASH_K4_VERIFY_ORACLE");
    const auto verify_copies=[&](llama_context * context,const std::array<llama_tokens,2> & copies,bool compare) {
        common_batch verify(context);
        for (llama_seq_id slot=0;slot<2;++slot) {
            if (!t.assert_equal(size_t(7),copies[slot].size())) return false;
            verify.add(20,12,slot,true);
            for (size_t i=0;i<copies[slot].size();++i) verify.add(copies[slot][i],llama_pos(13+i),slot,true);
        }
        if (compare) llama_set_kv_stream_decode(context,true);
        if (!t.assert_equal(0,llama_process(context,LLAMA_PROCESS_TYPE_DECODE,verify.get()))) return false;
        context->synchronize();kv_stream_bit_oracle exact;auto shape=profile;shape[0]=15;
        std::unique_ptr<FILE,kv_stream_exact_file_free> file;
        if (compare) {
            file.reset(std::fopen(verify_path,"rb"));if (!t.assert_true(file!=nullptr)) return false;
            exact.load_control(file.get(),shape);exact.compare();
        }
        for (auto layer : draft->target_layer_ids) {
            const auto * values=context->get_embeddings_layer_inp(uint32_t(layer));
            if (!t.assert_true(values && std::all_of(values,values+size_t(16)*model->hparams.n_embd,[](float x) {return std::isfinite(x);}))) return false;
            exact.bytes("features/"+std::to_string(layer),values,size_t(16)*model->hparams.n_embd*sizeof(float));
        }
        for (int32_t row=0;row<16;++row) {
            const auto * values=context->get_logits_ith(row);
            if (!t.assert_true(values && std::all_of(values,values+model->vocab.n_tokens(),[](float x) {return std::isfinite(x);}))) return false;
            exact.bytes("logits/"+std::to_string(row),values,size_t(model->vocab.n_tokens())*sizeof(float));
        }
        if (compare) {
            exact.finish();
            const auto before=common_speculative_get_stats(driver_for_verify);
            if (!t.assert_true(common_speculative_process(driver_for_verify,verify))) return false;
            decoder->synchronize();const auto after=common_speculative_get_stats(driver_for_verify);
            if (!t.assert_equal(before.size(),after.size())) return false;
            for (size_t i=0;i<before.size();++i)
                if (!t.assert_true(before[i].n_predict==after[i].n_predict && before[i].n_gen_tokens==after[i].n_gen_tokens)) return false;
            for (llama_seq_id slot=0;slot<2;++slot)
                if (!t.assert_equal(llama_pos(19),llama_memory_seq_pos_max(decoder->get_memory(),slot))) return false;
        } else {
            file.reset(std::fopen(verify_path,"wb"));if (!t.assert_true(file!=nullptr)) return false;
            exact.save_control(file.get(),shape);
        }
        t.out<<"K4V copied target verification records="<<exact.records()<<" bytes="<<exact.byte_count()<<" rows=8,8\n";
        return true;
    };
    if (target_export) {
        prefix_file.reset(std::fopen(target_reference,"wb"));if (!t.assert_true(prefix_file!=nullptr)) return;
        prefix.save_control(prefix_file.get(),profile);t.out<<"native K4V history records="<<prefix.records()<<" bytes="<<prefix.byte_count()<<'\n';
        if (verify_path) {
            std::array<llama_tokens,2> copies;
            for (llama_seq_id slot=0;slot<2;++slot) copies[slot].assign(histories[slot].begin()+3,histories[slot].begin()+10);
            verify_copies(target,copies,false);
        }
        return;
    }
    prefix.finish();common_params_speculative params;params.types={COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
    params.ngram_map_k4v.size_n=2;params.ngram_map_k4v.size_m=7;params.ngram_map_k4v.min_hits=1;
    params.draft.ctx_tgt=target;params.draft.ctx_dft=decoder;params.draft.n_max=3;params.draft.n_min=0;params.draft.p_min=0;
    common_speculative_ptr driver(common_speculative_init(params,2));if (!t.assert_true(driver!=nullptr)) return;
    driver_for_verify=driver.get();
    for (llama_seq_id slot=0;slot<2;++slot) {
        std::vector<float> gathered(size_t(12)*draft->hparams.n_embd_inp_enc());common_batch injection(decoder);
        for (size_t row=0;row<12;++row) {
            for (size_t layer=0;layer<draft->target_layer_ids.size();++layer)
                std::memcpy(gathered.data()+row*draft->hparams.n_embd_inp_enc()+layer*model->hparams.n_embd,
                    features.at(draft->target_layer_ids[layer]).data()+(size_t(slot)*12+row)*model->hparams.n_embd,size_t(model->hparams.n_embd)*sizeof(float));
            const llama_pos p=llama_pos(row),position[4]={p,p,p,0};
            injection.add_embd({gathered.data()+row*draft->hparams.n_embd_inp_enc(),1,draft->hparams.n_embd_inp_enc()},position,slot,false);
        }
        if (!t.assert_equal(0,llama_process(decoder,LLAMA_PROCESS_TYPE_DECODE,injection.get()))) return;
        common_speculative_begin(driver.get(),slot,histories[slot],{0,1});
    }
    auto draft_profile=profile;draft_profile[0]=14;kv_stream_bit_oracle output;std::unique_ptr<FILE,kv_stream_exact_file_free> file;
    if (streamed) {file.reset(std::fopen(draft_reference,"rb"));if (!t.assert_true(file!=nullptr)) return;output.load_control(file.get(),draft_profile);output.compare();}
    for (int32_t round=0;round<4;++round) {
        if (round==3) {
            if (streamed) {
                if (!t.assert_true(common_speculative_rebind(driver.get(),group->metadata_context(),group->auxiliary_metadata_context()) && group->park())) return;
                llama_kv_stream_selection selected;auto anchor=llama_batch_init(2,0,1);
                struct release_batch {llama_batch & batch;~release_batch() {llama_batch_free(batch);}} release{anchor};anchor.n_tokens=2;
                for (llama_seq_id slot=0;slot<2;++slot) {anchor.token[slot]=20;anchor.pos[slot]=12;anchor.n_seq_id[slot]=1;anchor.seq_id[slot][0]=slot;anchor.logits[slot]=true;}
                if (!t.assert_true(llama_kv_stream_select({512,2},{{0,12},{1,12}},selected)==llama_kv_stream_selection_status::success &&
                        group->activate(selected,group->source_handles(),anchor,llama_memory_work_phase::target_verify,1,2))) return;
                target=group->context();decoder=group->auxiliary_context();
                if (!t.assert_true(common_speculative_rebind(driver.get(),target,decoder))) return;
            } else {
                std::array<image_writer,2> states;
                for (llama_seq_id slot=0;slot<2;++slot) decoder->get_memory()->state_write(states[slot],slot,0);
                decoder->get_memory()->clear(true);
                for (llama_seq_id slot=0;slot<2;++slot) {
                    image_reader restored(states[slot].bytes);decoder->get_memory()->state_read(restored,slot,0);restored.flush_tensor_reads();
                    if (!t.assert_equal(states[slot].bytes.size(),restored.n_bytes())) return;
                }
            }
        }
        if (round==2 && !t.assert_true(llama_memory_seq_rm(decoder->get_memory(),1,12,-1))) return;
        for (llama_seq_id slot=0;slot<2;++slot) {
            proposals[slot].clear();auto & request=common_speculative_get_draft_params(driver.get(),slot);
            request.drafting=true;request.n_max=7;request.pos0=12;request.id_last=(round==1 && slot==1) ? 5000 : (round==2 && slot==0) ? 6000 : 20;
            request.prompt=&histories[slot];request.result=&proposals[slot];request.ngram_context={0,1};
        }
        common_speculative_draft(driver.get());decoder->synchronize();const auto stats=common_speculative_get_stats(driver.get());
        if (!t.assert_true(stats.size()==2 && stats[1].type==COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH && stats[1].n_predict==size_t(std::min(round,2)))) return;
        const auto name="round/"+std::to_string(round);
        for (llama_seq_id slot=0;slot<2;++slot) {
            const bool copied=round==0 || round==3 || (round==1 && slot==0) || (round==2 && slot==1);const auto type=common_speculative_get_last_type(driver.get(),slot);
            if (!t.assert_equal(copied ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,type) ||
                    !t.assert_equal(copied ? size_t(7) : size_t(3),proposals[slot].size())) return;
            if (copied) for (size_t i=0;i<7;++i) if (!t.assert_equal(llama_token(30+slot*10+i),proposals[slot][i])) return;
            const uint32_t source=uint32_t(type);output.bytes(name+"/source/"+std::to_string(slot),&source,sizeof(source));
            output.bytes(name+"/proposals/"+std::to_string(slot),proposals[slot].data(),proposals[slot].size()*sizeof(llama_token));
            const size_t rows=round==3 && slot==0 ? 16 : copied ? 12 : 16;
            kv_stream_exact_kv(output,name+"/KV/"+std::to_string(slot),decoder->get_kv_resources().bounded,0,rows,slot);
        }
        if (round==0 && verify_path) {
            if (!t.assert_true(streamed)) return;
            verify_copies(target,proposals,true);
            return;
        }
        if (round==1 || round==2) {
            const auto canonical=dflash_exact_lattice(decoder->get_embeddings_nextn(),4,draft->hparams.n_embd_out(),draft->hparams.dflash_selector_top_k,model->vocab.n_tokens());
            output.bytes(name+"/selector",canonical.data(),canonical.size()*sizeof(uint32_t));
        }
    }
    if (streamed) output.finish();
    else {file.reset(std::fopen(draft_reference,"wb"));if (!t.assert_true(file!=nullptr)) return;output.save_control(file.get(),draft_profile);}
    t.out<<"genuine K4V mixed routes records="<<output.records()<<" bytes="<<output.byte_count()<<" key=2 copy=7 DF=3 forwards=2\n";
}
