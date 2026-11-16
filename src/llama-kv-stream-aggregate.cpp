#include "llama-kv-stream-aggregate.h"
#include "llama-cparams.h"
#include "llama-hparams.h"
#include <algorithm>
#include <climits>
#include <new>
#include <stdexcept>

bool llama_kv_stream_aggregate_requirements_make(const std::vector<uint32_t> & roles,
        const std::vector<size_t> & tokens,llama_kv_stream_aggregate_requirements & output) {
    if (roles.empty() || roles.size()>LLAMA_MAX_LAYERS || tokens.empty() || tokens.size()%roles.size() ||
            tokens.size()/roles.size()>LLAMA_MAX_SEQ) return false;
    uint32_t layers=0;
    for (auto count:roles) {
        if (!count || count>LLAMA_MAX_LAYERS-layers) return false;
        layers+=count;
    }
    try {
        llama_kv_stream_aggregate_requirements next;
        next.owner_pages.resize(tokens.size());next.owner_first.resize(tokens.size());next.role_tokens.resize(roles.size());
        const size_t maximum=size_t(INT32_MAX)/256;
        size_t active_pages=0;
        for (size_t request=0;request<tokens.size()/roles.size();++request) {
            size_t largest=0;
            for (size_t role=0;role<roles.size();++role) {
                const size_t owner=request*roles.size()+role;
                if (tokens[owner]>size_t(INT32_MAX)-255) return false;
                const size_t pages=tokens[owner]/256+(tokens[owner]%256!=0);
                next.owner_pages[owner]=pages;largest=std::max(largest,pages);
            }
            if (largest>maximum-active_pages) return false;
            active_pages+=largest;
        }
        next.active_tokens=active_pages*256;
        for (size_t role=0;role<roles.size();++role) {
            size_t pages=0;
            for (size_t owner=role;owner<tokens.size();owner+=roles.size()) {
                next.owner_first[owner]=pages*256;
                if (next.owner_pages[owner]>maximum-pages) return false;
                pages+=next.owner_pages[owner];
            }
            next.role_tokens[role]=pages*256;
        }
        output=std::move(next);return true;
    } catch (const std::bad_alloc &) {return false;}
      catch (const std::length_error &) {return false;}
}

// Keep preferred roles resident by redistributing the same scalar resident quota.
static void prefer_resident_roles(const llama_kv_stream_policy_config & config,const llama_kv_stream_policy_state & state,
        const std::vector<uint32_t> & roles,const std::vector<uint8_t> & preferred,
        const llama_kv_stream_aggregate_requirements & requirements,llama_kv_stream_policy_layout & layout) {
    if (preferred.empty() || !state.decode_active_pages) return;
    const auto active=uint32_t(requirements.active_tokens/256);
    const uint32_t minimum=active>state.ring_slots ? active-state.ring_slots : 0;
    std::vector<uint32_t> capacities,room(layout.layers.size());
    std::vector<uint8_t> protected_layers(layout.layers.size());
    for (const auto & layer:layout.layers) capacities.push_back(layer.capacity_pages);
    size_t global=0;uint64_t need=0,available=0;
    for (size_t role=0;role<roles.size();++role) for (uint32_t layer=0;layer<roles[role];++layer,++global) {
        protected_layers[global]=preferred[role] && requirements.role_tokens[role];
        if (protected_layers[global] && capacities[global]<active) {
            need+=active-capacities[global];capacities[global]=active;
        } else if (!protected_layers[global] && capacities[global]>minimum) {
            room[global]=capacities[global]-minimum;available+=room[global];
        }
    }
    if (!need || need>available) return;
    uint64_t assigned=0;std::vector<std::pair<uint64_t,size_t>> remainder;
    for (size_t layer=0;layer<capacities.size();++layer) if (room[layer]) {
        const uint64_t product=need*room[layer],take=product/available;
        capacities[layer]-=uint32_t(take);assigned+=take;
        remainder.emplace_back(product%available,layer);
    }
    std::sort(remainder.begin(),remainder.end(),[](const auto & a,const auto & b) {return a.first!=b.first ? a.first>b.first : a.second<b.second;});
    for (size_t i=0;i<need-assigned;++i) --capacities[remainder[i].second];
    auto layers=layout.layers;size_t offset=layout.ring.bytes;
    for (size_t layer=0;layer<layers.size();++layer) {
        auto & entry=layers[layer];entry.capacity_pages=capacities[layer];
        entry.resident_live_pages=std::min(active,entry.capacity_pages);
        entry.streamed_pages=active-entry.resident_live_pages;
        entry.waves=entry.streamed_pages/state.ring_slots+(entry.streamed_pages%state.ring_slots!=0);
        entry.offset=offset;
        if (ggml_kv_stream_layout_make(config.shape,size_t(entry.capacity_pages)*256,entry.planes).status!=ggml_kv_stream_status::success ||
                offset>layout.conversion_offset || entry.planes.bytes>layout.conversion_offset-offset) return;
        offset+=entry.planes.bytes;
    }
    if (offset==layout.conversion_offset) layout.layers=std::move(layers);
}

bool llama_kv_stream_aggregate_layout_make(const llama_kv_stream_policy_config & config,
        const llama_kv_stream_policy_state & state,const std::vector<uint32_t> & roles,
        const std::vector<size_t> & tokens,llama_kv_stream_aggregate_layout & output,
        const std::vector<uint32_t> & resident_roles) {
    try {
        llama_kv_stream_aggregate_layout next;
        if (!llama_kv_stream_aggregate_requirements_make(roles,tokens,next.requirements) ||
                !next.requirements.active_tokens || config.shape.page_tokens!=256 ||
                (!config.caches.empty() && config.caches.size()!=roles.size())) return false;
        size_t layers=0;
        for (size_t role=0;role<roles.size();++role) {
            if (!config.caches.empty() && config.caches[role].layers!=roles[role]) return false;
            layers+=roles[role];
        }
        if (layers!=config.layers || llama_kv_stream_policy_layout_make(config,state,next.requirements.active_tokens,next.storage).status!=
                llama_kv_stream_policy_status::success) return false;
        if (!resident_roles.empty()) {
            std::vector<uint8_t> preferred(roles.size());
            for (auto role:resident_roles) {
                if (role>=roles.size() || preferred[role]) return false;
                preferred[role]=true;
            }
            prefer_resident_roles(config,state,roles,preferred,next.requirements,next.storage);
        }
        next.owners.resize(tokens.size());
        for (size_t owner=0;owner<tokens.size();++owner) next.owners[owner].resize(roles[owner%roles.size()]);
        const size_t requests=tokens.size()/roles.size();size_t global=0;
        for (size_t role=0;role<roles.size();++role) for (size_t layer=0;layer<roles[role];++layer,++global) {
            const auto & whole=next.storage.layers[global];size_t first=0;
            const size_t live=next.requirements.role_tokens[role]/256;
            std::vector<size_t> capacities(requests);
            if (whole.capacity_pages<live) {
                size_t assigned=0;std::vector<std::pair<uint64_t,size_t>> remainder;
                for (size_t request=0;request<requests;++request) {
                    const uint64_t product=uint64_t(whole.capacity_pages)*next.requirements.owner_pages[request*roles.size()+role];
                    capacities[request]=size_t(product/live);assigned+=capacities[request];
                    remainder.emplace_back(product%live,request);
                }
                std::sort(remainder.begin(),remainder.end(),[](const auto & a,const auto & b) {return a.first!=b.first ? a.first>b.first : a.second<b.second;});
                for (size_t i=0;i<whole.capacity_pages-assigned;++i) ++capacities[remainder[i].second];
            } else {
                for (size_t request=0;request<requests;++request) capacities[request]=next.requirements.owner_pages[request*roles.size()+role];
                capacities.back()+=whole.capacity_pages-live;
            }
            for (size_t request=0;request<requests;++request) {
                const size_t owner=request*roles.size()+role;
                const size_t capacity=capacities[request];
                auto & part=next.owners[owner][layer];part=whole;
                part.cache_layer=uint32_t(layer);part.capacity_pages=uint32_t(capacity);
                part.resident_live_pages=uint32_t(std::min(capacity,next.requirements.owner_pages[owner]));
                part.streamed_pages=uint32_t(next.requirements.owner_pages[owner]-part.resident_live_pages);
                part.waves=part.streamed_pages/state.ring_slots+(part.streamed_pages%state.ring_slots!=0);
                const size_t k_first=first*256*whole.planes.k_token_bytes;
                const size_t v_first=first*256*whole.planes.v_token_bytes;
                part.offset=whole.offset+k_first;
                part.planes.tokens=capacity*256;part.planes.pages=capacity;part.planes.tail_tokens=capacity ? 256 : 0;
                part.planes.k_bytes=capacity*256*whole.planes.k_token_bytes;
                part.planes.v_bytes=capacity*256*whole.planes.v_token_bytes;
                part.planes.v_offset=whole.planes.v_offset+v_first-k_first;
                part.planes.bytes=capacity ? part.planes.v_offset+part.planes.v_bytes : 0;
                first+=capacity;
            }
        }
        output=std::move(next);return true;
    } catch (const std::bad_alloc &) {return false;}
      catch (const std::length_error &) {return false;}
}

bool llama_kv_stream_aggregate_layout_make(const llama_kv_stream_policy_config & config,
        const llama_kv_stream_policy_state & state,const std::vector<uint32_t> & roles,
        const std::vector<size_t> & tokens,llama_kv_stream_aggregate_layout & output) {
    return llama_kv_stream_aggregate_layout_make(config,state,roles,tokens,output,{});
}
