#include "../tools/server/server-queue.h"
#include "testing.h"

#include <chrono>
#include <future>
#include <thread>

static server_task_result_ptr result(int id) {
    struct response : server_task_result {
        json to_json() override { return json{{"id",id}}; }
    };
    auto value=std::make_unique<response>();value->id=id;return value;
}

int main() {
    testing t;
    t.test("selection_can_observe_a_declined_mutation_without_consuming_it",[](testing & t) {
        server_queue queue;
        std::promise<void> initial_read,after_cancel;
        auto first=initial_read.get_future(),second=after_cancel.get_future();
        bool clean_before=false,mutation_after=false,stopped=false;
        size_t cancellations=0,reads=0;
        queue.on_new_task([&](server_task && task,bool yielding) {
            if (yielding && task.type==SERVER_TASK_TYPE_SLOT_GET) {
                if (++reads==1) initial_read.set_value();else after_cancel.set_value();
                return true;
            }
            if (yielding) return false;
            if (task.type==SERVER_TASK_TYPE_CANCEL) {++cancellations;queue.terminate();}
            return true;
        });
        const auto post=[&](server_task_type type) {
            server_task task(type);task.id=queue.get_new_id();queue.post(std::move(task));
        };
        queue.on_update_slots([&] {
            queue.yield_to_queue([&] {
                post(SERVER_TASK_TYPE_NEXT_RESPONSE);post(SERVER_TASK_TYPE_SLOT_GET);
                first.wait();clean_before=!queue.has_pending_mutation();
                post(SERVER_TASK_TYPE_CANCEL);post(SERVER_TASK_TYPE_SLOT_GET);
                second.wait();mutation_after=queue.has_pending_mutation();
            });
        });
        post(SERVER_TASK_TYPE_NEXT_RESPONSE);
        queue.start_loop();stopped=queue.has_pending_mutation();
        t.assert_true(clean_before && mutation_after && stopped);
        t.assert_equal(size_t(2),reads);t.assert_equal(size_t(1),cancellations);
    });
    t.test("responses_keep_their_request_identity",[](testing & t) {
        server_response queue;
        queue.add_waiting_task_ids({1,2});
        queue.send(result(2));queue.send(result(1));
        auto own=queue.recv_with_timeout({1},0);
        if (!t.assert_true(own!=nullptr)) return;
        t.assert_equal(1,own->id);
        t.assert_true(queue.recv_with_timeout({1},0)==nullptr);
        auto peer=queue.recv_with_timeout({2},0);
        if (!t.assert_true(peer!=nullptr)) return;
        t.assert_equal(2,peer->id);
    });
    t.test("unrelated_responses_cannot_extend_a_polling_deadline",[](testing & t) {
        server_response queue;queue.add_waiting_task_ids({1,2});
        std::promise<void> started;auto ready=started.get_future();
        auto receiver=std::async(std::launch::async,[&] {
            const auto first=std::chrono::steady_clock::now();started.set_value();
            auto own=queue.recv_with_timeout({1},1);
            return std::make_pair(own==nullptr,std::chrono::steady_clock::now()-first);
        });
        ready.wait();
        const auto guard=std::chrono::steady_clock::now()+std::chrono::seconds(3);
        size_t sent=0;
        while (receiver.wait_for(std::chrono::milliseconds(0))!=std::future_status::ready &&
                std::chrono::steady_clock::now()<guard) {
            queue.send(result(2));++sent;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        const bool expired_while_peer_active=receiver.wait_for(std::chrono::milliseconds(0))==std::future_status::ready;
        const auto received=receiver.get();
        t.assert_true(sent>=2);
        t.assert_true(received.first);
        t.assert_true(received.second>=std::chrono::milliseconds(700));
        t.assert_true("another request's traffic cannot postpone heartbeat/disconnect polling",expired_while_peer_active);
    });
    t.test("an_own_response_before_the_deadline_is_not_lost",[](testing & t) {
        server_response queue;queue.add_waiting_task_ids({1,2});
        auto producer=std::async(std::launch::async,[&] {
            for (int i=0;i<5;++i) {
                queue.send(result(2));std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            queue.send(result(1));
        });
        auto own=queue.recv_with_timeout({1},1);producer.get();
        if (!t.assert_true(own!=nullptr)) return;
        t.assert_equal(1,own->id);
        for (int i=0;i<5;++i) {
            auto peer=queue.recv_with_timeout({2},0);
            if (!t.assert_true(peer!=nullptr)) return;
            t.assert_equal(2,peer->id);
        }
        t.assert_true(queue.recv_with_timeout({2},0)==nullptr);
    });
    return t.summary();
}
