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
