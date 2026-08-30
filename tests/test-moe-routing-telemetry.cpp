#include "testing.h"

#include "../tools/server/server-moe-routing.h"

#include <cstring>
#include <limits>

static void test_record_validation(testing & t) {
    t.test("valid assignment", [](testing & t) {
        t.assert_true(server_moe_routing_assignment_is_valid(2, 3, 4, 4));
        t.assert_true(server_moe_routing_weight_is_usable(0.25f));
    });

    t.test("non-finite weight remains unavailable", [](testing & t) {
        t.assert_true(!server_moe_routing_weight_is_usable(std::numeric_limits<float>::quiet_NaN()));
        t.assert_true(!server_moe_routing_weight_is_usable(std::numeric_limits<float>::infinity()));
        t.assert_true(!server_moe_routing_weight_is_usable(-0.25f));
    });

    t.test("invalid expert and layer", [](testing & t) {
        t.assert_true(!server_moe_routing_assignment_is_valid(2, 4, 4, 4));
        t.assert_true(!server_moe_routing_assignment_is_valid(4, 3, 4, 4));
    });
}

static void test_invalid_records_do_not_look_capped(testing & t) {
    server_moe_routing_capture_counts counts;

    t.assert_equal(SERVER_MOE_ROUTING_CAPTURED, server_moe_routing_capture(
        counts, server_moe_routing_assignment_is_valid(2, 3, 4, 4), 2, 4));
    t.assert_equal(SERVER_MOE_ROUTING_INVALID, server_moe_routing_capture(
        counts, server_moe_routing_assignment_is_valid(2, 4, 4, 4), 1, 4));
    t.assert_equal(SERVER_MOE_ROUTING_INVALID, server_moe_routing_capture(
        counts, server_moe_routing_assignment_is_valid(4, 3, 4, 4), 1, 4));

    t.assert_equal(4ULL, counts.total);
    t.assert_equal(2ULL, counts.captured);
    t.assert_equal(2ULL, counts.invalid);
    t.assert_equal(0ULL, counts.cap_dropped);
    t.assert_true(!server_moe_routing_was_truncated(counts));
    t.assert_true(std::strcmp("available", server_moe_routing_capture_state(counts, true)) == 0);
}

static void test_cap_truncation(testing & t) {
    server_moe_routing_capture_counts counts;

    t.assert_equal(SERVER_MOE_ROUTING_CAPTURED, server_moe_routing_capture(counts, true, 2, 4));
    t.assert_equal(SERVER_MOE_ROUTING_CAP_DROPPED, server_moe_routing_capture(counts, true, 3, 4));

    t.assert_equal(5ULL, counts.total);
    t.assert_equal(2ULL, counts.captured);
    t.assert_equal(0ULL, counts.invalid);
    t.assert_equal(3ULL, counts.cap_dropped);
    t.assert_true(server_moe_routing_was_truncated(counts));
    t.assert_true(std::strcmp("truncated", server_moe_routing_capture_state(counts, true)) == 0);
}

static void test_cap_and_invalid_records_remain_distinct(testing & t) {
    server_moe_routing_capture_counts counts;

    t.assert_equal(SERVER_MOE_ROUTING_CAPTURED, server_moe_routing_capture(counts, true, 2, 4));
    t.assert_equal(SERVER_MOE_ROUTING_INVALID, server_moe_routing_capture(counts, false, 1, 4));
    t.assert_equal(SERVER_MOE_ROUTING_CAP_DROPPED, server_moe_routing_capture(counts, true, 3, 4));

    t.assert_equal(6ULL, counts.total);
    t.assert_equal(2ULL, counts.captured);
    t.assert_equal(1ULL, counts.invalid);
    t.assert_equal(3ULL, counts.cap_dropped);
    t.assert_true(server_moe_routing_was_truncated(counts));
    t.assert_true(std::strcmp("truncated", server_moe_routing_capture_state(counts, true)) == 0);
}

int main() {
    testing t;

    t.test("record validation", test_record_validation);
    t.test("invalid records do not look capped", test_invalid_records_do_not_look_capped);
    t.test("cap truncation", test_cap_truncation);
    t.test("cap and invalid records remain distinct", test_cap_and_invalid_records_remain_distinct);

    return t.summary();
}
