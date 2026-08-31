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

static void test_canonical_event_coverage(testing & t) {
    t.test("disabled and dense routing stay available", [](testing & t) {
        const server_moe_routing_chunk_coverage coverage;
        t.assert_true(!server_moe_routing_chunk_is_partial(coverage));
        t.assert_equal(0U, server_moe_routing_chunk_availability(coverage, true));
    });

    t.test("invalid records mark the canonical event partial", [](testing & t) {
        server_moe_routing_chunk_coverage coverage;
        coverage.invalid_rows = 1;
        t.assert_true(server_moe_routing_chunk_is_partial(coverage));
        t.assert_equal(1U, server_moe_routing_chunk_availability(coverage, true));
    });

    t.test("unavailable K+1 or routing weights mark the canonical event partial", [](testing & t) {
        server_moe_routing_chunk_coverage coverage;
        coverage.unavailable_rows = 1;
        t.assert_true(server_moe_routing_chunk_is_partial(coverage));
        t.assert_equal(1U, server_moe_routing_chunk_availability(coverage, true));
    });

    t.test("unavailable K+1 when K equals N marks the canonical event partial", [](testing & t) {
        server_moe_routing_chunk_coverage coverage;
        coverage.unavailable_rows = 1;
        t.assert_true(server_moe_routing_chunk_is_partial(coverage));
        t.assert_equal(1U, server_moe_routing_chunk_availability(coverage, true));
    });

    t.test("unmappable rows mark candidate traces partial without an exact loss count", [](testing & t) {
        server_moe_routing_chunk_coverage coverage;
        coverage.attribution_ambiguous = true;
        t.assert_equal(0ULL, coverage.unlocated_rows);
        t.assert_true(server_moe_routing_chunk_is_partial(coverage));
        t.assert_equal(1U, server_moe_routing_chunk_availability(coverage, true));
    });

    t.test("interrupted and unavailable final markers stay partial", [](testing & t) {
        server_moe_routing_chunk_coverage interrupted;
        interrupted.interrupted = true;
        t.assert_equal(1U, server_moe_routing_chunk_availability(interrupted, false));

        server_moe_routing_chunk_coverage source_unavailable;
        source_unavailable.source_unavailable = true;
        t.assert_equal(1U, server_moe_routing_chunk_availability(source_unavailable, false));
    });
}

static void test_serialization_loss_counts_pending_and_incoming(testing & t) {
    uint64_t lost_population = 3;
    t.assert_true(server_moe_routing_add_lost_population(7, lost_population));
    t.assert_equal(10ULL, lost_population);
    t.assert_true(!server_moe_routing_add_lost_population(std::numeric_limits<uint64_t>::max(), lost_population));
    t.assert_equal(10ULL, lost_population);
}

static void test_finalization_loss_combines_event_and_pending(testing & t) {
    uint64_t unlocated_rows = 0;
    t.assert_true(server_moe_routing_combine_lost_population(11, 7, unlocated_rows));
    t.assert_equal(18ULL, unlocated_rows);

    unlocated_rows = 18;
    t.assert_true(!server_moe_routing_combine_lost_population(11, std::numeric_limits<uint64_t>::max(), unlocated_rows));
    t.assert_equal(18ULL, unlocated_rows);
}

int main() {
    testing t;

    t.test("record validation", test_record_validation);
    t.test("invalid records do not look capped", test_invalid_records_do_not_look_capped);
    t.test("cap truncation", test_cap_truncation);
    t.test("cap and invalid records remain distinct", test_cap_and_invalid_records_remain_distinct);
    t.test("canonical event coverage", test_canonical_event_coverage);
    t.test("serialization loss counts pending and incoming", test_serialization_loss_counts_pending_and_incoming);
    t.test("finalization loss combines event and pending", test_finalization_loss_combines_event_and_pending);

    return t.summary();
}
