#include "testing.h"

#include "../tools/server/server-moe-routing.h"
#if defined(LLAMA_SERVER_TEST_HOOKS)
#include "../tools/server/server-context.h"
#endif
#if defined(_WIN32) && defined(LLAMA_SERVER_TEST_HOOKS)
#include "../tools/server/server-models.h"
#endif

#include <cstring>
#include <limits>

using json = common_json;

static json serialize_moe_routing_event_coverage(
        const server_moe_routing_chunk_coverage & coverage,
        bool has_routable_records) {
    json event = {
        {"event", "moe_routing_chunk"},
        {"is_final_for_trace", false},
    };
    server_moe_routing_apply_canonical_event_coverage(
        event, coverage, has_routable_records,
        "The producer retained a partial routing population:",
        "Routing rows were lost before complete routing coordinates were retained.");
    return json::parse(event.dump());
}

static json serialize_moe_routing_final_coverage(
        const server_moe_routing_chunk_coverage & coverage) {
    json event = {
        {"event", "moe_routing_chunk"},
        {"is_final_for_trace", true},
    };
    server_moe_routing_apply_canonical_event_coverage(
        event, coverage, false,
        "The request ended with partial routing coverage:",
        "Routing capture ended after an unavailable native or control-boundary interval.",
        "No routable MoE records were retained for this request.");
    return json::parse(event.dump());
}

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

static void test_canonical_event_coverage_serialization(testing & t) {
    t.test("dense routing remains a complete serialized event", [](testing & t) {
        const json event = serialize_moe_routing_event_coverage({}, true);
        t.assert_equal(0U, event.at("availability").get<uint32_t>());
        t.assert_true(!event.contains("reason"));
        t.assert_true(!event.contains("unlocated_coverage_loss"));
    });

    t.test("invalid router rows serialize partial producer coverage", [](testing & t) {
        server_moe_routing_chunk_coverage coverage;
        coverage.invalid_rows = 1;
        const json event = serialize_moe_routing_event_coverage(coverage, true);
        t.assert_equal(1U, event.at("availability").get<uint32_t>());
        t.assert_true(event.at("reason").get<std::string>().find("invalid router rows") != std::string::npos);
        t.assert_true(!event.contains("unlocated_coverage_loss"));
    });

    t.test("unavailable K plus one when K equals N serializes source loss", [](testing & t) {
        server_moe_routing_chunk_coverage coverage;
        coverage.unavailable_rows = 1;
        coverage.source_unavailable = true;
        const json event = serialize_moe_routing_event_coverage(coverage, true);
        t.assert_equal(1U, event.at("availability").get<uint32_t>());
        const std::string reason = event.at("reason").get<std::string>();
        t.assert_true(reason.find("unavailable native routing values") != std::string::npos);
        t.assert_true(reason.find("native routing source was unavailable") != std::string::npos);
    });

    t.test("uniquely unmappable rows serialize exact coordinate-free loss", [](testing & t) {
        server_moe_routing_chunk_coverage coverage;
        coverage.unlinked_rows = 1;
        coverage.unlocated_rows = 1;
        const json event = serialize_moe_routing_event_coverage(coverage, true);
        t.assert_equal(1U, event.at("availability").get<uint32_t>());
        t.assert_equal(1ULL, event.at("unlocated_coverage_loss").at("count").get<uint64_t>());
        t.assert_equal("Routing rows were lost before complete routing coordinates were retained.",
            event.at("unlocated_coverage_loss").at("reason").get<std::string>());
    });

    t.test("ambiguously unmappable rows serialize partial coverage without exact loss", [](testing & t) {
        server_moe_routing_chunk_coverage coverage;
        coverage.attribution_ambiguous = true;
        const json event = serialize_moe_routing_event_coverage(coverage, true);
        t.assert_equal(1U, event.at("availability").get<uint32_t>());
        t.assert_true(event.at("reason").get<std::string>().find("could not be attributed") != std::string::npos);
        t.assert_true(!event.contains("unlocated_coverage_loss"));
    });

    t.test("interrupted and source-unavailable final events retain distinct evidence", [](testing & t) {
        server_moe_routing_chunk_coverage interrupted;
        interrupted.interrupted = true;
        const json interrupted_event = serialize_moe_routing_final_coverage(interrupted);
        t.assert_equal(1U, interrupted_event.at("availability").get<uint32_t>());
        t.assert_true(interrupted_event.at("reason").get<std::string>().find("routing capture was interrupted") != std::string::npos);
        t.assert_true(!interrupted_event.contains("unlocated_coverage_loss"));

        server_moe_routing_chunk_coverage source_unavailable;
        source_unavailable.source_unavailable = true;
        source_unavailable.unlocated_rows = 3;
        const json unavailable_event = serialize_moe_routing_final_coverage(source_unavailable);
        t.assert_equal(1U, unavailable_event.at("availability").get<uint32_t>());
        t.assert_true(unavailable_event.at("reason").get<std::string>().find("native routing source was unavailable") != std::string::npos);
        t.assert_equal(3ULL, unavailable_event.at("unlocated_coverage_loss").at("count").get<uint64_t>());
        t.assert_equal("Routing capture ended after an unavailable native or control-boundary interval.",
            unavailable_event.at("unlocated_coverage_loss").at("reason").get<std::string>());
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

#if defined(LLAMA_SERVER_TEST_HOOKS)
static void test_saturated_native_dispatch_loss_serialization(testing & t) {
    const json gap = server_test_moe_dispatch_saturation_gap_json();
    const json chunk = server_test_moe_dispatch_saturation_chunk_json();
    t.assert_equal(17ULL, gap.at("first_sequence").get<uint64_t>());
    t.assert_equal(17ULL, gap.at("next_sequence").get<uint64_t>());
    t.assert_equal(288ULL, gap.at("first_physical_step").get<uint64_t>());
    t.assert_equal(311ULL, gap.at("next_physical_step").get<uint64_t>());
    t.assert_equal(287U, gap.at("first_physical_microbatch").get<uint32_t>());
    t.assert_equal(0U, gap.at("last_physical_microbatch").get<uint32_t>());
    t.assert_equal(101LL, gap.at("first_dispatch_monotonic_us").get<int64_t>());
    t.assert_equal(123LL, gap.at("last_dispatch_monotonic_us").get<int64_t>());
    t.assert_equal(23ULL, gap.at("physical_dispatch_count").get<uint64_t>());
    t.assert_equal(1ULL, gap.at("encode_physical_dispatch_count").get<uint64_t>());
    t.assert_equal(22ULL, gap.at("decode_physical_dispatch_count").get<uint64_t>());
    t.assert_equal("target", gap.at("physical_context").get<std::string>());
    t.assert_equal("decode", gap.at("operation").get<std::string>());
    t.assert_equal("encode", gap.at("last_operation").get<std::string>());
    t.assert_equal("mixed", gap.at("operation_state").get<std::string>());
    t.assert_true(gap.at("saturation").get<bool>());
    t.assert_equal("saturated_exact", gap.at("loss_descriptor_state").get<std::string>());
    t.assert_equal("mixed", gap.at("generation_state").get<std::string>());
    t.assert_equal("enabled", gap.at("native_moe_routing_state").get<std::string>());
    t.assert_equal(5ULL, gap.at("props_generation").get<uint64_t>());
    t.assert_equal(8ULL, gap.at("last_props_generation").get<uint64_t>());
    t.assert_equal(288ULL, gap.at("microbatch_generation").get<uint64_t>());
    t.assert_equal(310ULL, gap.at("last_microbatch_generation").get<uint64_t>());
    t.assert_equal("available", gap.at("timestamp_state").get<std::string>());
    t.assert_equal("native_dispatch_queue_overflow", gap.at("cause").get<std::string>());
    t.assert_equal(1U, chunk.at("availability").get<uint32_t>());
    t.assert_true(!chunk.contains("unlocated_coverage_loss"));
    t.assert_equal(2U, (uint32_t) chunk.at("gaps").size());
    t.assert_equal(239ULL, chunk.at("gaps").at(0).at("first_physical_step").get<uint64_t>());
    t.assert_equal(240ULL, chunk.at("gaps").at(0).at("next_physical_step").get<uint64_t>());
    t.assert_equal(288ULL, chunk.at("gaps").at(1).at("first_physical_step").get<uint64_t>());
    t.assert_equal(311ULL, chunk.at("gaps").at(1).at("next_physical_step").get<uint64_t>());
}

static void test_streamed_native_dispatch_loss_finalization(testing & t) {
    const json stream = server_test_moe_dispatch_loss_stream_finalization_json();
    const uint64_t cap = stream.at("chunk_limit_bytes").get<uint64_t>();
    const json & events = stream.at("events");
    t.assert_equal(4096ULL, cap);
    t.assert_equal(0ULL, stream.at("dropped_events").get<uint64_t>());
    t.assert_true(events.size() > 2);

    uint64_t expected_first_step = 1;
    uint64_t exact_gap_count = 0;
    uint64_t saturation_count = 0;
    uint64_t final_marker_count = 0;
    bool final_seen = false;
    for (const json & event : events) {
        t.assert_equal(3U, event.at("schema_version").get<uint32_t>());
        t.assert_equal("moe_routing_chunk", event.at("event").get<std::string>());
        t.assert_true(event.at("serialized_bytes").get<uint64_t>() <= cap);
        t.assert_equal(1U, event.at("availability").get<uint32_t>());
        t.assert_true(!event.contains("chunk_capacity_state"));
        if (event.value("is_final_for_trace", false)) {
            ++final_marker_count;
            final_seen = true;
            t.assert_true(!event.contains("gaps") || event.at("gaps").empty());
            continue;
        }
        t.assert_true(!final_seen);
        const json & gaps = event.at("gaps");
        t.assert_true(!gaps.empty());
        for (const json & gap : gaps) {
            t.assert_equal(expected_first_step, gap.at("first_physical_step").get<uint64_t>());
            const uint64_t next_step = gap.at("next_physical_step").get<uint64_t>();
            t.assert_true(next_step > expected_first_step);
            t.assert_equal("target", gap.at("physical_context").get<std::string>());
            t.assert_equal("decode", gap.at("operation").get<std::string>());
            t.assert_equal(1U, event.at("availability").get<uint32_t>());
            if (exact_gap_count < 255) {
                t.assert_equal(expected_first_step + 1, next_step);
                t.assert_equal("detailed_exact", gap.at("loss_descriptor_state").get<std::string>());
                t.assert_true(!gap.at("saturation").get<bool>());
            } else {
                t.assert_equal(256ULL, expected_first_step);
                t.assert_equal(280ULL, next_step);
                t.assert_equal(24ULL, gap.at("physical_dispatch_count").get<uint64_t>());
                t.assert_equal("saturated_exact", gap.at("loss_descriptor_state").get<std::string>());
                t.assert_true(gap.at("saturation").get<bool>());
                ++saturation_count;
            }
            ++exact_gap_count;
            expected_first_step = next_step;
        }
    }
    t.assert_equal(256ULL, exact_gap_count);
    t.assert_equal(280ULL, expected_first_step);
    t.assert_equal(1ULL, saturation_count);
    t.assert_equal(1ULL, final_marker_count);
}
#endif

#if defined(_WIN32) && defined(LLAMA_SERVER_TEST_HOOKS)
static void test_router_child_api_key_file_security(testing & t) {
    const server_child_api_key_file_security_test_result result = server_test_child_api_key_file_security();
    t.assert_true("protected DACL", result.protected_dacl);
    t.assert_true("current-user owner", result.owner_is_current_user);
    t.assert_true("current-user full control", result.current_user_full_control);
    t.assert_true("SYSTEM full control", result.system_full_control);
    t.assert_true("Administrators full control", result.administrators_full_control);
    t.assert_true("only explicit current-user/SYSTEM/Administrators ACEs", result.only_expected_explicit_allow_aces);
    t.assert_true("exclusive open rejects a shared peer", result.exclusive_open_rejects_shared_open);
    t.assert_true("cleanup", result.cleanup_succeeded);
}
#endif

int main() {
    testing t;

    t.test("record validation", test_record_validation);
    t.test("invalid records do not look capped", test_invalid_records_do_not_look_capped);
    t.test("cap truncation", test_cap_truncation);
    t.test("cap and invalid records remain distinct", test_cap_and_invalid_records_remain_distinct);
    t.test("canonical event coverage", test_canonical_event_coverage);
    t.test("canonical event coverage serialization", test_canonical_event_coverage_serialization);
    t.test("serialization loss counts pending and incoming", test_serialization_loss_counts_pending_and_incoming);
    t.test("finalization loss combines event and pending", test_finalization_loss_combines_event_and_pending);
#if defined(LLAMA_SERVER_TEST_HOOKS)
    t.test("saturated native dispatch loss serialization", test_saturated_native_dispatch_loss_serialization);
    t.test("streamed native dispatch loss finalization", test_streamed_native_dispatch_loss_finalization);
#endif
#if defined(_WIN32) && defined(LLAMA_SERVER_TEST_HOOKS)
    t.test("router child API key file security", test_router_child_api_key_file_security);
#endif

    return t.summary();
}
