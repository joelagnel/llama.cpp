#include "server-task.h"
#include "testing.h"

#include <cmath>

static completion_token_output output_with_logprob(double logprob) {
    completion_token_output output = {};
    output.tok = 1;
    output.prob = (float) std::exp(logprob);
    output.logprob = logprob;
    output.text_to_send = "x";
    return output;
}

int main() {
    testing test;

    test.test("completion log probability precision", [&](testing & t) {
        const double below_float_range = -200.125;
        const completion_token_output underflowed = output_with_logprob(below_float_range);
        const json underflowed_json = completion_token_output::probs_vector_to_json({underflowed}, false);

        t.assert_true(std::exp(below_float_range) > 0.0);
        t.assert_equal(0.0f, underflowed.prob);
        t.assert_true(underflowed_json.at(0).contains("logprob"));
        t.assert_true(!underflowed_json.at(0).contains("prob"));
        t.assert_equal(below_float_range, underflowed_json.at(0).at("logprob").get<double>());

        const double normal = -1.5;
        const completion_token_output normal_output = output_with_logprob(normal);
        const json normal_json = completion_token_output::probs_vector_to_json({normal_output}, false);
        const json post_sampling_json = completion_token_output::probs_vector_to_json({normal_output}, true);

        t.assert_equal(normal, normal_json.at(0).at("logprob").get<double>());
        t.assert_true(post_sampling_json.at(0).contains("prob"));
        t.assert_true(!post_sampling_json.at(0).contains("logprob"));
        t.assert_equal((double) normal_output.prob, post_sampling_json.at(0).at("prob").get<double>());
    });

    return test.summary();
}
