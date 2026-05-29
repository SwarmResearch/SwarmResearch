// Naive speculative decoding — bare-bones Leviathan/Chen 2022 reference impl.
// arXiv: 2211.17192  https://arxiv.org/abs/2211.17192
//
// PERSISTENT-PROCESS VERSION: loads target+draft once, then loops over prompts
// read from stdin. Each prompt is delimited by "\n<<END_PROMPT>>\n". Per-prompt
// stats and TOKENS line are emitted between "=== PROMPT i BEGIN ===" and
// "=== PROMPT i END ===" markers so the harness can split the stream.
//
// Invocation (no -p; prompts come on stdin):
//   ./llama-naive-spec -m TARGET -md DRAFT -ngl 99 -ngld 99 -c 4096 -n 128 \
//                      --draft-min 8 --draft-max 8 --draft-p-min 0.0 --temp 0
//                      < prompts.delim
//
// Algorithm (greedy / temperature 0, single-chain, lossless):
//   1. Draft autoregressively generates GAMMA candidate tokens d_1..d_GAMMA.
//   2. Target does ONE forward pass on (so_far + d_1..d_GAMMA).
//   3. Walk left-to-right: accept d_i iff target's argmax at position i equals
//      d_i. On first mismatch, use target's argmax and discard d_{i+1}..d_GAMMA.
//   4. If all GAMMA accepted, sample one bonus token from target's logits at
//      position GAMMA.
//   5. Trim both KV caches to the new accepted prefix. Repeat.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>
#include <algorithm>

static const std::string PROMPT_DELIM = "\n<<END_PROMPT>>\n";

static std::vector<std::string> read_prompts_from_stdin() {
    std::string buf((std::istreambuf_iterator<char>(std::cin)),
                    std::istreambuf_iterator<char>());
    std::vector<std::string> out;
    size_t start = 0;
    while (start < buf.size()) {
        size_t pos = buf.find(PROMPT_DELIM, start);
        if (pos == std::string::npos) {
            std::string p = buf.substr(start);
            if (!p.empty()) out.push_back(p);
            break;
        }
        std::string p = buf.substr(start, pos - start);
        if (!p.empty()) out.push_back(p);
        start = pos + PROMPT_DELIM.size();
    }
    return out;
}

static llama_token greedy_argmax(const float * logits, int n_vocab) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return (llama_token) best;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    if (params.speculative.mparams_dft.path.empty()) {
        LOG_ERR("%s: --model-draft (-md) is required\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // --- load target model (ONCE) -------------------------------------------
    auto init_tgt = common_init_from_params(params);
    llama_model   * model_tgt = init_tgt->model();
    llama_context * ctx_tgt   = init_tgt->context();
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // --- load draft model (ONCE) --------------------------------------------
    common_params params_dft = params;
    params_dft.n_parallel   = 1;
    params_dft.n_ctx        = params.speculative.n_ctx;
    params_dft.n_batch      = llama_n_ctx_seq(ctx_tgt);
    params_dft.devices      = params.speculative.devices;
    params_dft.model        = params.speculative.mparams_dft;
    params_dft.n_gpu_layers = params.speculative.n_gpu_layers;
    auto mparams_dft = common_model_params_to_llama(params_dft);
    auto cparams_dft = common_context_params_to_llama(params_dft);
    llama_model   * model_dft = llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft);
    if (!model_dft) {
        LOG_ERR("failed to load draft model '%s'\n", params_dft.model.path.c_str());
        return 1;
    }
    llama_context * ctx_dft = llama_init_from_model(model_dft, cparams_dft);

    // --- speculation parameters (constant across prompts) -------------------
    const int GAMMA = std::max(1, params.speculative.n_max);
    const int MAX_NEW = params.n_predict > 0 ? params.n_predict : 128;
    LOG_INF("\nGAMMA=%d  MAX_NEW=%d  vocab=%d\n", GAMMA, MAX_NEW, n_vocab);

    // --- read prompts from stdin --------------------------------------------
    auto prompts = read_prompts_from_stdin();
    LOG_INF("read %zu prompts from stdin\n", prompts.size());
    if (prompts.empty()) {
        LOG_ERR("no prompts on stdin (delimited by '<<END_PROMPT>>')\n");
        return 1;
    }

    // ========================================================================
    // PER-PROMPT LOOP   (agent edits the algorithm inside, but should keep
    //                    the KV reset + per-prompt emit format intact)
    // ========================================================================
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        // Fully clear KV caches for both contexts.
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, 0, -1);

        // Tokenize this prompt and feed prompt[:-1] to both contexts.
        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompts[pi], true, true);
        const int n_prompt = (int) inp.size();
        if (n_prompt < 2) {
            fprintf(stdout, "\n=== PROMPT %zu BEGIN ===\n", pi);
            fprintf(stdout, "ERROR: prompt too short (n_prompt=%d)\n", n_prompt);
            fprintf(stdout, "=== PROMPT %zu END ===\n", pi);
            fflush(stdout);
            continue;
        }
        llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), n_prompt - 1));
        int n_past_tgt = n_prompt - 1;
        llama_decode(ctx_dft, llama_batch_get_one(inp.data(), n_prompt - 1));
        int n_past_dft = n_prompt - 1;

        llama_token id_last = inp.back();
        int n_predict = 0;
        int n_drafted = 0;
        int n_accept  = 0;
        int n_cycles  = 0;
        std::vector<llama_token> out_tokens;
        out_tokens.reserve(MAX_NEW + GAMMA);

        auto t0 = ggml_time_us();

        // ====================================================================
        // SPECULATIVE-DECODING ALGORITHM   (agent edits this section)
        // ====================================================================
        while (n_predict < MAX_NEW) {
            ++n_cycles;

            // (1) Draft generates GAMMA candidate tokens autoregressively.
            std::vector<llama_token> draft;
            draft.reserve(GAMMA);
            llama_decode(ctx_dft, llama_batch_get_one(&id_last, 1));
            n_past_dft += 1;

            for (int i = 0; i < GAMMA; ++i) {
                const float * d_logits = llama_get_logits_ith(ctx_dft, -1);
                llama_token d_tok = greedy_argmax(d_logits, n_vocab);
                draft.push_back(d_tok);
                llama_decode(ctx_dft, llama_batch_get_one(&d_tok, 1));
                n_past_dft += 1;
            }
            n_drafted += GAMMA;

            // (2) Target verifies in ONE parallel forward pass.
            llama_batch batch_tgt = llama_batch_init(GAMMA + 1, 0, 1);
            common_batch_add(batch_tgt, id_last, n_past_tgt, { 0 }, true);
            for (int i = 0; i < GAMMA; ++i) {
                common_batch_add(batch_tgt, draft[i], n_past_tgt + 1 + i, { 0 }, true);
            }
            llama_decode(ctx_tgt, batch_tgt);

            // (3) Greedy accept/reject walk.
            int accepted = 0;
            llama_token new_tok = -1;
            for (int i = 0; i < GAMMA; ++i) {
                const float * t_logits = llama_get_logits_ith(ctx_tgt, i);
                llama_token t_argmax = greedy_argmax(t_logits, n_vocab);
                if (t_argmax == draft[i]) {
                    accepted += 1;
                } else {
                    new_tok = t_argmax;
                    break;
                }
            }
            if (accepted == GAMMA) {
                const float * t_logits = llama_get_logits_ith(ctx_tgt, GAMMA);
                new_tok = greedy_argmax(t_logits, n_vocab);
            }

            n_accept += accepted;
            const int produced = accepted + 1;
            n_predict += produced;

            // (5) KV trim
            const int keep_tgt = n_past_tgt + 1 + accepted;
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, keep_tgt, -1);
            n_past_tgt = keep_tgt;
            for (int i = 0; i < accepted; ++i) {
                out_tokens.push_back(draft[i]);
            }
            out_tokens.push_back(new_tok);

            if (llama_vocab_is_eog(vocab, new_tok)) break;

            const int keep_dft = (n_past_dft - GAMMA) + accepted;
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, keep_dft, -1);
            n_past_dft = keep_dft;

            id_last = new_tok;

            llama_batch_free(batch_tgt);
        }
        // ====================================================================

        auto t1 = ggml_time_us();
        const double dt = (t1 - t0) / 1e6;

        // Per-prompt block emit (parsed by harness on stdout).
        fprintf(stdout, "\n=== PROMPT %zu BEGIN ===\n", pi);
        fprintf(stdout, "decoded %d tokens in %.3fs = %.2f tok/s\n",
                n_predict, dt, n_predict / dt);
        fprintf(stdout, "cycles %d  gamma %d\n", n_cycles, GAMMA);
        fprintf(stdout, "n_drafted %d\n", n_drafted);
        fprintf(stdout, "n_accept %d\n", n_accept);
        fprintf(stdout, "TOKENS:");
        for (auto t : out_tokens) fprintf(stdout, " %d", (int)t);
        fprintf(stdout, "\n=== PROMPT %zu END ===\n", pi);
        fflush(stdout);
    }
    // ========================================================================

    llama_free(ctx_dft);
    llama_model_free(model_dft);
    llama_backend_free();
    return 0;
}
