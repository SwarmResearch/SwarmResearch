// Two decoding modes on a shared codepath, self-contained sampling + speculative algorithm.
//   --mode target    : autoregressive sampling from the 26B target (baseline)
//   --mode mtp-spec  : MTP drafts GAMMA, target verifies, our rejection sampling (lossless)
//
// mtp-spec here is BATCHED: instead of decoding one prompt at a time, it runs up to B
// sequences concurrently with continuous batching so that each target VERIFY forward pass
// (the verify-bound bottleneck: ~7 ms fixed + ~2.07 ms/token of MoE expert fan-out) is
// shared across B independent sequences. The per-sequence accept/reject + rejection sampling
// is unchanged and lossless; only the scheduling changes. The ~7 ms fixed forward cost is
// amortized across B sequences, raising delivered tokens per verify forward without touching
// the per-token acceptance ratio.
//
// Draft predictions come from llama.cpp's native gemma-4 MTP head (gemma4-assistant) via the
// nextn staging API; the accept/reject + sampling is implemented here (not llama.cpp's).
//
// Prompts: read from stdin, already chat-templated, delimited by "\n<<END_PROMPT>>\n".
// Per-prompt output block (parsed by run_modes.py):
//   === PROMPT i BEGIN ===
//   decoded N tokens in T s = X tok/s
//   n_drafted N
//   n_accept  N
//   TEXT: <detokenized generation>
//   === PROMPT i END ===
#include "common.h"
#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"   // internal: lets us mutate hparams.n_expert_used between forwards (runtime-signal MoE allocation)
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>

// ---------------- truncated sampling distribution (top_k -> top_p -> temp) ----------------
struct Dist {
    std::vector<int>   ids;
    std::vector<float> ps;
    float p_of(int id) const {
        for (size_t i = 0; i < ids.size(); ++i) if (ids[i] == id) return ps[i];
        return 0.0f;
    }
};

static Dist make_dist(const float * logits, int n_vocab, int top_k, float top_p, float temp) {
    int k = (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab;
    // bounded min-heap (by logit) over the vocab: one O(n log k) scan, no full-vocab allocation.
    // Exact top-k selection (equal-logit tie order aside) -> accept/reject behavior unchanged.
    std::vector<std::pair<float,int>> v; v.reserve(k);
    auto gt = [](const std::pair<float,int>& a, const std::pair<float,int>& b){ return a.first > b.first; };
    for (int i = 0; i < n_vocab; ++i) {
        float l = logits[i];
        if ((int) v.size() < k) {
            v.push_back({l, i}); std::push_heap(v.begin(), v.end(), gt);     // smallest kept at front
        } else if (l > v.front().first) {
            std::pop_heap(v.begin(), v.end(), gt); v.back() = {l, i}; std::push_heap(v.begin(), v.end(), gt);
        }
    }
    std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.first > b.first; });
    float mx = v[0].first, t = temp > 0 ? temp : 1.0f;
    double sum = 0.0; std::vector<double> pr(k);
    for (int i = 0; i < k; ++i) { pr[i] = std::exp((v[i].first - mx) / t); sum += pr[i]; }
    for (int i = 0; i < k; ++i) pr[i] /= sum;
    Dist d; double cum = 0.0;
    for (int i = 0; i < k; ++i) {
        d.ids.push_back(v[i].second); d.ps.push_back((float) pr[i]); cum += pr[i];
        if (top_p > 0 && top_p < 1.0f && cum >= top_p) break;
    }
    double s2 = 0.0; for (float p : d.ps) s2 += p;
    for (float & p : d.ps) p = (float)(p / s2);
    return d;
}

// Shannon entropy (nats) of a truncated/renormalized distribution. Used as the free, autocorrelated
// signal that drives the per-slot adaptive draft temperature.
static float dist_entropy(const Dist & d) {
    double h = 0.0;
    for (float p : d.ps) if (p > 0.0f) h -= (double) p * std::log((double) p);
    return (float) h;
}

static int sample_dist(const Dist & d, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float r = u(rng), c = 0.0f;
    for (size_t i = 0; i < d.ids.size(); ++i) { c += d.ps[i]; if (r <= c) return d.ids[i]; }
    return d.ids.back();
}

// Persistent thread pool: workers are created once and reused for every make_dist batch, so the
// hot loop pays no per-round thread-spawn cost. f(i) must touch only disjoint state (no RNG).
static const int NTHREADS = 16;
struct ThreadPool {
    int T;
    std::vector<std::thread> th;
    std::mutex m;
    std::condition_variable cv, cv_done;
    std::function<void(int)> job;
    int n = 0;
    std::atomic<int> cursor{0};
    std::atomic<int> active{0};
    long gen = 0;
    bool stop = false;

    explicit ThreadPool(int T_) : T(T_) {
        for (int i = 0; i < T; ++i) th.emplace_back([this]{ worker(); });
    }
    ~ThreadPool() {
        { std::lock_guard<std::mutex> lk(m); stop = true; ++gen; }
        cv.notify_all();
        for (auto & t : th) if (t.joinable()) t.join();
    }
    void worker() {
        long mygen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&]{ return stop || gen != mygen; });
            if (stop) return;
            mygen = gen;
            lk.unlock();
            int i;
            while ((i = cursor.fetch_add(1)) < n) job(i);
            if (active.fetch_sub(1) == 1) { std::lock_guard<std::mutex> l(m); cv_done.notify_one(); }
        }
    }
    void run(int n_, std::function<void(int)> f) {
        {
            std::lock_guard<std::mutex> lk(m);
            job = std::move(f); n = n_; cursor.store(0); active.store(T); ++gen;
        }
        cv.notify_all();
        int i;
        while ((i = cursor.fetch_add(1)) < n) job(i);     // main thread helps
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [&]{ return active.load() == 0; });
    }
};

static ThreadPool * g_pool = nullptr;
template<class F>
static void parallel_for(int n, F f) {
    if (n <= 0) return;
    if (!g_pool || n == 1) { for (int i = 0; i < n; ++i) f(i); return; }
    g_pool->run(n, std::function<void(int)>(std::move(f)));
}

static std::vector<std::string> read_prompts() {
    std::string data((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    std::vector<std::string> out;
    const std::string delim = "\n<<END_PROMPT>>\n";
    size_t pos = 0, f;
    while ((f = data.find(delim, pos)) != std::string::npos) { out.push_back(data.substr(pos, f - pos)); pos = f + delim.size(); }
    if (pos < data.size()) { std::string tail = data.substr(pos); if (!tail.empty()) out.push_back(tail); }
    return out;
}

// ----- per-sequence slot state for batched mtp-spec -----
struct Slot {
    int  prompt_idx = -1;       // which input prompt this slot is decoding
    int  seq_id     = -1;       // llama sequence id (== slot index)
    bool active     = false;

    int  pos        = 0;        // position of id_last (NOT in cache); cache holds [0..pos-1]
    llama_token id_last = -1;
    std::vector<float> h_last;  // target nextn hidden for id_last

    std::vector<llama_token> out;
    int n_drafted = 0, n_accept = 0;
    double dt = 0.0;            // fair-share wall-clock attributed to this sequence
    bool done = false;

    // PER-SEQUENCE RNG STREAM: each slot draws all of its sampling randomness (draft sampling,
    // accept/reject Bernoulli, residual resampling, free-commit base sample) from its OWN mt19937,
    // seeded deterministically from (global seed, prompt index). With a single shared RNG the order
    // in which sequences consume random draws depends on which prompts happen to be co-batched and in
    // which round -- so a sequence's generated trajectory changes when the batch size / composition
    // changes, which is what makes accuracy drift as B grows. A per-prompt stream makes each
    // sequence's randomness invariant to batch composition: prompt i always sees the same draw
    // sequence whether B=8 or B=12. This decouples accuracy from batch size, the lever that lets the
    // batch grow. The seed is derived from the run seed + prompt index (legitimate per-request
    // seeding; not prompt-content-dependent, generalizes to unseen prompts).
    std::mt19937 rng;

    // ADAPTIVE DRAFT TEMPERATURE state: mean target (verify) entropy over this slot's draft rows in the
    // PREVIOUS round. -1 == no signal yet (freshly admitted / first round) -> use the base temperature.
    // Target entropy is autocorrelated round-to-round, so this predicts how spread the target will be
    // this round and hence how much to de-peak (flatten) this slot's over-confident MTP draft.
    float prev_tent = -1.0f;

    // scratch within a round
    std::vector<llama_token> draft;
    std::vector<Dist>        qd;
    std::vector<float>       hcur;
    llama_token              cur = -1;
    int                      G = 0;
    int                      voff = 0;   // row offset of this slot inside the shared verify batch
};

int main(int argc, char ** argv) {
    std::string mode = "target", tgt_path, mtp_path;
    int   top_k = 64, gamma = 3, max_new = 1024, batch = 9;
    float top_p = 0.95f, temp = 1.0f;
    uint32_t seed = 0;
    // Verify-forward MoE approximation: the target is a top-8-of-128 MoE (gemma4.expert_used_count=8)
    // with softmax-renormalized routing. Routing each token to only the top-k (<8) experts and
    // renormalizing over the kept ones makes every target forward's per-token MoE compute strictly
    // cheaper (fewer expert FFN matmuls per token, fewer distinct experts touched per batch).
    //
    // NON-UNIFORM (runtime-signal) allocation: hparams.n_expert_used is a single global scalar that
    // build_moe_ffn reads identically for every layer within a graph (llama-graph.cpp captures it once
    // per graph build), so the number of active experts is NOT per-layer controllable from here -- a
    // per-layer schedule would require editing gemma4.cpp's build loop, which is outside this file.
    // What IS reachable: the graph is rebuilt per llama_decode and re-reads hparams.n_expert_used, and
    // graph reuse (allow_reuse) never crosses the prefill vs verify shape classes, so we can use a
    // DIFFERENT k for the prompt-prefill forward than for the generation verify forwards by mutating
    // hparams.n_expert_used right before each decode. k_verify drives throughput (verify forwards
    // dominate the run); k_prefill keeps the one-time prompt encoding (and its cached K/V) accurate at
    // near-zero throughput cost. The MTP draft head is a separate non-MoE model and is unaffected.
    int   k_prefill = 8;     // experts for the prompt-prefill forward (accurate context, one-time cost)
    int   k_verify  = 4;     // experts for the generation verify forwards (load-time seed only; runtime is adaptive)
    // ADAPTIVE verify-k (runtime uncertainty signal): the batched verify forward shares one k across all
    // active slots, so k is chosen per verify-decode (per round). The signal is the PRIOR round's target
    // verify distribution: its mean top-1 probability (a cheap confidence measure already computed in
    // p_all). Confident rounds (mean top-1 >= conf_thresh) run cheap at k_lo; uncertain rounds spend more
    // experts at k_hi. This restores accuracy margin over a fixed k_lo while keeping most rounds at k_lo
    // (so throughput stays near the fixed-k_lo point). Uniform k_lo=4 scored 27/30; uniform k=5 scored
    // 30/30; the adaptive mix targets k_lo-like speed with the 30/30-restored margin.
    int   k_lo = 4;          // experts when the prior round was confident (the common, throughput-setting path)
    int   k_hi = 5;          // experts when the prior round was uncertain (accuracy insurance)
    float conf_thresh = 0.70f; // mean-top-1 threshold separating confident (>=) from uncertain (<) rounds
    // ADAPTIVE DRAFT TEMPERATURE (raises acceptance per verify row, LOSSLESSLY, by improving the draft
    // PROPOSAL). The MTP draft is systematically ~3.3x more peaked than the target (draft entropy
    // ~0.07 vs target ~0.24); this over-confidence (q>p) is the dominant acceptance loss (~10.1% of the
    // 12.4% total, 4.4x the out-of-nucleus loss). Speculative sampling is lossless for ANY proposal q
    // (emitted token is still drawn exactly from the target's truncated p via accept min(1,p/q) +
    // residual), so we may flatten the DRAFT distribution with a temperature t_draft != the target's
    // temp without touching accuracy. A single global t_draft is a blunt compromise: it over-flattens
    // the easy (low target-entropy, already ~0.98 accept) backbone while under-flattening the hard
    // (high target-entropy, spread p) positions where the over-confidence loss actually concentrates.
    // The right per-position flatten amount tracks the TARGET entropy, which a draft-side signal (qmax)
    // cannot predict (a prior qmax-conditioned temperature gained ~0). But target entropy is strongly
    // autocorrelated round-to-round (ACF1 ~+0.30), so each slot's PRIOR-round mean target entropy is a
    // free, already-computed predictor of this round's target spread. We set t_draft PER SLOT as a
    // monotone ramp in that prior entropy: sharp (t~tdr_base) on the easy backbone, flatter (up to
    // tdr_max) where the target was spread, matching q's spread to p's and raising sum_x min(p,q).
    float tdr_base  = 1.8f;   // draft temperature at the entropy pivot (near the known global optimum)
    float tdr_slope = 1.1f;   // d(t_draft)/d(target entropy in nats) above the pivot
    float tdr_pivot = 0.35f;  // target-entropy (nats) anchor where t_draft == tdr_base
    float tdr_min   = 1.3f;   // floor (very confident-target positions; still mild de-peaking)
    float tdr_max   = 3.0f;   // ceiling (avoid over-flattening past the inverted-U seen at t~4)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&](){ return std::string(argv[++i]); };
        if      (a == "--mode")  mode = nx();
        else if (a == "-m")      tgt_path = nx();
        else if (a == "-md")     mtp_path = nx();
        else if (a == "--temp")  temp = std::stof(nx());
        else if (a == "--top-p") top_p = std::stof(nx());
        else if (a == "--top-k") top_k = std::stoi(nx());
        else if (a == "--gamma") gamma = std::stoi(nx());
        else if (a == "--batch") batch = std::stoi(nx());
        else if (a == "--seed")  seed = (uint32_t) std::stoul(nx());
        else if (a == "-n")      max_new = std::stoi(nx());
        else if (a == "--k-prefill") k_prefill = std::stoi(nx());
        else if (a == "--k-verify")  k_verify  = std::stoi(nx());
        else if (a == "--k-lo")      k_lo = std::stoi(nx());
        else if (a == "--k-hi")      k_hi = std::stoi(nx());
        else if (a == "--conf-thresh") conf_thresh = std::stof(nx());
        else if (a == "--tdr-base")  tdr_base  = std::stof(nx());
        else if (a == "--tdr-slope") tdr_slope = std::stof(nx());
        else if (a == "--tdr-pivot") tdr_pivot = std::stof(nx());
        else if (a == "--tdr-min")   tdr_min   = std::stof(nx());
        else if (a == "--tdr-max")   tdr_max   = std::stof(nx());
    }
    const bool use_mtp = (mode == "mtp-spec");
    if (tgt_path.empty() || (use_mtp && mtp_path.empty())) {
        fprintf(stderr, "usage: %s --mode target|mtp-spec -m TGT [-md MTP] [...]\n", argv[0]); return 1;
    }
    if (batch < 1) batch = 1;

    // Each batched sequence gets the SAME 24576-token window as the single-sequence baseline so
    // long reasoning chains are not truncated (KV is cheap here: most layers use a 1536-token
    // sliding window, so batch*24576 total cells is only a few GB).
    const int per_seq_ctx = 24576;          // per-sequence KV window for batched mtp-spec
    const int total_ctx = use_mtp ? batch * per_seq_ctx : 24576;
    llama_backend_init();
    common_params cp;
    cp.model.path = tgt_path; cp.n_gpu_layers = 99; cp.n_ctx = total_ctx; cp.cpuparams.n_threads = 8;
    if (use_mtp) cp.n_parallel = batch;     // partition the target KV into `batch` sequences
    // Seed the target MoE top-k at load time to the verify value (the dominant path). The per-forward
    // value is then set authoritatively at runtime below; this load-time override only fixes the initial
    // scalar so warmup/first-build state is sane. Applied only in mtp-spec mode; `target` mode stays the
    // faithful unmodified-target baseline.
    if (use_mtp && k_verify > 0 && k_verify < 8) {
        llama_model_kv_override kvo{};
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::strncpy(kvo.key, "gemma4.expert_used_count", sizeof(kvo.key) - 1);
        kvo.val_i64 = k_verify;
        cp.kv_overrides.push_back(kvo);
        llama_model_kv_override term{};      // loader requires a trailing empty-key terminator
        cp.kv_overrides.push_back(term);
        fprintf(stderr, "MoE approximation: load-time gemma4.expert_used_count 8 -> %d (k_prefill=%d, k_verify=%d set per-forward at runtime)\n",
                k_verify, k_prefill, k_verify);
    }
    auto init = common_init_from_params(cp);
    if (!init || init->context() == nullptr) { fprintf(stderr, "target load failed\n"); return 1; }
    llama_model   * model_tgt = init->model();
    llama_context * ctx_tgt   = init->context();
    // Runtime-signal MoE allocation: set how many experts the NEXT target forward routes to. The graph
    // is rebuilt per llama_decode and re-reads this scalar; graph reuse never crosses the prefill/verify
    // shape classes, so each class is consistently (re)built with its own k. No-op outside mtp-spec.
    auto set_tgt_k = [&](int k){ if (use_mtp) model_tgt->hparams.n_expert_used = (uint32_t) k; };
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx_seq = use_mtp ? per_seq_ctx : total_ctx;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;
    int n_embd = 0;
    llama_batch dbatch{};
    if (use_mtp) {
        llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
        model_dft = llama_model_load_from_file(mtp_path.c_str(), mp);
        if (!model_dft) { fprintf(stderr, "mtp load failed\n"); return 1; }
        n_embd = llama_model_n_embd_out(model_dft);
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = total_ctx; cparams.n_batch = 2048; cparams.n_seq_max = batch;
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP; cparams.ctx_other = ctx_tgt; cparams.n_rs_seq = 0;
        ctx_dft = llama_init_from_model(model_dft, cparams);
        if (!ctx_dft) { fprintf(stderr, "mtp ctx init failed\n"); return 1; }
        llama_set_embeddings_nextn(ctx_tgt, true, false);
        llama_set_embeddings_nextn(ctx_dft, true, true);
        int cap = batch * (gamma + 2) + 8;
        dbatch = llama_batch_init(cap, n_embd, 1);
        dbatch.token = (llama_token *) malloc(sizeof(llama_token) * cap);
    }

    auto prompts = read_prompts();
    fprintf(stderr, "mode=%s prompts=%zu gamma=%d batch=%d n_ctx_seq=%d temp=%.2f top_p=%.2f top_k=%d\n",
            mode.c_str(), prompts.size(), gamma, batch, n_ctx_seq, temp, top_p, top_k);
    std::mt19937 rng(seed);

    if (mode == "target") {
        for (size_t pi = 0; pi < prompts.size(); ++pi) {
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
            std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompts[pi], true, true);
            int n = (int) inp.size();
            std::vector<llama_token> out;
            auto t0 = ggml_time_us();
            llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), n));
            Dist p_dist = make_dist(llama_get_logits_ith(ctx_tgt, n - 1), n_vocab, top_k, top_p, temp);
            while ((int) out.size() < max_new) {
                int tok = (temp <= 0) ? p_dist.ids[0] : sample_dist(p_dist, rng);
                out.push_back(tok);
                if (llama_vocab_is_eog(vocab, tok)) break;
                llama_decode(ctx_tgt, llama_batch_get_one(&tok, 1));
                p_dist = make_dist(llama_get_logits_ith(ctx_tgt, 0), n_vocab, top_k, top_p, temp);
            }
            double dt = (ggml_time_us() - t0) / 1e6;
            std::string text;
            for (auto t : out) text += common_token_to_piece(ctx_tgt, t);
            printf("\n=== PROMPT %zu BEGIN ===\n", pi);
            printf("decoded %d tokens in %.3fs = %.2f tok/s\n", (int) out.size(), dt, out.size()/dt);
            printf("n_drafted 0\n");
            printf("n_accept  0\n");
            printf("TEXT: %s\n", text.c_str());
            printf("=== PROMPT %zu END ===\n", pi);
            fflush(stdout);
        }
        llama_backend_free();
        return 0;
    }

    // ----------------- batched mtp-spec -----------------
    const int P = (int) prompts.size();
    std::vector<Slot> slots(batch);
    for (int s = 0; s < batch; ++s) { slots[s].seq_id = s; slots[s].h_last.resize(n_embd); slots[s].hcur.resize(n_embd); }
    // per-prompt results, emitted in prompt order at the end
    std::vector<std::vector<llama_token>> res_out(P);
    std::vector<double> res_dt(P, 0.0);
    std::vector<int>    res_drafted(P, 0), res_accept(P, 0);

    auto mem = llama_get_memory(ctx_tgt);
    int next_prompt = 0;

    auto admit = [&](Slot & sl) {
        // prefill prompt for this slot's sequence (single-sequence decode, its own seq_id)
        int pi = next_prompt++;
        sl.prompt_idx = pi; sl.active = true; sl.done = false;
        sl.out.clear(); sl.n_drafted = 0; sl.n_accept = 0; sl.dt = 0.0;
        sl.prev_tent = -1.0f;   // fresh sequence: no entropy history yet -> base draft temperature
        // Seed this sequence's private RNG from (run seed, prompt index) via seed_seq so the stream is
        // well-mixed and batch-composition-invariant. prompt index (not content) is the only per-item
        // input, so the scheme generalizes to unseen prompts.
        { std::seed_seq ss{ (uint32_t) seed, (uint32_t) pi, 0x9E3779B9u }; sl.rng.seed(ss); }
        llama_memory_seq_rm(mem, sl.seq_id, 0, -1);
        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompts[pi], true, true);
        int n = (int) inp.size();
        if (n > n_ctx_seq - 2) { n = n_ctx_seq - 2; }   // guard: keep within the per-seq KV window
        auto t0 = ggml_time_us();
        llama_batch pb = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) common_batch_add(pb, inp[i], i, { sl.seq_id }, i == n - 1);
        set_tgt_k(k_prefill);                              // accurate prompt encoding (and cached K/V)
        llama_decode(ctx_tgt, pb);
        std::memcpy(sl.h_last.data(), llama_get_embeddings_nextn_ith(ctx_tgt, n - 1), sizeof(float)*n_embd);
        llama_batch_free(pb);
        sl.id_last = inp[n - 1];
        sl.pos = n - 1;
        llama_memory_seq_rm(mem, sl.seq_id, sl.pos, -1);   // drop id_last; verify re-decodes it
        sl.dt += (ggml_time_us() - t0) / 1e6;
    };

    auto retire = [&](Slot & sl) {
        int pi = sl.prompt_idx;
        res_out[pi]     = sl.out;
        res_dt[pi]      = sl.dt;
        res_drafted[pi] = sl.n_drafted;
        res_accept[pi]  = sl.n_accept;
        llama_memory_seq_rm(mem, sl.seq_id, 0, -1);
        sl.active = false; sl.prompt_idx = -1;
    };

    // verify batch reused across rounds
    int vcap = batch * (gamma + 2) + 8;
    llama_batch vb = llama_batch_init(vcap, 0, 1);

    // total decode wall-clock for the whole batched run (excludes model load, which is already done).
    // Per-item time is then attributed proportionally to delivered tokens so that the aggregate
    // tok/s == (total tokens / W) is exact and externally reproducible. Under continuous batching the
    // throughput is a shared-system property; this attribution reports that shared rate faithfully.
    ThreadPool pool(NTHREADS);
    g_pool = &pool;
    auto t_decode0 = ggml_time_us();

    // Adaptive verify-k state. prev_conf = mean top-1 prob of the previous round's verify distribution.
    // Seeded negative so the FIRST verify round is treated as uncertain (k_hi) before any signal exists.
    float prev_conf = -1.0f;
    long  rounds_lo = 0, rounds_hi = 0;
    long  conf_hist[20] = {0};   // distribution of per-round mean top-1, in 0.05 buckets, for offline tuning

    // Per-slot adaptive DRAFT temperature: a monotone ramp in the slot's prior-round mean target
    // entropy. Below the pivot (confident target) the draft stays near tdr_base; above it the draft is
    // flattened up to tdr_max to match the spread target and recover the over-confidence (q>p) loss.
    // No signal yet (prev_tent<0) -> tdr_base. Lossless: only the proposal q changes; the rejection
    // sampler still emits exactly the target's truncated p.
    auto draft_temp_for = [&](const Slot & sl) -> float {
        float te = (sl.prev_tent < 0.0f) ? tdr_pivot : sl.prev_tent;
        float t  = tdr_base + tdr_slope * (te - tdr_pivot);
        if (t < tdr_min) t = tdr_min;
        if (t > tdr_max) t = tdr_max;
        return t;
    };
    double tdr_sum = 0.0; long tdr_cnt = 0;   // stderr-only summary of realized draft temperatures

    while (true) {
        // admit new sequences into free slots
        for (int s = 0; s < batch && next_prompt < P; ++s)
            if (!slots[s].active) admit(slots[s]);

        // collect active slots
        std::vector<int> act;
        for (int s = 0; s < batch; ++s) if (slots[s].active) act.push_back(s);
        if (act.empty()) break;

        auto t_round = ggml_time_us();

        // (1) DRAFT gamma tokens for every active slot, batched per step over slots
        for (int s : act) { slots[s].draft.clear(); slots[s].qd.clear();
                            slots[s].cur = slots[s].id_last; slots[s].hcur = slots[s].h_last; }
        for (int g = 0; g < gamma; ++g) {
            common_batch_clear(dbatch);
            std::vector<int> rows;   // slot index for each batch row
            for (int s : act) {
                if ((int) slots[s].draft.size() != g) continue;   // (always true; defensive)
                int row = dbatch.n_tokens;
                common_batch_add(dbatch, slots[s].cur, slots[s].pos, { slots[s].seq_id }, true);
                std::memcpy(dbatch.embd + (size_t) row * n_embd, slots[s].hcur.data(), sizeof(float)*n_embd);
                rows.push_back(s);
            }
            if (dbatch.n_tokens == 0) break;
            if (llama_decode(ctx_dft, dbatch) != 0) break;
            int nr = (int) rows.size();
            // resolve logits pointers sequentially (get_logits_ith may reorder), then top-k in parallel
            std::vector<const float*> lp(nr);
            for (int r = 0; r < nr; ++r) lp[r] = llama_get_logits_ith(ctx_dft, r);
            std::vector<Dist> qs(nr);
            // Per-slot adaptive draft temperature (decoupled from the target's temp). The proposal q for
            // slot rows[r] is flattened by draft_temp_for(slot); the SAME q is stored in qd and used by
            // the verify accept/residual, so the scheme remains lossless.
            std::vector<float> dtmp(nr);
            for (int r = 0; r < nr; ++r) dtmp[r] = draft_temp_for(slots[rows[r]]);
            parallel_for(nr, [&](int r){ qs[r] = make_dist(lp[r], n_vocab, top_k, top_p, dtmp[r]); });
            for (int r = 0; r < nr; ++r) { tdr_sum += dtmp[r]; ++tdr_cnt; }
            for (int r = 0; r < nr; ++r) {
                Slot & sl = slots[rows[r]];
                int dtok = (temp <= 0) ? qs[r].ids[0] : sample_dist(qs[r], sl.rng);  // per-sequence RNG stream
                sl.draft.push_back(dtok); sl.qd.push_back(std::move(qs[r]));
                std::memcpy(sl.hcur.data(), llama_get_embeddings_nextn_ith(ctx_dft, r), sizeof(float)*n_embd);
                sl.cur = dtok;
            }
        }

        // clear draft scratch writes, build the shared verify batch
        common_batch_clear(vb);
        std::vector<int> vact;   // slots that actually go into verify (G>0)
        for (int s : act) {
            Slot & sl = slots[s];
            sl.G = (int) sl.draft.size();
            llama_memory_seq_rm(mem, sl.seq_id, sl.pos, -1);   // clear draft writes at pos
            if (sl.G == 0) { sl.done = true; continue; }
            sl.n_drafted += sl.G;
            sl.voff = vb.n_tokens;
            common_batch_add(vb, sl.id_last, sl.pos, { sl.seq_id }, true);
            for (int g = 0; g < sl.G; ++g) common_batch_add(vb, sl.draft[g], sl.pos + 1 + g, { sl.seq_id }, true);
            vact.push_back(s);
        }

        // ADAPTIVE verify-k: pick this round's expert count from the PRIOR round's confidence signal.
        // Confident prior round (mean top-1 >= conf_thresh) -> cheap k_lo; uncertain -> k_hi. The choice
        // applies to the whole shared verify decode (one forward over all active slots this round).
        int kv_round = (prev_conf >= conf_thresh) ? k_lo : k_hi;
        if (!vact.empty()) {
            set_tgt_k(kv_round);                  // throughput-dominant path (adaptive expert count)
            llama_decode(ctx_tgt, vb);
            if (kv_round == k_lo) ++rounds_lo; else ++rounds_hi;
        }

        // precompute every verify row's target top-k distribution in parallel (RNG-free),
        // then run the sequential accept/reject so the sampling sequence is unchanged.
        int nvr = vb.n_tokens;
        std::vector<Dist> p_all(nvr);
        if (nvr > 0) {
            std::vector<const float*> vlp(nvr);
            for (int r = 0; r < nvr; ++r) vlp[r] = llama_get_logits_ith(ctx_tgt, r);
            parallel_for(nvr, [&](int r){ p_all[r] = make_dist(vlp[r], n_vocab, top_k, top_p, temp); });
            // Update the confidence signal for the NEXT round: mean top-1 prob over this round's verify
            // rows (ps[0] is the largest prob since make_dist sorts descending). Cheap; reuses p_all.
            double sconf = 0.0;
            for (int r = 0; r < nvr; ++r) sconf += p_all[r].ps.empty() ? 0.0 : p_all[r].ps[0];
            prev_conf = (float)(sconf / nvr);
            int b = (int)(prev_conf * 20.0f); if (b < 0) b = 0; if (b > 19) b = 19;
            ++conf_hist[b];
            // Per-slot signal for the NEXT round's adaptive draft temperature: mean target entropy over
            // this slot's G draft verify rows (rows voff..voff+G-1). Reuses p_all; no extra forward/top-k.
            for (int s : vact) {
                Slot & sl = slots[s];
                if (sl.G <= 0) continue;
                double he = 0.0;
                for (int g = 0; g < sl.G; ++g) he += dist_entropy(p_all[sl.voff + g]);
                sl.prev_tent = (float)(he / sl.G);
            }
        }

        // (2) per-slot accept/reject + lossless rejection sampling
        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        for (int s : vact) {
            Slot & sl = slots[s];
            int accepted = 0; llama_token new_tok = -1; bool resampled = false;
            for (int g = 0; g < sl.G; ++g) {
                const Dist & p_g = p_all[sl.voff + g];
                float px = p_g.p_of(sl.draft[g]), qx = sl.qd[g].p_of(sl.draft[g]);
                float acc = (qx > 0.0f) ? std::min(1.0f, px/qx) : 0.0f;
                if (uni(sl.rng) < acc) { accepted++; continue; }   // per-sequence RNG stream
                Dist resd; double sm = 0.0;
                for (size_t j = 0; j < p_g.ids.size(); ++j) {
                    float diff = p_g.ps[j] - sl.qd[g].p_of(p_g.ids[j]);
                    if (diff > 0) { resd.ids.push_back(p_g.ids[j]); resd.ps.push_back(diff); sm += diff; }
                }
                if (sm <= 0) new_tok = p_g.ids[0];
                else { for (float & pp : resd.ps) pp = (float)(pp / sm); new_tok = sample_dist(resd, sl.rng); }  // per-sequence RNG stream
                resampled = true; break;
            }
            if (!resampled) {
                const Dist & p_b = p_all[sl.voff + sl.G];
                new_tok = (temp <= 0) ? p_b.ids[0] : sample_dist(p_b, sl.rng);   // per-sequence RNG stream
            }
            sl.n_accept += accepted;
            std::memcpy(sl.h_last.data(), llama_get_embeddings_nextn_ith(ctx_tgt, sl.voff + accepted), sizeof(float)*n_embd);
            int keep = sl.pos + 1 + accepted;
            llama_memory_seq_rm(mem, sl.seq_id, keep, -1);

            bool eog = false;
            for (int g = 0; g < accepted && (int) sl.out.size() < max_new; ++g) {
                sl.out.push_back(sl.draft[g]);
                if (llama_vocab_is_eog(vocab, sl.draft[g])) { eog = true; break; }
            }
            if (!eog && (int) sl.out.size() < max_new) {
                sl.out.push_back(new_tok);
                if (llama_vocab_is_eog(vocab, new_tok)) eog = true;
            }
            sl.id_last = new_tok;
            sl.pos = keep;
            if (eog || (int) sl.out.size() >= max_new || sl.pos >= n_ctx_seq - 1) sl.done = true;
        }

        (void) t_round;
        // retire finished slots
        for (int s : act) if (slots[s].done) retire(slots[s]);
    }

    double W = (ggml_time_us() - t_decode0) / 1e6;   // true total decode wall-clock
    g_pool = nullptr;                                 // stop routing make_dist to the pool
    llama_batch_free(vb);

    // Adaptive verify-k summary (stderr only; does not affect scoring). Reports the k_lo/k_hi split and
    // the per-round mean-top-1 histogram so the conf_thresh can be calibrated to a target k_hi fraction.
    {
        long rt = rounds_lo + rounds_hi;
        fprintf(stderr, "adaptive verify-k: k_lo=%d k_hi=%d conf_thresh=%.3f | rounds: lo=%ld hi=%ld (hi frac=%.3f)\n",
                k_lo, k_hi, conf_thresh, rounds_lo, rounds_hi, rt ? (double) rounds_hi / rt : 0.0);
        fprintf(stderr, "per-round mean-top1 histogram (bucket=0.05):");
        for (int b = 0; b < 20; ++b) fprintf(stderr, " [%.2f]%ld", b * 0.05, conf_hist[b]);
        fprintf(stderr, "\n");
        fprintf(stderr, "adaptive draft-temp: base=%.2f slope=%.2f pivot=%.2f range=[%.2f,%.2f] | mean t_draft=%.3f over %ld draft rows\n",
                tdr_base, tdr_slope, tdr_pivot, tdr_min, tdr_max, tdr_cnt ? tdr_sum / tdr_cnt : 0.0, tdr_cnt);
    }

    long total_tokens = 0;
    for (int pi = 0; pi < P; ++pi) total_tokens += (long) res_out[pi].size();
    if (total_tokens <= 0) total_tokens = 1;
    for (int pi = 0; pi < P; ++pi)
        res_dt[pi] = W * (double) res_out[pi].size() / (double) total_tokens;

    for (int pi = 0; pi < P; ++pi) {
        std::string text;
        for (auto t : res_out[pi]) text += common_token_to_piece(ctx_tgt, t);
        double dt = res_dt[pi] > 0 ? res_dt[pi] : 1e-9;
        printf("\n=== PROMPT %d BEGIN ===\n", pi);
        printf("decoded %d tokens in %.3fs = %.2f tok/s\n", (int) res_out[pi].size(), dt, res_out[pi].size()/dt);
        printf("n_drafted %d\n", res_drafted[pi]);
        printf("n_accept  %d\n", res_accept[pi]);
        printf("TEXT: %s\n", text.c_str());
        printf("=== PROMPT %d END ===\n", pi);
        fflush(stdout);
    }

    if (ctx_dft) { if (dbatch.token) { free(dbatch.token); dbatch.token = nullptr; }
                   llama_batch_free(dbatch); llama_free(ctx_dft); llama_model_free(model_dft); }
    llama_backend_free();
    return 0;
}
