# Commercial VLM Quantitative Replication Protocol

**Version**: v2 (revised to align with paper_v8)
**Date**: 2026-05-25
**Companion to**: `paper_v8.md` §2.2 / §3.1.2
**Stimulus**: `104.jpg` (Watanabe Illusion, ground truth Q1 = 1, Q2 = 11)
**Reference data**:
- Human N = 132 (II μ = 4.15, SD = 2.27)
- Qwen2.5-VL-32B on A100 N = 60 (II μ = 4.91, SD = 0.63)
- Qwen2.5-VL-32B on DGX Spark N = 60 (II μ = 4.92, SD = 0.65)
**Target output (per model)**: `n60_progress_<model>.json` in the same schema as the existing `n60_progress_a100.json` / `n60_progress_spark.json`.

---

## 1. Purpose

paper_v8 §3.1.2 has established that Qwen2.5-VL-32B, run on two different hardware platforms (A100 / DGX Spark), produces near-identical Illusion-Index distributions on the Watanabe Illusion. This demonstrates **hardware-level reproducibility** of the VLM bias within a single open-weights model.

This protocol adds a third axis of replication: **software-level / cross-vendor reproducibility**. We test three commercial VLMs from three different vendors with three different architectures and three different training pipelines, asking whether the same illusion-magnitude region (II ≈ 4–5) is reached. A positive result would establish the bias as a robust property of VLMs as a class rather than an idiosyncrasy of one open-weights model. A negative result — e.g. one or more commercial VLMs sitting near II = 0 — would be equally informative, sharpening the question of which architectural / training features produce the bias.

The commercial VLM data also functions as the quantitative successor to the §3.1.1 exploratory observations, which were N = 1 per model and could not distinguish a genuine perceptual-like bias from a single noisy sample.

---

## 2. Scope and design alignment

Because the Watanabe Illusion is *defined* by the mental-extension bias measured by Q1 (from top) and Q2 (from bottom) — and because the Illusion Index II is the paper's primary illusion-magnitude measure (paper_v8 §2.4) — this protocol restricts itself to **Q1 and Q2 only**. The Q3 angle question is, per paper_v8 §3.1.3, a separate language-driven phenomenon and is documented in Appendix A; we do not include it here.

The protocol matches the human psychophysics screen wording and the **two-turn independent** structure used for the Qwen runs. Each Q1 and Q2 turn is a fresh conversation; the model never sees its own previous answer to the other question. This matches what human participants experienced on the G1/G2 screens of the online experiment.

A Sequential variant is **not** included. The paper's main claim concerns the bias-magnitude distribution under conditions matched to humans; adding a within-conversation context to commercial models would test a different question (in-context modulation), which is left for future work.

---

## 3. Models

All three are vision-capable, **non-reasoning** or reasoning explicitly disabled, with date-stamped API snapshots:

| Vendor | Model | API ID | Thinking/reasoning |
|---|---|---|---|
| Anthropic | Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | Off by default; do not send `thinking` parameter |
| OpenAI | GPT-4o | `gpt-4o-2024-11-20` | Non-reasoning model class |
| Google | Gemini 2.5 Flash | `gemini-2.5-flash` | `thinkingBudget: 0` (explicit off) |

**Models intentionally not included.** Current frontier models (Claude Opus 4.7, GPT-5.5, Gemini 3.x Pro) have reasoning layers that cannot be turned off and would conflate vision-encoder→language-generation effects with reasoning-trace effects. Restricting to non-reasoning models matches the Qwen2.5-VL-32B condition.

---

## 4. Stimulus delivery

`104.jpg` (560 × 280 px, JPEG) is sent to each API as base64-encoded image data alongside the textual prompt.

**Per-API image payload format:**
- Anthropic: `{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "<b64>"}}`
- OpenAI (Chat Completions): `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<b64>", "detail": "high"}}`
- Google: `{"inline_data": {"mime_type": "image/jpeg", "data": "<b64>"}}`

OpenAI's `detail: "high"` is set explicitly to avoid auto-downsampling that would risk dropping the right-edge dots.

**Image preprocessing verification.** Before the main run, each API is queried once with the diagnostic prompt "Describe the image in detail. List every distinct visible element and its location." The response is checked manually for evidence that the circle, the line segment, and all 11 right-edge dots are present. This is the commercial-API analog of the Qwen pixel-value reconstruction described in paper_v8 §2.2 and §2.5. The verification responses are stored in `preprocessing_check_<ts>.md` and a one-sentence outcome is logged for paper §2 reporting.

---

## 5. Decoding parameters

Matched to Qwen2.5-VL-32B N = 60 (paper_v8 §2.2) wherever the API allows:

| Parameter | Value | Notes |
|---|---|---|
| temperature | 1.0 | All three APIs accept this |
| top_p | 1.0 | Set explicitly where supported |
| max_output_tokens | 1024 | Sufficient for the brief responses; matches Qwen `max_new_tokens=512` order-of-magnitude |
| seed | `trial_idx * 2 + question_idx` (Q1 = 0, Q2 = 1) | OpenAI: passed via `seed` param; Anthropic/Google: logged only (no public seed API) |
| system prompt | empty / minimal | No task-specific framing |
| tools | none | All tool/function-calling explicitly disabled |
| repetition_penalty | 1.05 (Qwen) → not exposed in any commercial API | Documented as unmatched |

**Reproducibility note.** Because Anthropic and Google APIs do not expose a generation seed, full bit-exact replication is not possible for those two models. The Qwen2.5-VL-32B replication itself showed only ~45% bit-exact agreement across two different machines using identical seeds (paper_v8 §3.1.2). The paper argues for a **distribution-level reproducibility standard** for stochastic VLM psychophysics. The commercial-VLM data are evaluated under the same standard.

---

## 6. Prompts (English, exact paper_v8 §2.2 wording)

**Instruction preamble** (prepended to every Q1 and Q2 turn):

> Look at the image below straight on from the front. Please answer based on your visual impression alone — do not use any tools such as Python, image processing, a ruler, or a protractor to measure.

**Q1 (from top):**

> Estimate which dot on the right edge the line in the circle would hit if extended. Answer like "the Xth dot from the top."

**Q2 (from bottom):**

> Estimate which dot on the right edge the line in the circle would hit if extended. Answer like "the Xth dot from the bottom."

The exact strings above are byte-identical to the Qwen2.5-VL-32B prompts as stored in `n60_progress_a100.json` (key `prompts`). We do **not** append `\boxed{}` instructions; the Qwen runs did not use them, and parsing them was robust. We use the same parsing strategy on the commercial responses.

---

## 7. Protocol: 2-turn independent

Each trial consists of **two independent API calls**, each starting from an empty conversation. The model never sees its previous Q1 answer when responding to Q2.

```
trial 0:
  call A: [image, preamble + Q1] → response_Q1_0
  call B: [image, preamble + Q2] → response_Q2_0

trial 1: (new pair of independent calls)
  ...
```

This matches the Qwen2.5-VL-32B condition exactly. Direct comparison to the §3.1.2 A100 / Spark distributions is valid.

**N = 60 trials per model.** Total API calls: 60 × 2 × 3 = **360 calls**.

---

## 8. Response extraction

Identical to the Qwen N = 60 parser used in `analyze_n60_q1q2.py`:

1. Search response text for a numeric dot value (digits 1–11) with associated direction wording ("from the top" / "from the bottom" / "from top" / "from bottom").
2. If a `\boxed{}` is present, prefer the value inside the box.
3. Fall back to spelled-out ordinals ("first", "second", ..., "tenth", "eleventh") since the Qwen A100 trial 19 used "third dot from the top" without digits.
4. **Direction normalization.** If a Q1 response reports its answer in "from the bottom" form, transform via `value → N_DOTS − value + 1` (N_DOTS = 11). Symmetric for Q2. Per-trial direction compliance is logged so that the rate can be reported in the paper (Qwen Q1 compliance was 57/60 on A100, 59/60 on Spark).
5. **Tool-use detection.** If the response indicates the model executed code, used a measurement tool, or refused the no-tool instruction, the trial is flagged `tool_violation` and excluded from the primary analysis. If the rate exceeds 10% per model, pause and adjust.

---

## 9. Output file: `n60_progress_<model>.json`

Exact same schema as `n60_progress_a100.json`. Per-trial records contain:

```json
{
  "started_at": "2026-05-XX...",
  "model_id": "claude-sonnet-4-5-20250929",
  "stimulus": "./104.jpg",
  "preprocessing": "API-side (Anthropic vision pipeline, image preserved at native resolution per §4 verification)",
  "prompt_set": "human_G1G2_aligned_v1",
  "structure": "2-turn independent (no conversation carry-over, Q1/Q2 only)",
  "machine": "MacBook (or Windows PC) + Anthropic API",
  "n_trials_target": 60,
  "temperature": 1.0,
  "top_p": 1.0,
  "max_new_tokens": 1024,
  "seed_strategy": "seed = trial_idx * 2 + question_idx (Q1=0, Q2=1); not enforced via API for Anthropic/Google",
  "prompts": {
    "Q1": "Look at the image below straight on from the front. Please answer based on your visual impression alone — do not use any tools such as Python, image processing, a ruler, or a protractor to measure.\n\nEstimate which dot on the right edge the line in the circle would hit if extended. Answer like 'the Xth dot from the top.'",
    "Q2": "Look at the image below straight on from the front. Please answer based on your visual impression alone — do not use any tools such as Python, image processing, a ruler, or a protractor to measure.\n\nEstimate which dot on the right edge the line in the circle would hit if extended. Answer like 'the Xth dot from the bottom.'"
  },
  "trials": [
    {
      "trial_idx": 0,
      "Q1": {
        "prompt": "...(same as above)...",
        "response": "...(full raw text returned by API)..."
      },
      "Q2": {
        "prompt": "...",
        "response": "..."
      },
      "elapsed_seconds": 4.7,
      "completed_at": "2026-05-XX..."
    },
    ...
  ]
}
```

**Three output files target:**
- `n60_progress_sonnet45.json`
- `n60_progress_gpt4o.json`
- `n60_progress_gemini25flash.json`

These plug directly into the existing `analyze_n60_q1q2.py` and `plot_qwen25vl_n60_v2_vs_human.py` pipelines.

---

## 10. Sample size

N = 60 per model matches Qwen N = 60 on each machine. With three commercial models, the total is 180 trials and 360 API calls. Statistical power for detecting mean II differences ≥ 0.5 dots between any pair of models, given Qwen-like SDs around 0.6–0.7, is approximately 90% — sufficient to declare "same illusion magnitude" or "different illusion magnitude" relative to the Qwen-A100 reference of II = 4.91.

---

## 11. Execution environment

- **Hardware**: Windows desktop (always-on, dedicated for this run); MacBook also possible.
- **Runtime estimate**: ~5 sec per API call × 360 calls × small retry overhead ≈ **45 minutes to 1 hour**, plus a few minutes for the preprocessing-check phase. Gemini paid tier removes the rate-limit constraint; Anthropic and OpenAI tier 1 limits are far above what this run requires.
- **Dependencies**: `anthropic`, `openai`, `google-genai`, `python-dotenv`, `pandas` — pure Python.
- **API keys**: `.env` file at project root, gitignored. Variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`.

---

## 12. Cost estimate

Per-model figures assume ~1500 input tokens (image + prompt) and ~300 output tokens averaged across 120 calls per model.

| Model | Input tok | Output tok | Input cost | Output cost | Subtotal |
|---|---|---|---|---|---|
| Sonnet 4.5 | 120 × 1500 = 180k | 120 × 300 = 36k | $0.54 | $0.54 | ~$1.1 |
| GPT-4o | same | same | $0.45 | $0.36 | ~$0.8 |
| Gemini 2.5 Flash | same | same | ~$0.01 | ~$0.01 | ~$0.03 |

**Total: ~$2.0 (~300 JPY).** Well within the chosen $10 charge per provider and the $20 spending cap.

---

## 13. Failure handling

- **429 rate limit**: exponential backoff (1, 2, 4, 8, 16 s) up to 5 retries, then mark trial as `rate_limit_failed` and continue.
- **5xx server error**: same backoff, max 3 retries.
- **Refusal / safety filter**: log full response, mark `refused`, do not retry.
- **Tool-use violation**: log, mark `tool_violation`, continue; investigate if rate > 10% per model.
- **Crash mid-run**: partial results are written to disk per-trial. Resuming reads the latest JSON, finds the highest `trial_idx`, and continues from `trial_idx + 1`. This matches how the Qwen `n60_progress_*.json` files were built.

---

## 14. Analysis plan

After all three `n60_progress_<model>.json` are collected, the following are run (mostly reusing existing scripts):

1. **Direction-compliance audit**: per-model rate of correct direction wording on Q1 and Q2. Compare to Qwen 57–59/60.
2. **Q1, Q2, II distribution plots**: extend Figure 2 of paper_v8 (currently Human / Qwen-A100 / Qwen-Spark) to include Sonnet 4.5 / GPT-4o / Gemini 2.5 Flash. Plot script: `plot_qwen25vl_n60_v2_vs_human.py` modified to accept the three new JSONs.
3. **Table 2 extension**: add three rows to Table 2 of paper_v8 with the same measures (Q1 / Q2 / IIQ1 / IIQ2 / II / Q1+Q2 sum, mean ± SD, median, range).
4. **Q1 + Q2 internal-consistency check**: report sum-mean per model; compare to Qwen A100 / Spark (10.62 / 11.17) and human (11.00) — i.e. does each commercial model also pass the "single perceived dot" test that licenses use of II as a per-respondent index?
5. **Variance-compression check**: report II SD per model; the central paper_v8 §3.1.2 finding is that Qwen SD ≈ 1/3.5 of human SD. Replication across vendors would strengthen the "narrow attractor" claim.
6. **Veridical-answer rate**: count trials with Q1 = 1 or Q2 = 11 or II = 0; compare to human 4.5% and Qwen 0/120.

---

## 15. Paper integration

Results will be added to `paper_v8.md` as:

- **New §3.1.3** (or §3.1.2 expansion, pushing the current §3.1.3 Q3 observation to §3.1.4 and Appendix A): "Cross-vendor replication on commercial VLMs". Figure 2 extended with three additional distribution curves; Table 2 extended with three rows.
- **§3.1.1 update**: cross-reference the new §3.1.3 from the exploratory paragraph — "the N = 1 exploratory observations of §3.1.1 are revisited quantitatively (N = 60 each) in §3.1.3 below".
- **§4 / Discussion**: extended to note that the bias is reproduced across three commercial VLMs from three different vendors, strengthening the "VLMs as a class" framing of the §4 interpretation. The §3.2–§3.5 vision-encoder analysis remains specific to Qwen because open weights are needed.

---

## 16. Open questions for review

1. **Seed handling for Anthropic and Google.** OpenAI accepts `seed`; the others do not. Is "T=1.0 stochastic, 60 samples, no API seed" acceptable for those two models? (Recommended: yes — Qwen itself showed only 45% bit-exact match across machines despite identical seeds; the paper's reproducibility standard is distributional.)
2. **System prompt.** Empty for all three APIs? (Recommended: yes; the Qwen runs used the model's default chat template with no system content.)
3. **OpenAI `detail: "high"`.** Triples image token cost but ensures the dots remain resolvable. (Recommended: keep `high`.)
4. **Run order.** Gemini → GPT-4o → Sonnet 4.5 (cheapest / fastest first as a shake-out). (Recommended: yes.)
5. **Preprocessing-check failure threshold.** If any of the three APIs fails to enumerate all 11 right-edge dots in the diagnostic response, do we (a) proceed and document, (b) attempt a higher-resolution version of the stimulus, or (c) drop that model? (Recommended: (a) — paper_v8 §3.1.1 already includes commercial VLMs with imperfect descriptions; the behavioral data are what matter, the preprocessing check is for paper §2 disclosure.)

---

*Prepared 2026-05-25 in collaboration with Claude. To be reviewed by Eiji Watanabe before code implementation.*
