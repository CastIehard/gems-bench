# Baselines

Measured runs, one directory per system. They are reference points, not targets:
a score depends on the model, the architecture around it and the hardware, so a
number here says what *that* system did on GEMS-Bench and nothing about yours.

Each directory holds

| File | What it is |
|---|---|
| `results.json` | per-question verdict, answer, latency and tool metrics — one entry per mode |
| `benchmark_run.json` | the run index: which `run_id`/`session_id` produced each answer |
| `summary.md` / `summary.csv` | the aggregate table `analyze.py` prints |
| `settings.yaml` | the driver settings the run used |

Plots are not committed. `analyze.py` regenerates all of them from
`results.json`:

```bash
cp baselines/<system>/results.json results/results.json
python analyze.py
```

The answer recordings are not committed either — 200 WAVs per run. `results.json`
carries the transcript that was graded, which is what the verdict rests on.

## gpt-realtime-1.5 (plain Azure OpenAI Realtime)

One realtime session per question, server VAD owning the turn boundary, the two
GEMS tools registered, no orchestration around the model. Produced with
[`driver/azure_realtime.py`](../driver/azure_realtime.py) and the settings in
`gpt-realtime-1.5/settings.yaml`; the prompt is `prompts.yaml` unchanged.

Both audio channels were run as separate modes, `baseline_synthetic` and
`baseline_real`, over all 100 questions. Each item records the channel it was
asked on in `audio_source`.

| | synthesized | human-read |
|---|---|---|
| accuracy | 41/100 | 37/100 |
| TTFA median | 4.33 s | 4.34 s |
| tool calls, after end of speech | 369 | 375 |
| tool calls, during listening | 0 | 10 |
| turns split by server VAD | 1/100 | 29/100 |

| category | synthesized | human-read |
|---|---|---|
| `one_hop` | 95 % | 80 % |
| `serial` | 5 % | 5 % |
| `early` | 5 % | 5 % |
| `select` | 65 % | 50 % |
| `combined` | 35 % | 45 % |

**The channel gap is not significant.** Paired over the same 100 questions: 29
correct on both, 51 wrong on both, 12 only on synthesized, 8 only on human-read.
Exact McNemar p = 0.50. TTFA matches to 0.04 s (paired median).

**`serial` and `early` collapse to 5 %.** Not a scoring artifact. Both need a
depth-3/4 chain, and the model typically stops after two tool calls: it reads the
product's own supplier out of `product_lookup` and answers from there. For half
the products that supplier differs from the brand's supplier
(`schema.marke_override_rate`), which is exactly the shortcut the graph is built
to punish, so the chain has to be walked rather than guessed.

**A stalled turn is a real result, not a missing measurement.** On the fan-out
categories the model sometimes looks up two of the products, says it will
continue, and ends the turn. Nothing in a plain session triggers a further
response, so the answer stays incomplete and scores incorrect. That is the
configuration behaving as configured.

**Two items were re-run** after an interrupted turn: one hit an Azure
`first_output_timeout` (a server-side failure, not a model failure), one was
answered before the question finished and is described above. Both were re-run
with the same configuration under the same label; both then scored incorrect on
their merits, so the headline number is unchanged either way.
