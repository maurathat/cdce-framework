# CDCE Compression Harness

**Recursive compression experiment testing whether LLMs under token budget constraints converge on stable algebraic attractors.**

This harness implements the Compositional Decentralized Compression Emergence (CDCE) framework: a systematic test of whether AI reasoning, when forced through progressively tighter token budgets, exhibits universal compression signatures independent of model substrate.

## Experimental Results

| Metric | Value |
|--------|-------|
| Stored geometries | 992 |
| Compression generations | 38 |
| Models tested | 6 (Claude, Haiku, GPT-4o, GPT-5.5, Gemini Flash, Gemini Pro) |
| Task families | 3 (Optimization, Prediction, Translation) |
| Round-trip structural fidelity | 91% |
| Budget levels | 2000 / 1000 / 500 / 250 / 125 tokens |

## CDCE Predictions Tested

1. **P1 — Monotonic operator reduction**: Distinct reasoning operations decrease as budget decreases
2. **P2 — Plateau stages**: Power-law decay with phase transitions at critical budgets
3. **P3 — Cross-family convergence**: Strategies for optimization, prediction, and translation converge under compression
4. **P4 — Substrate independence**: Pattern is consistent across Claude, GPT-4o, Gemini (different model architectures)

## Architecture

```
LLM Response (strategy text)
    │
    ▼
compute_hash() ─── UOR canonical κ-label (sha256:<64-hex>)
    │                via uor-addr JCS+NFC+SHA-256
    ▼
MemoryEntry
    ├── content_hash (UOR address)
    ├── model, task_id, task_family
    ├── generation (compression cycle count)
    ├── lineage (chain of ancestor addresses)
    └── strategy_text (the compressed geometry)
    │
    ▼
MemoryStore.store()
    ├── Write: memory/{addr}.json
    ├── Index: by_model, by_task
    └── Detect: cross-model convergence
    │
    ▼
Next compression round (feed prior strategy back)
```

### UOR Content Addressing

Compressed geometries are canonically addressed via the [uor-addr](https://github.com/UOR-Foundation/uor-addr) crate. Each strategy is wrapped in a JSON envelope and passed through the JSON realization pipeline (RFC 8785 JCS + Unicode NFC + SHA-256), producing a deterministic 71-byte `sha256:...` label. This ensures identical strategies produce identical addresses regardless of whitespace, key ordering, or Unicode normalization differences.

### Decompression Round-Trip Test

Tests whether compressed strategies are stable attractors:

1. **Self-decompression**: Model A compresses, Model A decompresses
2. **Cross-model decompression**: Model A compresses, Model B decompresses
3. **Round-trip**: Compress → Decompress → Recompress. Hash match = fixed point.

Structural fidelity is measured via verb-set Jaccard similarity between original and recompressed strategies.

## Setup

```bash
# 1. Clone and set up
git clone https://github.com/maurathat/cdce-framework.git
cd cdce-framework
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API keys
cp .env.example .env
# Edit .env with your keys

# 4. Run full compression cycle
python run.py

# 5. Run decompression round-trip test
python run.py --decompress

# 6. Quick mode (2 budget levels only)
python run.py --quick
```

## API Keys

- `ANTHROPIC_API_KEY` — Claude Sonnet, Haiku
- `OPENAI_API_KEY` — GPT-4o, GPT-5.5
- `GOOGLE_API_KEY` — Gemini Flash

At minimum you need one. The harness runs whichever models have keys.

## Project Structure

```
src/
  config.py         # Model configs, budget levels, task families
  memory.py         # UOR content-addressed storage, convergence detection
  tasks.py          # 9 tasks across 3 families
  orchestrator.py   # Main experiment loop
  decompress.py     # Round-trip attractor stability test
  llm_clients.py    # Unified API wrapper (Anthropic, OpenAI, Google)
  metrics.py        # Operator extraction, Jaccard convergence
  energy.py         # Thermodynamic efficiency metrics
  visualize.py      # Plot generation

memory/samples/     # 30 representative geometries (see Dataset below)
results/            # Experiment JSONs and plots
```

## Dataset

The `memory/samples/` directory contains 30 representative compressed geometries spanning:
- All 6 models
- Generations 1 through 38
- All 3 task families
- Entries with deep lineage chains showing compression evolution

> **Full dataset** (992 geometries) available on request.

## Output

Results are saved to `results/` as JSON files with full run data, plus PNG plots:
- `operator_reduction.png` — Operator count vs budget level per model
- `convergence.png` — Cross-family Jaccard similarity under compression
- `reuse_ratio.png` — Operation reuse patterns
- `dashboard.png` — Combined overview

## Dependencies

- Python 3.10+
- [uor-addr](https://pypi.org/project/uor-addr/) — UOR canonical content addressing
- anthropic, openai, google-genai — LLM API clients
- numpy, matplotlib — Numerics and plotting

## Related Work

- [UOR Foundation / uor-addr](https://github.com/UOR-Foundation/uor-addr) — Typed content-addressing
- [UOR Foundation / prism](https://github.com/UOR-Foundation/prism) — UOR standard library
- [UOR Foundation / atlas-embeddings](https://github.com/UOR-Foundation/atlas-embeddings) — E8 root system constructions

## License

MIT
