# Research Alpha Agent MVP Implementation Plan

## Summary

Build a local CLI AI/ML idea agent that learns "idea moves" from historical landmark papers in `CCF-A AI venues + ICLR`, maps those moves onto current frontier trends, and outputs structured `Idea Dossier` files that are useful for real project planning.

The MVP is intentionally constrained:

- Scope: `AAAI`, `NeurIPS`, `ACL`, `CVPR`, `ICCV`, `ICML`, `IJCAI`, `ICLR`, plus journals `AI`, `TPAMI`, `IJCV`, `JMLR`
- Data depth: metadata, abstract, citation and award/oral signals only
- Interface: local CLI
- Outputs: Markdown, JSON, HTML

## Core Design

This system is not a generic "auto research agent". It is a:

- quality-weighted idea prior generator
- plus adversarial evaluator

The design came from two observations:

1. Reading all papers does not teach a model what a good top-tier idea looks like.
2. Strong ideas can often be reverse-engineered into reusable patterns.

So the system should learn from strong papers first, then generate and criticize candidate ideas.

## Main Objects

The original thread settled on three core artifacts:

1. `Idea Genome Card`
   A structured reverse-engineering card for one strong paper.

2. `Idea Pattern Library`
   A library of reusable idea patterns mined from many strong papers.

3. `Candidate Idea Dossier`
   A structured proposal for one generated research idea, with novelty, feasibility, value, risk, and first experiments.

## Idea Genome Card

Each landmark paper should be decomposed into a structured card with fields like:

- venue, year, award type, citation profile
- what the field believed before publication
- what bottleneck or hidden assumption the paper attacked
- what problem reframing happened
- what made the idea possible at that time
- what evidence design made the paper convincing
- how the story was organized
- what part of the move is transferable
- where the idea should fail

Important constraint for v1:

- if we only have metadata and abstract, store `evidence_level: abstract_only`
- do not pretend we have full-paper understanding

## Idea Patterns

The original design identified an initial set of reusable patterns:

- hidden assumption reversal
- representation or architecture reframing
- objective or evaluation mismatch
- new tools unlock an old blocked problem
- benchmark or task redefinition
- engineering phenomenon gets theorized
- negative result becomes a boundary theory
- cross-domain structural transfer

Each pattern should carry:

- canonical examples
- preconditions
- migration template
- failure conditions

## End-to-End Flow

The intended flow is:

1. Build a weighted historical corpus.
2. Extract `Idea Genome Cards` from strong papers.
3. Mine `Idea Pattern Library`.
4. Build trend clusters from recent frontier papers.
5. Match `Pattern x Frontier Gap`.
6. Generate candidate ideas.
7. Run novelty, feasibility, why-not, and value critics.
8. Write ranked dossiers.

## Key Modules

### 1. Venue and Source Registry

Responsibilities:

- normalize venue aliases
- define year ranges and categories
- encode source priority
- keep `CCF-A AI` as the core scope
- keep `ICLR` as an explicit extra top-tier venue

Primary sources:

- Semantic Scholar for paper metadata, abstracts, citations
- OpenAlex for bibliographic coverage and fallback
- OpenReview for review-time and acceptance-related signals where available
- arXiv for frontier trend expansion

### 2. Corpus Harvester

Responsibilities:

- fetch paper metadata
- merge identifiers
- deduplicate papers
- cache API responses
- support retries and resume

Recommended command:

```bash
research-alpha harvest --venues all --from 1998 --to 2026
```

Fields to keep:

- title
- abstract
- authors
- year
- venue
- DOI
- arXiv ID
- Semantic Scholar ID
- OpenAlex ID
- citation count
- influential citation count
- URL
- fields of study

### 3. Quality Weight Engine

Each paper gets a `paper_weight`.

Signals proposed in the source thread:

- Best Paper: `+5`
- Outstanding or runner-up: `+4`
- Test-of-Time: `+4`
- Oral: `+2`
- Spotlight: `+1.5`
- citation percentile within same venue and same year:
  - `p99 +3`
  - `p95 +2`
  - `p90 +1`
- influential citation percentile as adoption proxy: `+0 to +2`

Important rule:

- citation signals must be normalized within the same venue and same year

Useful sets:

- `Gold Set`: weighted landmark papers
- `Trend Set`: recent frontier papers for trend analysis

### 4. Trend Cartographer

Responsibilities:

- cluster recent abstracts
- estimate topic growth
- measure concentration and saturation
- build opportunity views

Practical MVP approach:

- `TF-IDF + SVD + MiniBatchKMeans`
- optional LLM-based topic labeling

Output views:

- Topic Velocity
- Award Enrichment
- Saturation Map
- Method-Task-Dataset Graph
- Opportunity Quadrant

Expected output files:

- `outputs/<run_id>/trend_report.md`
- `outputs/<run_id>/opportunity_map.html`

### 5. Idea Genome Analyst

Responsibilities:

- generate structured `Idea Genome Cards` for top-weight papers
- validate card schema
- keep traceability to source paper IDs

### 6. Pattern Miner

Responsibilities:

- aggregate many genome cards into reusable pattern cards
- attach examples and migration rules
- retain confidence and evidence level

### 7. Transfer Engine

Responsibilities:

- align historical patterns with current frontier clusters
- produce candidate ideas that explain:
  - which historical pattern they come from
  - which current frontier gap they map to
  - which bottleneck they aim to break
  - why now is the right time
  - how they differ from prior work

### 8. Prior-Art Hunter

Responsibilities:

- search nearest prior work for each candidate idea
- classify overlap

Suggested labels:

- `duplicate`
- `near_miss`
- `complementary`
- `distant`

Rule:

- `duplicate` should sharply down-rank an idea
- `near_miss` must force an explicit differentiation note in the dossier

### 9. Feasibility, Why-Not, and Value Critics

The source thread converged on three separate critic roles.

Feasibility checks:

- data availability
- compute cost
- baseline reproducibility
- evaluation clarity
- whether a two-week first signal exists

Why-Not checks:

- missing data
- missing model capability
- insufficient compute
- wrong evaluation regime
- cross-domain disconnect
- hidden assumption barrier
- the idea is actually already done

Value checks:

- does it redefine the problem
- does it create a strong top-tier story
- does it affect multiple subfields
- could it yield a benchmark, protocol, or theory contribution

Suggested combined score:

```text
0.20 novelty
+ 0.20 value
+ 0.15 feasibility
+ 0.15 why_now
+ 0.10 trend_support
+ 0.10 story_strength
+ 0.10 defensibility
- penalties
```

### 10. Dossier Writer

Each final idea should produce:

- one Markdown dossier
- one JSON dossier

Expected dossier sections:

- one-line idea
- linked historical pattern
- frontier evidence
- prior-art check
- feasibility
- why-not analysis
- value judgment
- risk summary
- 2/4/8 week experiment plan
- target venue

## CLI Shape

The original thread proposed this CLI surface:

```bash
research-alpha init
research-alpha harvest --venues all --from 1998 --to 2026
research-alpha score
research-alpha trends --frontier-years 5
research-alpha genome build --top-n 500
research-alpha patterns build
research-alpha ideas generate --query "LLM agents for scientific discovery" --n 10
research-alpha run --query "AI agents for research automation" --ideas 5
```

`research-alpha run` should be the single-command MVP path.

## Local Storage

Recommended storage model:

- SQLite for durable local state
- file outputs for reports and dossiers

Suggested tables:

- `papers`
- `paper_ids`
- `quality_signals`
- `topic_clusters`
- `idea_cards`
- `pattern_cards`
- `candidate_ideas`
- `runs`

Suggested output directory:

- `outputs/<run_id>/`

## LLM Integration

Use a provider-agnostic adapter.

Suggested environment variables:

- `RA_LLM_PROVIDER`
- `RA_LLM_MODEL`
- `RA_LLM_API_KEY`

Expected behavior without an LLM key:

- `harvest`, `score`, and `trends` still work
- `genome`, `patterns`, and `ideas generate` fail clearly and early

## Implementation Order

### Phase 1: Project Scaffold

Recommended stack:

- Python 3.11+
- `uv`
- `typer`
- `pydantic`
- `sqlite`
- `httpx`
- `tenacity`
- `pandas`
- `scikit-learn`
- `plotly`
- `rich`
- `jinja2`

Initial config files:

- `configs/venues.yaml`
- `configs/award_signals.yaml`
- `.env.example`

### Phase 2: Data Pipeline

Build connectors for:

- Semantic Scholar
- OpenAlex
- OpenReview
- arXiv

Also implement:

- venue normalization
- ID merge logic
- missing abstract handling
- caching
- retry logic
- resume support

Practical rule:

- first run the pipeline on a capped corpus, for example `--max-papers 2000`
- only scale up after the mini pipeline is stable

### Phase 3: Scoring and Trends

Implement:

- citation percentile scoring
- award and oral weighting
- recent abstract clustering
- trend report
- HTML visualizations

This is the first phase that should feel useful even without idea generation.

### Phase 4: Genome and Pattern Library

Implement:

- LLM-driven structured genome cards
- schema validation
- retry on invalid JSON
- pattern aggregation with examples and transfer rules

### Phase 5: Idea Generation and Evaluation

Implement:

- query-conditioned frontier selection
- `Pattern x Frontier Gap` candidate generation
- prior-art checks
- feasibility critic
- why-not critic
- value reviewer
- dossier ranking and writing

### Phase 6: Usable MVP

The acceptance bar defined in the source thread:

```bash
research-alpha run --query "AI agents for research automation" --ideas 5
```

Should produce at least:

- one `trend_report.md`
- one `opportunity_map.html`
- five dossier files
- each dossier with trend evidence, historical pattern, prior-art risk, feasibility, why-not, value, and 2/4/8 week plan

## Tests

### Unit Tests

- venue alias normalization
- DOI/arXiv/title deduplication
- citation percentile calculation
- paper weight calculation
- topic clustering stability
- LLM JSON schema parsing
- prior-art overlap classification

### Integration Tests

Run a mock mini pipeline over a small corpus with:

- award papers
- high citation papers
- ordinary papers
- missing abstracts
- duplicate records

Verify:

- CLI runs end to end
- trend report is produced
- HTML plot is produced
- dossier files are produced

### Golden Evaluation

Use a hand-picked set of classic AI/ML papers and check whether:

- genome cards recover the right idea moves
- generated ideas include real prior-art references
- critics return non-empty feasibility and why-not reasoning

## Assumptions

- In this project, `CCF-A 全量` means the CCF AI A-list plus `ICLR`, not every CCF A-list field.
- Historical corpus default range is `1998` to current year.
- Frontier trend default window is the most recent `5` years.
- v1 does not fetch PDFs or full text.
- v1 may rely on a manually maintained `award_signals.yaml` when oral and award metadata are inconsistent across sources.

## Practical Build Guidance

The source thread also landed on a useful product rule:

- do not begin with "all papers from the last decade"
- begin with a smaller but cleaner weighted corpus
- establish labeling taste first
- then scale

That means we should likely start with:

- one narrow but strong venue registry
- a small seeded corpus
- a few hand-checked genome cards
- an end-to-end command that works on a mini dataset

This will keep the project from turning into a giant paper dump before it becomes a usable idea engine.
