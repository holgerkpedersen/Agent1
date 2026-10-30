# Prospector-strategi i Agent1-harnessen — plan og sikkerhedsargumentation

**Dato:** 2026-10-30 · **Kilde:** arXiv:2610.37121 (LAMM/MIT; PI Markus J.
Buehler), HF: `lamm-mit/Prospector` (LoRA-adapters på `Qwen/Qwen3.5-4B`,
Apache-2.0), kode: github.com/lamm-mit/Prospector · **Målmodel:**
`google/gemma-4-E4B-it` (native function calling).

**Verifikationsstatus (læs først):** Papiret, HF-repoet og GitHub-repoet er
bekræftet, og gate-reglerne i afsnit 0 er verificeret mod både paper-abstract,
HF-modelkort og GitHub (`agent/gate.py`: "cites at least one edge, every cited
edge was seen, and it names a node of the cited chain: set membership, with no
judge"). **Målmodellens eksakte navn og modelkort-citat er derimod IKKE
genverificeret** (web-søgning var utilgængelig under denne gennemgang), og
førsteforfatterens navn er ikke verificeret. Planen er derfor udformet så den
**ikke afhænger af modelnavnet**: fase 0 måler på den lokalt tilgængelige
function-calling-model, og fase 4's modelprofil er parametriseret. Bekræft
modelkortet før fase 4 sættes i drift.

## 0. Hvad papiret faktisk siger (verificeret, se status ovenfor)

1. Prospector er en **traverseringspolitik**: den kalder tre værktøjer ét ad
   gangen — `find(name)`, `expand(node, relation?, offset?)`,
   `connect(a, b, max_hops?)` — og bygger kontekst ud fra hvad den allerede
   har fundet. Budget: 20 tool-kald, 1.024 tokens/turn.
2. **Gaten** er streng sætmedlemskab uden judge: et svar accepteres kun hvis
   (a) mindst én kant citeres, (b) hver citeret kant *faktisk er returneret
   af et værktøj* ("seen"-sættet), (c) svaret nævner en node i den citerede
   kæde. Målet er "gated success" (gate OK + svar nævner guld-label).
3. **Nøgletal (model card, 8 frø):** Prospector-4B semiconductor 0.839±0.01;
   zero-shot transfer: bio-inspirerede materialer 0.803, PFAS 0.766,
   materials properties 0.809. 2B: 0.755; 0.8B: 0.694.
4. **Træningsrecept (valgfri):** SFT på teacher-demo → GRPO over hele
   episoder → blandet SFT → to decision-level GRPO-runder + verifier.
5. Latent tilstand aflæser næste træk (follow/backtrack/answer) og kan
   styres.

**Vigtig distinktion:** Papirets kernegevinst (selektion *under* ræsonnement
+ verificerbar beviskæde) er **strategi, ikke vægte**. Den kan implementeres
uoptrænet i et harness som Agent1. Finetuning er trin 5 og valgfrit.

## 1. Hvorfor det passer til Agent1 (kode-verificeret)

| Prospector-princip | Agent1-komponent (eksisterende) |
|---|---|
| Episode-loop med sekventielle tool-kald | `agent_core/llm/tool_loop.py::ToolLoopRunner` (max_iterations, deadline, compaction, tracing) |
| Værktøjsschemas til modellen | `agent_core/tool_schemas.py::NLP_TOOL_SCHEMAS`, native OpenAI-format |
| Lokal model med function calling | `agent_core/llm/llama_provider.py::LlamaProvider` (LM Studio/llama.cpp, `_build_payload(messages, tools, ...)`), `model_profiles.py` |
| "Seen"-sæt / sporbarhed | `harnessfix/tracing.py` — JSONL per task med `tool_call`/`tool_result`/`llm_response`-events (beslutning #033: `reports/traces/`, gitignored) |
| Eksterne datakildefacader uden kerneændring | `agent_core/mcp/manager.py` — MCP-servere eksponerer tools til hovedloopet |
| Måling på faste scenarier | `agentic_bench.py` — kører RIGTIGE main-loop-tools, scorer pr. run |

Konklusion: harnessen mangler kun (a) et graf-facade og (b) en gate. Alt
andet er genbrug.

## 2. Planen (6 faser, ingen kerne-ødelæggelse)

### Fase 0 — Etabling af baseline (måling før ændring)
- Vælg et lokalt KG-datasæt (fx Papirets `tsmc_5b10p.graphml` fra
  HF-repoet, Apache-2.0, eller en egen GraphML/JSON).
- Kør 3 betingelser på ~30 spørgsmål via `agentic_bench`-mønsteret:
  (A) Gemma-4-E4B lukket bog, (B) E4B + præselekteret subgraf (RAG),
  (C) E4B + traverseringsværktøjer. Registrér andel besvarelser med
  citerte kanter og andel gate-beståede.
- **Artefakter:** `benchmarks/prospector_baseline.md` (rapporten; selve
  datasættet downloades lokalt og holdes uden for version control).

### Fase 1 — Graf-facade som MCP-server (`graphkg`)
- Implementér `find/expand/connect` som **MCP-tools** via den eksisterende
  `McpManager`-infrastruktur. Ingen ændring i ToolLoopRunner, ingen nye
  native tools → samspil med Kanban/dashboard/MCP-status inspiceres (fase 5).
- Samme grænser som papiret: find ≤10 noder, expand ≤40 kanter (>150
  forbindelser → relationssummary), connect ≤3 stier.
- Facaden returnerer **kanter med stabile ID'er** (`e:<hash>`), så gate og
  "seen"-sæt kan arbejde på sætmedslemskab, ikke strenghed.

### Fase 2 — Episodeloop + "seen"-registrering
- Genbrug `ToolLoopRunner` uændret som loop; kørslen har allerede TraceSink
  til `reports/traces/`.
- Ny lille komponent `agent_core/prospector/seen.py`: udled "seen"-mængden af
  **værktøjsresultaterne i sporet** (maskinelæsbar: facaden emitterer kanter
  som JSON i tool-responsen). Alene værktøjsresultater er sandhedskilden —
  modellens egen påstand om en kant tæller ikke.

### Fase 3 — Gaten (deterministisk, ingen judge)
- `agent_core/prospector/gate.py`: parser final-svar-format
  (`ANSWER: ... KÆDE: e1,e2,...`), verificerer: ≥1 kant; alle kanter ∈ seen;
  svaret nævner en node i kæden. Normalisering som papiret (Unicode, case,
  whitespace).
- Gate-afvisning er **ikke** en fejl: loopet får ét omstartforsøg med
  præcis besked om hvilke citater der fejlede (budget: maks 2 forsøg), ellers
  leveres svaret markeret `GATE: FAIL` — aldrig som verificeret.

### Fase 4 — Prompt og profil for Gemma-4-E4B
- Systemprompt med de tre værktøjers signaturer + eksempel-episode (fra
  papirets prompt-ordlyst, HF-repo `agent/prompts.py` kan studeres som
  reference — Apache-2.0).
- Modelprofil via `model_profiles.py`: E4B, temp lav, maks 20 tool-kald pr.
  episode (harness' `max_iterations` sat pr. scenarie, ikke globalt).

### Fase 5 — Måling og rapport
- Kør A/B/C igen + gate-metrik; sammenlign med papirets tal som referenceramme
  (ikke som mål: andet datasæt/andre spørgsmål).
- Tjek integrationer: MCP-status, dashboard, `agentic_bench`, Kanban-kørsler
  (uvirksomme ruter må ikke ændre adfærd for eksisterende tools).

### Fase 6 (valgfri) — Finetuning af E4B (SFT/GRPO-recept)
- Træningsdata genereres naturligt: **sporene fra fase 0–5 er træningsrækker**
  (episoder med gate-beståede svar = positive demoer; gate-fejlede = hard
  negatives). Format: ChatML-turper med tool-kald — gemmes som JSONL i
  `reports/`-træet (gitignored, jf. beslutning #033).
- Kør uden for harnessen (llama-factory/TRL-lignende, LoRA på E4B,
  quantiseret bagefter til GGUF). **Altid ny adapter, aldrig overskriv
  base-vægte.**
- Evaluer A/B mod ufinetunet E4B på holdout; tag kun adapteren i brug hvis
  gate-metrikken forbedres.

## 3. Hvorfor planen er sikker (argumenter pr. lag)

**A. Ingen eksisterende adfærd ændres.** Alle nye filer er additive:
MCP-server `graphkg` leverer tools via den eksisterende manager; gate og seen
er nye moduler i `agent_core/prospector/`, som kun aktivere i dedikerede
prospector-kørsler (opt-in kommando/profil). ToolLoopRunner, NLP_TOOL_SCHEMAS
og eksisterende tool-ruter forbliver bit-identiske — verificeret af at de
eksisterende 209 testfiler i `tests/` kører grønne før og efter (CI-krav:
fuld pytest + ruff + mypy før commit/push).

**B. Deterministisk verifikation, ingen judge.** Gaten er sætmedslemskab mod
værktøjets egne returnerede kanter. En model kan ikke "lyve sig til" et
verificeret svar: citerer den en kant der aldrig blev returneret, fejler
gaten. Det er papirets eget design (ingen LLM-judge) — vi kopierer netop den
del der *fjerner* tillid til modellens selvrapportering.

**C. Indlæst data er ufindt indhold.** Graf-noder/kanter er eksternt
indhold som alle andre datafiler; de vises som værktøjsresultater, aldrig som
instruktioner. Facaden serialiserer kanter struktureret (felter, ikke fri
tekst i system-rollen), og prospector-episoder kører med samme
tool-gating/plan-mode som resten af harnessen.

**D. Ressource-grænser arves.** ToolLoopRunner har i forvejen iteration loft,
deadline-window, no-mutation-detektion og tvunget final-svar; episoder får
papirets 20 kald. Ingen uendelige løkker, ingen ukontrolleret token-vækst
(compaction er allerede bygget ind).

**E. Sporbarhed gratis.** Hver episode er i forvejen en JSONL-trace med
tool_kald/resultater; gate-afgørelsen skrives som én ekstra hændelse i
samme spor. Reproducerbarhed: task_id + correlation-id eksisterer.

**F. Model-sikkerhed.** Gemma-4-E4B er lokal (LM Studio/llama.cpp) — data
forlader ikke værten; det matcher papirets privacy-first-præmis. Finetuning
(fase 6) er valgfri, reversibel (adapter), og tagges kun i brug efter A/B.

**G. Integrations-tjek (vaner).** Før push: dokumentation opdateret
(README/USAGE-afsnit om prospector-tilstanden), CI-grøn, MCP/Kanban-inspektion
(fase 5). Emojis undgås i alle tekstfiler (beslutning #079 — brug `[GATE OK]`
/`[GATE FAIL]`).

## 3.1 Hvad "helt sikker" kan og ikke kan betyde

"Fuldstændig sikker" findes ikke for en plan der involverer en sprogmodel.
Det, der kan gøres sikkert, og som er gjort sikkert her, er
**fejltilstandene**. Konkret, med verificerede mekanismer:

| Fejltilstand | Hvorfor den ikke kan give skade | Verificeret i |
|---|---|---|
| Modellen finder på en kant | Gaten er sætmedlemskab mod "seen"; en ikke-returneret kant kan ikke passere | `agent/gate.py`-regel + planens fase 3 |
| Modellen løber i ring | `max_iterations`, `deadline_window`, `no_mutation_limit`, `force_after_no_mutation` findes allerede som konstruktørparametre og håndhæves af loopet | `agent_core/llm/tool_loop.py:685` (`ToolLoopRunner.__init__`) |
| Modellen muterer repoet utilsigtet | Prospector-episoder kører read-only over grafen; ingen skrive-tools i værktøjssættet | Fase 1 (kun `find`/`expand`/`connect`) |
| Kontekst eksploderer | Compaction og char-budget er indbygget (`_ensure_char_budget`, `_compact_messages`) | `tool_loop.py:1710`, `:416` |
| Tavs fejl uden spor | Hver episode er en JSONL-trace; gate-afgørelsen bliver en ekstra hændelse | `harnessfix/tracing.py` (`KIND_TOOL_CALL`/`KIND_TOOL_RESULT`) |
| Dataindhold instruerer modellen | Grafnoder er værktøjsresultater, aldrig systeminstruktioner; tool-gating/plan-mode gælder | Samme model som resten af harnessen |
| Data lækker ud | Lokal model (LM Studio/llama.cpp); intet forlader værten | `agent_core/llm/llama_provider.py:151` |
| Finetuning ødelægger basismodellen | LoRA-adapter, ny adapter pr. forsøg, aldrig overskrevne base-vægte, A/B før ibrugtagning | Fase 6 |

**Det, der IKKE er garanteret** (og som planen derfor måler i fase 0 frem for
at antage): at E4B's tool-kald-kvalitet er høj nok på store grafer, og at
gate-metrikken bliver brugbar. Begge er åbne spørgsmål der besvares af måling
— ikke af argumentation. Derfor er det mest sikre træk i planen rækkefølgen:
**baseline først (fase 0), gate før model-ambitioner (fase 3), finetuning
sidst og valgfrit (fase 6).**

## 4. Risici og modtræk

| Risiko | Modtræk |
|---|---|
| E4B's tool-kald-kvalitet er lavere end Qwen3.5 på store grafer | Måling i fase 0 (A/B/C) afslører det før nogen investering; recepten (fase 6) er netop designet til at lukke hullet |
| Facadens kanter ændrer sig mellem kald (hash-drift) | Kanter er indholds-hashede og graph-facaden er read-only pr. episode |
| Større grafer end hukommelse | expand's paginering + relationssummary (papirets grænser), GraphML → NetworkX/SQLite-index |
| Gate for striks (sandt svar uden node-nævn) | Følg papirets metrik 1:1; justér kun med dokumenteret fravigelse |

## 5. Estimeret omfang

- Fase 0–2: ~2–3 arbejdsdage (MCP-server + seen, genbrug loop).
- Fase 3–5: ~1–2 dage (gate + profil + bench).
- Fase 6: separat projekt (GPU/eksperimenter), uden for harnessen.

## 6. Referencer (se verifikationsstatus øverst for hvad der er genverificeret)

- arXiv:2610.37121 — abstract bekræftet via arxiv.org.
- HF `lamm-mit/Prospector` — model card med resultattabel, LoRA på Qwen3.5,
  Apache-2.0, data (GraphML) og trænede rækker følger repoet.
- github.com/lamm-mit/Prospector — README: værktøjssignaturer, gate-regler,
  `gated_acc`-metrik, trainingsrecept (SFT → GRPO → blandet SFT → 2× decision-GRPO).
- HF `google/gemma-4-E4B-it` — "Function Calling — Native support for
  structured tool use, enabling agentic workflows" (model card). **Ikke
  genverificeret i denne gennemgang — se verifikationsstatus øverst.**
