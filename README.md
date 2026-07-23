# Agent1

LM Studio-powered autonomous agent for code analysis, planning, and implementation.

## What it does

Agent1 is an interactive CLI agent that uses LM Studio (local LLM) to:

- **Analyze** codebases for bugs, circular imports, and architectural issues
- **Plan** refactoring steps prioritized by impact
- **Extract** shared entities to avoid circular dependencies
- **Generate** task plans with dependency ordering
- **Implement** files automatically, verifying compilation

It works as an **iterative improvement loop**: `workflow → implement → workflow → ...` until the LLM reports no issues.

## Commands

```
workflow <target> [--from spec.md] [--force] [--workspace <path>]
```

Runs the full pipeline: analyze → plan → entities → taskplan.

- **Brownfield**: `workflow .` (analyzes existing .py files)
- **Greenfield**: `workflow . --from spec.md --workspace /c/Dev/newproject`

```
implement <taskplan.md> [--keep] [--force] [--workspace <path>]
```

Implements files from a task plan in batches, compiling each .py file.

- `--keep`: Skip existing+compiling files (except the analyzed file)
- `--force`: Regenerate all files
- `--workspace`: Target directory for generated files

Individual commands: `analyze`, `plan`, `entities`, `taskplan`

## Requirements

- [LM Studio](https://lmstudio.ai/) running locally with an API server (default: `localhost:1234`)
- Python 3.10+
- Token limit should be high enough for your use case (default: 50000)

## Setup

```bash
git clone https://github.com/holgerkpedersen/Agent1.git
cd Agent1

# Create .env with LM Studio URL (optional, defaults to localhost:1234)
echo "LMSTUDIO_URL=http://localhost:1234/v1" > .env

# Start LM Studio, then:
python agent.py
```

## Pros

- No API keys or cloud services needed - runs entirely local
- Iterative: each pass improves the code until no issues remain
- Compilation verified at every step
- Greenfield support via specification file
- Cache avoids redundant LLM calls
- All generated files are tracked and compilable

## Cons

- Depends on LM Studio being running and responsive
- Large files may cause token limit truncation
- `urllib` used instead of async HTTP (blocking in event loop)
- Path handling assumes Git Bash on Windows (`/c/...` format)
- Timeout handling could be more robust
- No incremental saving - if interrupted, partial results are lost

## Hints

1. **Start small**: `analyze agent.py` first, then gradually expand to the full project
2. **Use `--keep`**: Avoids regenerating files that already compile correctly
3. **Use `--workspace`**: Keep generated files separate from the agent's own code
4. **Check `output.md`**: The analysis file tells you exactly what needs fixing
5. **Greenfield projects**: Write a `spec.md` describing what you want, then:
   ```
   workflow . --from spec.md --workspace /c/Dev/myproject
   implement ... -workspace /c/Dev/myproject --force
   ```
6. **LM Studio timeout**: Close and reopen LM Studio if requests hang
7. **Cache**: `.implement_cache.json` saves file lists between runs
