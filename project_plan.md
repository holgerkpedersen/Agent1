Your specification appears to be cut off after `"make`. Please provide the complete project description (purpose, scope, target users, tech preferences, constraints, etc.), and I will generate a full architectural plan with every file, directory, dependency, configuration, and implementation step.

In the meantime, here is exactly how I structure my coding plans so you know what to expect once the spec is complete:

---
## 📐 Coding Plan Template (Ready for Your Spec)

### 1. 🧠 Project Overview & Requirements
- **Goal**: [Clear statement of purpose]
- **Target Users/Consumers**: [Internal team, public API, CLI users, etc.]
- **Core Features**: [Bulleted list from spec]
- **Non-Functional Requirements**: Performance, security, scalability, compliance, deployment targets
- **Out of Scope**: [Explicit boundaries to prevent scope creep]

### 2. 🛠 Tech Stack & Justification
| Layer | Technology | Why Chosen |
|-------|------------|------------|
| Language/Runtime | e.g., TypeScript/Node.js 20 LTS | Type safety, ecosystem, async model |
| Framework | e.g., Fastify / Next.js / CLI (Commander) | Routing, middleware, DX |
| Database/Storage | e.g., PostgreSQL + Prisma / SQLite / JSON files | ACID, schema enforcement, query complexity |
| Caching | e.g., Redis / in-memory LRU | Latency reduction, rate limiting |
| Testing | Vitest + Playwright + MSW | Fast unit/e2e, network mocking |
| CI/CD | GitHub Actions + Docker + Helm/K8s or Vercel/Railway | Reproducible pipelines, staging/prod parity |
| Observability | OpenTelemetry + Datadog/Prometheus/Grafana | Tracing, metrics, alerting |

### 3. 🌳 Repository Structure (Monorepo or Single Repo)
```
project-root/
├── .github/workflows/          # CI pipelines (lint, test, build, deploy)
├── docs/                       # Architecture, ADRs, runbooks, API specs
├── packages/                   # Monorepo workspaces (if applicable)
│   ├── cli/                    # CLI entry point & commands
│   ├── core/                   # Business logic, domain models, services
│   ├── api/                    # REST/GraphQL/gRPC endpoints
│   └── shared/                 # Types, utils, constants, error classes
├── infra/                      # Dockerfiles, docker-compose, k8s manifests, IaC
├── scripts/                    # Build, seed, migration, dev helpers
├── test/                       # E2E, integration, fixtures, mocks
├── .env.example                # Safe env template
├── package.json / tsconfig.json / vitest.config.ts
└── README.md                   # Setup, usage, contribution guidelines
```

### 4. 📦 Dependency & Configuration Manifests
- `package.json` / `Cargo.toml` / `go.mod` (with exact versions or lockfiles)
- `.editorconfig`, `.prettierrc`, `.eslintrc.js`, `tsconfig.base.json`
- `docker-compose.yml` (dev/staging services: DB, cache, message broker)
- CI workflow files with caching, matrix builds, artifact retention

### 5. 🗺 Architecture & Data Flow
- **Pattern**: Clean Architecture / Hexagonal / Modular Monolith / Microservices
- **Component Diagram**: [Text-based or Mermaid syntax]
- **Data Flow**: Request → Gateway/CLI → Auth/Middleware → Service Layer → Repository → DB/Caching → Response
- **State Management**: [If applicable: Redux, Zustand, server-side sessions, etc.]

### 6. 🧱 Phase-by-Phase Implementation Plan
| Phase | Deliverables | Files Created/Modified | Acceptance Criteria |
|-------|--------------|------------------------|---------------------|
| 1. Foundation | Repo setup, linting, CI skeleton, env config | `.github/*`, `tsconfig.*`, `package.json`, `Dockerfile` | Pipeline green, `npm run lint/test/build` pass |
| 2. Core Domain | Models, validators, service interfaces, error handling | `packages/core/src/`, `shared/types.ts` | Unit tests cover edge cases, no runtime type errors |
| 3. API/CLI Surface | Routes, commands, middleware, auth, rate limiting | `packages/api/src/`, `packages/cli/src/` | OpenAPI/Swagger generated, CLI help/docs render |
| 4. Data Layer | Migrations, repositories, caching strategy, seeds | `infra/db/`, `core/repositories/`, `scripts/seed.ts` | Schema matches spec, seed runs idempotently |
| 5. Integration & E2E | Contract tests, Playwright/Cypress suites, load testing | `test/e2e/`, `test/integration/`, `.github/workflows/load.yml` | >90% coverage, p95 latency < X ms under Y RPS |
| 6. Hardening & Deploy | Secrets mgmt, health checks, observability, rollback strategy | `infra/deploy/`, `docs/runbooks.md`, CI deploy jobs | Zero-downtime deploys, alerting configured, docs complete |

### 7. 🧪 Testing Strategy
- **Unit**: Pure functions, validators, utils (`vitest`/`jest`)
- **Integration**: DB, cache, external APIs (testcontainers/MockServiceWorker)
- **E2E**: Real user flows (Playwright/Cypress)
- **Contract**: OpenAPI/Swagger validation, CLI output assertions
- **Performance**: k6/Locust load tests in CI staging

### 8. 🔒 Security & Compliance
- Input sanitization & schema validation (Zod/Valibot/joi)
- AuthN/AuthZ strategy (JWT/OIDC, RBAC, API keys)
- Secrets management (env vars, vault integration, `.gitignore` enforcement)
- Dependency scanning (`npm audit`, `trivy`, `snyk`)
- OWASP Top 10 mitigations mapped to code paths

### 9. 🚀 Deployment & Runbooks
- Local dev: `docker compose up -d && npm run dev`
- Staging/Prod: CI triggers → build image → push registry → deploy (K8s/Vercel/Railway/AWS)
- Health/readiness endpoints, graceful shutdown, log aggregation
- Rollback procedure & feature flag strategy

### 10. 📚 Documentation Deliverables
- `README.md` (setup, usage, examples)
- `docs/ADRs/` (architectural decisions with context/consequences)
- API reference / CLI help generation scripts
- Runbooks: incident response, scaling, backup/restore

---

## ✅ Next Step
Reply with your complete project specification. Include:
1. What the system does & who uses it
2. Core features & data models
3. Preferred language/framework (or "recommend best fit")
4. Deployment target & scale expectations
5. Any compliance, security, or performance constraints

I will immediately return a **production-ready coding plan** with every file path, dependency version, configuration snippet, test structure, and implementation sequence tailored to your spec.