Since your specification was cut off after `"make`, I’ve generated a **production-ready task plan** structured strictly in dependency order. This follows the architectural flow you outlined (`shared ← domain ← application ← infrastructure ← interface`) and can be executed sequentially without circular imports or missing references.

Replace placeholder entity names (e.g., `User`, `Post`) with your actual domains once you share the full spec. The file paths, dependency chains, and validation steps remain identical.

---
## 📋 Dependency-Ordered Task Plan

### 🔁 Dependency Flow Rule
```
shared/ (types, errors, constants) 
   ↑
domain/ (models, repository interfaces) 
   ↑
application/ (use-cases, service interfaces) 
   ↑
infrastructure/ (DB adapters, external clients, DI bindings) 
   ↑
interface/ (controllers, routes, CLI commands) 
   ↑
bootstrap/ (config, app wiring, entrypoint)
```
**Build Order:** Always create leaf nodes first. No file may be created until all its imports exist.

---

### 🟢 Phase 1: Shared Contracts (Zero Dependencies)
*Purpose:* Define pure types, error bases, and constants. No business logic or framework imports.

| # | File Path | Dependencies | Implementation Steps | Acceptance Criteria |
|---|-----------|--------------|----------------------|---------------------|
| 1.1 | `packages/shared/src/types/common.ts` | None | Define `ApiResponse<T>`, `PaginatedResult<T>`, `IdType`, `Timestamps` | Exports only interfaces/types; compiles with strict TS |
| 1.2 | `packages/shared/src/errors/base-error.ts` | None | Create `AppError extends Error` with `code`, `statusCode`, `isOperational` | Instance check works: `err instanceof AppError` |
| 1.3 | `packages/shared/src/errors/http-errors.ts` | `base-error.ts` | Extend for `NotFoundError`, `ValidationError`, `UnauthorizedError` | All throw correct HTTP status & structured JSON |
| 1.4 | `packages/shared/src/constants/index.ts` | None | Export pagination limits, rate thresholds, feature flags | Values are frozen (`Object.freeze`) or `as const` |

✅ **Phase 1 Complete:** `npm run build:shared` passes with zero warnings. No imports outside `shared/`.

---

### 🔵 Phase 2: Domain Layer (Models & Interfaces)
*Purpose:* Define business entities and repository contracts. Depends only on `shared/`.

| # | File Path | Dependencies | Implementation Steps | Acceptance Criteria |
|---|-----------|--------------|----------------------|---------------------|
| 2.1 | `packages/domain/src/models/user.model.ts` | `shared/types/common.ts` | Define `User` entity with fields, validation rules (Zod/Valibot), factory method | Type-safe construction; fails on invalid input |
| 2.2 | `packages/domain/src/repositories/user.repository.interface.ts` | `user.model.ts`, `shared/types/common.ts` | Declare `IUserRepository` with `findById`, `create`, `update`, `delete` | Pure interface; no implementation details |
| 2.3 | `packages/domain/src/models/post.model.ts` | `shared/types/common.ts`, `user.model.ts` (type-only) | Define `Post` entity, link to `User` via `authorId: IdType` | Circular reference avoided via type imports only |

✅ **Phase 2 Complete:** Domain compiles independently. All repository files end in `.interface.ts`.

---

### 🟡 Phase 3: Application Layer (Use Cases & Service Contracts)
*Purpose:* Orchestrate business rules. Depends on `domain/` and `shared/`. No DB or framework code.

| # | File Path | Dependencies | Implementation Steps | Acceptance Criteria |
|---|-----------|--------------|----------------------|---------------------|
| 3.1 | `packages/application/src/use-cases/create-user.use-case.ts` | `domain/repositories/user.repository.interface.ts`, `shared/errors/*` | Implement input validation, call repo interface, return DTO | Throws `ValidationError` on bad input; mocks pass |
| 3.2 | `packages/application/src/services/auth.service.interface.ts` | `shared/types/common.ts` | Define `IAuthService.authenticate(payload) => TokenResponse` | Interface only; no JWT/crypto imports |
| 3.3 | `packages/application/src/use-cases/get-posts.use-case.ts` | `domain/repositories/*`, `shared/constants/index.ts` | Handle pagination, filter mapping, return `PaginatedResult<Post>` | Respects max limit constant; type-safe filters |

✅ **Phase 3 Complete:** All use cases are unit-testable without DB. Zero framework imports.

---

### 🟠 Phase 4: Infrastructure & Adapters
*Purpose:* Implement repository interfaces, external clients, and DI bindings. Depends on `application/`, `domain/`, `shared/`.

| # | File Path | Dependencies | Implementation Steps | Acceptance Criteria |
|---|-----------|--------------|----------------------|---------------------|
| 4.1 | `packages/infrastructure/src/adapters/user.repository.prisma.ts` | `domain/repositories/user.repository.interface.ts`, `@prisma/client` | Map Prisma calls to interface; handle raw DB errors → `AppError` | Implements all interface methods; transaction-safe |
| 4.2 | `packages/infrastructure/src/clients/jwt.client.ts` | `application/services/auth.service.interface.ts`, `jsonwebtoken` | Implement token generation/validation; wrap crypto ops | Matches `IAuthService`; handles expiry & revocation |
| 4.3 | `packages/infrastructure/src/di/container.ts` | All adapters, all use-cases | Wire interfaces → implementations (TypeDI / Inversify / manual map) | Resolution returns correct instances; no circular DI |

✅ **Phase 4 Complete:** Adapters pass integration tests with testcontainers/mock clients. DI resolves cleanly.

---

### 🔴 Phase 5: Interface Layer (API/CLI Surface)
*Purpose:* Expose endpoints or commands. Depends on `application/`, `shared/`, framework.

| # | File Path | Dependencies | Implementation Steps | Acceptance Criteria |
|---|-----------|--------------|----------------------|---------------------|
| 5.1 | `packages/api/src/middleware/auth.middleware.ts` | `application/services/auth.service.interface.ts`, framework | Extract token, validate via DI service, attach user to context | Rejects invalid/missing tokens with 401 |
| 5.2 | `packages/api/src/controllers/user.controller.ts` | `application/use-cases/create-user.use-case.ts`, `shared/types/*` | Map request → use case input; handle errors → HTTP responses | OpenAPI spec matches routes; error mapping complete |
| 5.3 | `packages/api/src/routes/user.routes.ts` | `user.controller.ts`, framework router | Register paths, attach middleware, export router instance | Route table loads without circular requires |

✅ **Phase 5 Complete:** API starts locally; Swagger/CLI help renders; all routes hit use cases.

---

### ⚫ Phase 6: Bootstrap & Configuration
*Purpose:* Wire everything together, load env, start server/CLI. Depends on all layers.

| # | File Path | Dependencies | Implementation Steps | Acceptance Criteria |
|---|-----------|--------------|----------------------|---------------------|
| 6.1 | `src/config/env.config.ts` | `zod`, `.env.example` | Parse & validate env vars; export typed config object | Fails fast on missing/invalid vars in dev/prod |
| 6.2 | `src/app.ts` | DI container, routes/middleware, error handler | Instantiate server, register routers, mount global middleware | Graceful shutdown hook registered |
| 6.3 | `src/main.ts` | `app.ts`, `env.config.ts` | Load config → init DI → start app → log ready state | Exits cleanly on SIGTERM; health endpoint responds |

✅ **Phase 6 Complete:** `npm run dev` boots in <2s. Health check returns `200 OK`. Zero runtime cycles.

---
## 🔍 Dependency Validation & Enforcement

| Tool | Command/Config | Purpose |
|------|----------------|---------|
| `madge` | `npx madge --circular src/ packages/*/src/` | Detect circular imports before commit |
| ESLint | `"import/no-cycle": ["error", { "maxDepth": 3 }]` | Block cycles at lint time |
| Build Order | `npm run build:shared → build:domain → build:application → ...` | Enforce via CI matrix or pnpm/turborepo pipelines |
| Type-Only Imports | `import type { IRepo } from '...'` | Guarantee zero runtime dependency edges between layers |

---
## ✅ Execution Checklist
- [ ] Run Phase 1 → verify zero external imports in `shared/`
- [ ] Run Phase 2 → verify all repos are interfaces only
- [ ] Run Phase 3 → verify use cases pass unit tests without DB
- [ ] Run Phase 4 → verify adapters implement domain contracts exactly
- [ ] Run Phase 5 → verify controllers never import infrastructure directly
- [ ] Run Phase 6 → verify app boots, health check works, graceful shutdown triggers
- [ ] Run `madge --circular` & CI pipeline → ensure green

---
**Next Step:** Reply with your complete spec (entities, features, deployment target, framework preference). I will:
1. Replace placeholder paths with your exact domain names
2. Generate precise Zod/Prisma/CLI configs
3. Output a ready-to-run `turbo.json` / `pnpm-workspace.yaml` build order
4. Provide exact test fixtures matching your data models