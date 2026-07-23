You haven’t provided the actual code or entities to extract from, but I’ll give you a **production-ready pattern** for extracting shared classes/types while strictly preventing circular imports. Apply this directly to your codebase, or paste your entities and I’ll refactor them into it.

---
## 🔁 Dependency Direction Rule (Non-Negotiable)
```
shared/          ← domain/        ← application/     ← infrastructure/
(types/interfaces)   (business logic)    (use cases/services)  (DB, APIs, adapters)
```
**`shared/` must be a leaf node.** It imports **nothing**. Everything else may import from it.

---
## 📦 What Belongs in `shared/`
| ✅ Include | ❌ Exclude |
|-----------|------------|
| DTOs, request/response interfaces | Controllers, routes, CLI commands |
| Enums, string literals, constants | Business rules, validators with side effects |
| Base error classes & codes | Repository implementations, DB queries |
| Cross-cutting types (e.g., `PaginatedResult<T>`, `ApiResponse<T>`) | State management, caching logic |

---
## 🛡 Circular Import Prevention Tactics

### 1. **Never export concrete classes that depend on downstream layers**
```ts
// ❌ BAD: shared/UserService.ts imports from domain/
import { UserRepository } from 'domain/repositories' // Cycle risk

// ✅ GOOD: shared only defines contracts
export interface IUserRepository {
  findById(id: string): Promise<User | null>;
}
```

### 2. **Avoid barrel `index.ts` files that re-export everything**
Barrel files hide import graphs and silently create cycles when modules reference each other through the barrel.
```ts
// ❌ shared/index.ts → exports A, B, C → A imports from B via index → cycle
// ✅ Import directly: import { UserDTO } from 'shared/types/user'
```

### 3. **Use interfaces over classes for cross-boundary contracts**
Interfaces are erased at runtime and don’t trigger circular initialization issues in TS/JS.

### 4. **Break unavoidable cycles with dependency inversion or lazy imports**
```ts
// If module A must reference B, but B references A:
import type { IB } from './b'; // Type-only import → zero runtime cycle
export class A { process(b: IB) { ... } }

// Or use dynamic/lazy import for runtime graphs:
const loadB = () => import('./b');
```

---
## 📐 Ready-to-Apply Structure
```
packages/shared/src/
├── types/
│   ├── api.ts          # ApiResponse<T>, PaginatedResult<T>
│   ├── user.ts         # UserDTO, CreateUserInput, UpdateUserInput
│   └── enums.ts        # RoleEnum, StatusEnum (as const + type)
├── errors/
│   ├── base-error.ts   # AppError extends Error
│   └── http-errors.ts  # NotFoundError, ValidationError
├── constants/
│   └── pagination.ts   # DEFAULT_PAGE_SIZE, MAX_LIMIT
└── utils/              # Pure functions only (slugify, formatDate)
    └── date.ts
```

**`tsconfig.json` path alias:**
```json
"paths": {
  "@shared/*": ["./packages/shared/src/*"]
}
```

---
## 🔍 Enforcement Tooling
Add these to catch cycles early:
```bash
npm i -D eslint-plugin-import
```
`.eslintrc.json`:
```json
{
  "rules": {
    "import/no-cycle": ["error", { "maxDepth": 3, "ignoreExternal": true }],
    "import/no-relative-packages": "error"
  }
}
```

---
## ✅ Next Step
Paste your entities, classes, or type definitions. I will:
1. Extract cross-cutting types into `shared/`
2. Refactor concrete dependencies to interfaces
3. Remove hidden cycles via type-only imports or structural splits
4. Output exact file paths + import statements ready for copy/paste