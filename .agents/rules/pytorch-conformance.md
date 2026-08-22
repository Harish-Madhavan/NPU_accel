# PyTorch Conformance Rule

## Intent
Enforce that all features, optimizations, and workflows conform strictly to standard PyTorch idioms.

## Rules
1. **Never Invent Custom Top-Level APIs When Standard PyTorch Exists**:
   - Prefer `torch.compile(model, backend="npu")` over custom compilation functions.
   - Use `loss.backward()` and `optimizer.step()` for training.
   - Use `torch.autocast(...)` for mixed precision.
   - Use standard `torch.nn.Module` subclasses.

2. **Automatic Optimization Default**:
   - All hardware Level Zero optimizations (Turbo boost, QDQ fusion, multi-stream execution, memory preservation) must be active **by default**.
   - Do not require users to pass special flags or configuration dictionaries to achieve peak hardware performance.

3. **Clean Error Handling**:
   - In `strict=False` mode, unsupported operator subgraphs must automatically partition and fall back to PyTorch CPU execution without raising unexpected crashes.
