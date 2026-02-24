# Step 1 - Understand Intent

## Functional Requirements

### FR-1: Module Structure and Exports
Create `lotf/configs/__init__.py` that exports `load_config` function, dataclass configs, and `merge_config` utility.

### FR-2: Dataclass Config Definitions
Define frozen dataclass configs with type hints:
- `TrainingConfig`: fields `seed`, `num_envs`, `max_epochs`, `learning_rate`
- `EnvConfig`: fields `dt`, `delay`, `max_steps_in_episode`
- `SimConfig`: simulation parameters (extensible)

### FR-3: YAML Config Loader
Implement `load_config` function that:
- Accepts YAML file path as string or Path
- Returns a frozen dataclass instance
- Supports optional dict overrides

### FR-4: FileNotFoundError for Invalid Paths
Invalid YAML paths raise `FileNotFoundError` with helpful message containing the path.

### FR-5: ValidationError for Missing Fields
Missing required fields raise `ValidationError` (or ValueError with field name information).

### FR-6: Config Merge Utility
`merge_config` utility that:
- Overrides scalar values from dict to dataclass
- Preserves unspecified nested values
- Returns new frozen dataclass instance

## Assumptions

- Using standard Python `dataclasses` with `frozen=True` (not jax-dataclasses) since configs don't need JAX compatibility
- Using `ValidationError` from `dataclasses` module or custom exception - will use `ValueError` as it's simpler and matches test plan expectations
- YAML parsing via `pyyaml` which is already in dependencies
- All config fields are required (no Optional fields with defaults in base configs)
- `merge_config` creates a new dataclass instance rather than modifying in-place (respecting frozen constraint)
