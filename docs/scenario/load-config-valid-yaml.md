# Scenario: Load Config from Valid YAML

- Given: A valid YAML file exists at the specified path
- When: `load_config` is called with the file path and target config class
- Then: A frozen dataclass instance is returned with all fields populated from YAML

## Test Steps

- Case 1 (happy path): Load TrainingConfig from valid YAML with all required fields
- Case 2 (happy path): Load EnvConfig from valid YAML with all required fields
- Case 3 (happy path): Load SimConfig from valid YAML with all required fields
- Case 4 (edge case): Load config with optional overrides dict applied

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
