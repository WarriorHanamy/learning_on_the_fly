# Scenario: Load Config with Malformed YAML

- Given: A file exists but contains malformed YAML content
- When: `load_config` is called with the file path
- Then: `ValueError` is raised indicating YAML parsing failure

## Test Steps

- Case 1 (happy path): Invalid YAML syntax raises ValueError
- Case 2 (edge case): Empty file raises appropriate error
- Case 3 (edge case): File with wrong structure (not a dict) raises appropriate error

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
