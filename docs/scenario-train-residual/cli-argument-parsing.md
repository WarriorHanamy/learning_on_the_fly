# Scenario: CLI Argument Parsing

- Given: Command line arguments
- When: Script is invoked with --config, --dataset, --output
- Then: Arguments are correctly parsed into namespace

## Test Steps

- Case 1 (happy path): Parse all three arguments correctly
- Case 2 (edge case): Default values when arguments not provided
- Case 3 (edge case): --help flag works correctly

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
