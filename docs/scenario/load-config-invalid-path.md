# Scenario: Load Config with Invalid Path

- Given: A file path that does not exist
- When: `load_config` is called with the non-existent path
- Then: `FileNotFoundError` is raised with helpful message containing the path

## Test Steps

- Case 1 (happy path): Non-existent path raises FileNotFoundError
- Case 2 (edge case): Empty string path raises FileNotFoundError
- Case 3 (edge case): Path to directory instead of file raises appropriate error

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
