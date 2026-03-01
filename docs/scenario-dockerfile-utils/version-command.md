# Scenario: Version Command
- Given: dockerfile_utils.py is available
- When: User runs `./dockerfile_utils.py version`
- Then: Tool version is displayed in correct format

## Test Steps

- Case 1 (happy path): version command displays version in correct format
- Case 2 (version format): Version string follows pattern "dockerfile_utils.py version X.Y.Z"

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
