# Scenario: Version command
- Given: User runs dockrun --version
- When: Command is parsed
- Then: Version information is displayed

## Test Steps

- Case 1 (happy path): dockrun --version displays version
- Case 2 (edge case): dockrun -v short form displays version

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
