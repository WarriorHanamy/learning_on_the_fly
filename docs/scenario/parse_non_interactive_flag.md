# Scenario: Parse --non-interactive flag
- Given: User runs dockrun with --non-interactive flag
- When: Command is parsed
- Then: Flag is recognized and non-interactive mode is enabled

## Test Steps

- Case 1 (happy path): dockrun --non-interactive echo "hello"
- Case 2 (edge case): dockrun without --non-interactive flag

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
