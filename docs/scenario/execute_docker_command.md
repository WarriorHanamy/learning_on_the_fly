# Scenario: Execute docker command
- Given: A docker command is constructed
- When: execute function is called
- Then: Command is executed and exit code is returned

## Test Steps

- Case 1 (happy path): Execute valid docker command
- Case 2 (edge case): Execute invalid docker command

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
