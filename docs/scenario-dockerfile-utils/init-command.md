# Scenario: Init Command
- Given: dockerfile_utils.py is available
- When: User runs `./dockerfile_utils.py init`
- Then: Dockerfile is created at .dockman/Dockerfile with proper content

## Test Steps

- Case 1 (happy path): init creates .dockman/Dockerfile with correct content
- Case 2 (overwrite): init --force overwrites existing Dockerfile
- Case 3 (no overwrite): init without --force fails when Dockerfile exists

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
