# Scenario: Clean Separation from Dockman
- Given: Both dockerfile_utils.py and dockman.py exist
- When: Code is reviewed
- Then: dockerfile_utils.py imports nothing from dockman.py and has no shared code

## Test Steps

- Case 1 (no imports): dockerfile_utils.py does not import from dockman
- Case 2 (standalone): dockerfile_utils.py can run independently

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
