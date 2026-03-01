# Scenario: Build Command
- Given: Root Dockerfile exists
- When: User runs `./dockerfile_utils.py build`
- Then: Docker image is built and tagged as lotf:latest

## Test Steps

- Case 1 (happy path): build command runs docker build with correct tag
- Case 2 (dockerfile check): build command verifies Dockerfile exists before building

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
