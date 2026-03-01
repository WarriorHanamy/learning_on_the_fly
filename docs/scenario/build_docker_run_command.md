# Scenario: Build docker run command with fixed parameters
- Given: User provides a command to run in container
- When: build_docker_run_command is called
- Then: Docker command includes lotf:latest, --gpus=all, -v $(pwd):/app, -w /app, --rm

## Test Steps

- Case 1 (happy path): Command built with all fixed parameters
- Case 2 (edge case): Empty command argument

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
