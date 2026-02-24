# Scenario: Create Ensemble from Config

- Given: A ResidualDynamicsConfig with num_models specified
- When: create_ensemble is called with config
- Then: Returns correct number of train states initialized

## Test Steps

- Case 1 (happy path): Config with num_models=3 returns 3 train states
- Case 2 (edge case): Config with num_models=1 returns 1 train state
- Case 3 (edge case): Train states have correct parameter shapes

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
