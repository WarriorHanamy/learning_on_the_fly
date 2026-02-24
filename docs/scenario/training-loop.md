# Scenario: Training Loop
- Given: Valid config and environment
- When: Running training loop
- Then: Training completes and checkpoint is saved

## Test Steps

- Case 1 (minimal run): Short training run completes without errors
- Case 2 (checkpoint saved): Policy checkpoint file is created

## Status
- [x] Write scenario document
- [ ] Write solid test according to document
- [ ] Run test and watch it failing
- [ ] Implement to make test pass
- [ ] Run test and confirm it passed
- [ ] Refactor implementation without breaking test
- [ ] Run test and confirm still passing after refactor
