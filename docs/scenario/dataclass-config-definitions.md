# Scenario: Dataclass Config Definitions

- Given: The config module is imported
- When: User accesses TrainingConfig, EnvConfig, SimConfig
- Then: All dataclasses are properly defined with correct fields and type hints

## Test Steps

- Case 1 (happy path): TrainingConfig has fields: seed, num_envs, max_epochs, learning_rate
- Case 2 (happy path): EnvConfig has fields: dt, delay, max_steps_in_episode
- Case 3 (happy path): SimConfig exists and is a frozen dataclass
- Case 4 (edge case): All configs are frozen (immutable)

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
