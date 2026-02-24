# Scenario: Environment Creation
- Given: Valid config for trajectory tracking
- When: Calling `create_env(config)`
- Then: TrajTrackingStateEnv is created with correct observation space

## Test Steps

- Case 1 (happy path): create_env returns wrapped environment with correct attributes
- Case 2 (observation space): Observation space has correct dimensions and is normalized
- Case 3 (action space): Action space has 4 dimensions (thrust + 3 rates)
- Case 4 (max steps): max_steps_in_episode calculated correctly from config

## Status
- [x] Write scenario document
- [ ] Write solid test according to document
- [ ] Run test and watch it failing
- [ ] Implement to make test pass
- [ ] Run test and confirm it passed
- [ ] Refactor implementation without breaking test
- [ ] Run test and confirm still passing after refactor
