# Scenario: Reference Trajectory Loading
- Given: Config with ref_traj_name field
- When: Environment is created with config
- Then: Reference trajectory is loaded correctly

## Test Steps

- Case 1 (FIG8): Loading FIG8 trajectory works
- Case 2 (CIRCLE): Loading CIRCLE trajectory works
- Case 3 (STAR): Loading STAR trajectory works

## Status
- [x] Write scenario document
- [ ] Write solid test according to document
- [ ] Run test and watch it failing
- [ ] Implement to make test pass
- [ ] Run test and confirm it passed
- [ ] Refactor implementation without breaking test
- [ ] Run test and confirm still passing after refactor
