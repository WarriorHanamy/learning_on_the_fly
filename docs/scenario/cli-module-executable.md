# Scenario: CLI Module Executable
- Given: Script file `lotf/scripts/train_traj_tracking.py` exists
- When: Running `uv run python -m lotf.scripts.train_traj_tracking --help`
- Then: Script executes and shows help message

## Test Steps

- Case 1 (happy path): Script can be imported as module
- Case 2 (help flag): `--help` shows usage information

## Status
- [x] Write scenario document
- [ ] Write solid test according to document
- [ ] Run test and watch it failing
- [ ] Implement to make test pass
- [ ] Run test and confirm it passed
- [ ] Refactor implementation without breaking test
- [ ] Run test and confirm still passing after refactor
