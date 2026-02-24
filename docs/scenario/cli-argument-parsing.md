# Scenario: CLI Argument Parsing
- Given: Script with argparse CLI
- When: Parsing command-line arguments
- Then: All required arguments are parsed correctly

## Test Steps

- Case 1 (default values): No arguments returns default config, checkpoint, trajectory-output paths
- Case 2 (custom config): `--config custom.yaml` parses correctly
- Case 3 (custom checkpoint): `--checkpoint my_checkpoint` parses correctly
- Case 4 (trajectory output): `--trajectory-output traj.csv` parses correctly
- Case 5 (all arguments): All three arguments together parse correctly

## Status
- [x] Write scenario document
- [ ] Write solid test according to document
- [ ] Run test and watch it failing
- [ ] Implement to make test pass
- [ ] Run test and confirm it passed
- [ ] Refactor implementation without breaking test
- [ ] Run test and confirm still passing after refactor
