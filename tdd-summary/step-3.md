# Step 3 - Write Failing Test (RED)

## Failing Tests Created

- FR-1: CLI Module Executable - `docs/scenario/cli-module-executable.md` - `tests/scripts/test_train_traj_tracking.py::TestModuleImport`
- FR-2: CLI Argument Parsing - `docs/scenario/cli-argument-parsing.md` - `tests/scripts/test_train_traj_tracking.py::TestArgparseCLI`
- FR-3: Environment Creation - `docs/scenario/create-env-traj-tracking.md` - `tests/scripts/test_train_traj_tracking.py::TestCreateEnv`
- FR-4: Reference Trajectory Loading - `docs/scenario/reference-trajectory-loading.md` - `tests/scripts/test_train_traj_tracking.py::TestReferenceTrajectory`
- FR-5: Training Loop - `docs/scenario/training-loop.md` - `tests/scripts/test_train_traj_tracking.py::TestIntegration`
- FR-6: Trajectory Export - `docs/scenario/trajectory-export.md` - `tests/scripts/test_train_traj_tracking.py::TestTrajectoryExport`

## Test Run Results

28 tests collected, 26 failing (module not found), 1 passing (CSV columns check), 2 skipped (require GPU)

All failures are expected - module `lotf.scripts.train_traj_tracking` does not exist yet.
