# Scenario: Trajectory Export
- Given: Trained policy and trajectory data
- When: Exporting trajectory to CSV
- Then: CSV file contains required columns

## Test Steps

- Case 1 (valid columns): CSV contains index, t, px, py, pz, qw, qx, qy, qz, vx, vy, vz columns
- Case 2 (file created): CSV file is created at specified path

## Status
- [x] Write scenario document
- [ ] Write solid test according to document
- [ ] Run test and watch it failing
- [ ] Implement to make test pass
- [ ] Run test and confirm it passed
- [ ] Refactor implementation without breaking test
- [ ] Run test and confirm still passing after refactor
