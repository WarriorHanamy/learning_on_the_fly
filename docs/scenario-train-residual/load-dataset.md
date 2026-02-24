# Scenario: Load Dataset from CSV

- Given: A valid CSV file path with numeric data
- When: load_dataset is called with path and input_dim
- Then: Returns tuple of (X, y) JAX arrays with correct shapes

## Test Steps

- Case 1 (happy path): Load example_dataset.csv with input_dim=19, verify X shape (N, 19) and y shape (N, 3)
- Case 2 (edge case): FileNotFoundError raised for non-existent path
- Case 3 (edge case): Arrays are float32 dtype

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
