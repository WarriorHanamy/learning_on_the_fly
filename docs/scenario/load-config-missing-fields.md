# Scenario: Load Config with Missing Required Fields

- Given: A valid YAML file that is missing required fields
- When: `load_config` is called with the file path
- Then: `ValueError` is raised with field name information

## Test Steps

- Case 1 (happy path): Missing single required field raises ValueError with field name
- Case 2 (edge case): Missing multiple required fields raises ValueError
- Case 3 (edge case): Extra fields in YAML are ignored (not an error)

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
