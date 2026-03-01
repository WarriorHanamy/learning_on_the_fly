# Scenario: Tool Structure and Executability
- Given: dockerfile_utils.py exists in project root
- When: The file is checked for execution permission
- Then: File should be executable (chmod +x) and have proper shebang line

## Test Steps

- Case 1 (file exists): dockerfile_utils.py exists in project root
- Case 2 (executable): File has execute permissions and shebang line

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
