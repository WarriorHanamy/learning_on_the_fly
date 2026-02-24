# Scenario: Merge Config with Overrides

- Given: A base frozen dataclass config instance and a dict with override values
- When: `merge_config` is called with the config and overrides
- Then: A new frozen dataclass instance is returned with overridden values

## Test Steps

- Case 1 (happy path): Override scalar values correctly
- Case 2 (edge case): Unspecified values are preserved
- Case 3 (edge case): Nested config structures are handled
- Case 4 (edge case): Empty override dict returns equivalent config

## Status
- [x] Write scenario document
- [x] Write solid test according to document
- [x] Run test and watch it failing
- [x] Implement to make test pass
- [x] Run test and confirm it passed
- [x] Refactor implementation without breaking test
- [x] Run test and confirm still passing after refactor
