"""Named parameter cases for resource_lifecycle scenarios."""

RESOURCES_ARE_CLEANED_UP_AFTER_EACH_FAILURE_FAILURE_FAILED_ERRORS_CLEANUP_CASES = [
        ("none", 0, 0, ["folder", "workspace"]), ("workspace-validation", 0, 1, ["workspace"]),
        ("folder-validation", 0, 1, ["folder", "workspace"]), ("upload", 0, 1, ["folder", "workspace"]),
        ("indexing", 0, 1, ["folder", "workspace"]), ("test", 1, 0, ["folder", "workspace"]),
        ("cleanup", 1, 1, ["folder", "workspace"])]
