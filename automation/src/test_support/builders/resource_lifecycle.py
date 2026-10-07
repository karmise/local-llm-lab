"""Named scenario preparation and test doubles."""


def prepare_resources_are_cleaned_up_after_each_failure_case(failure, result):
    if failure == "cleanup":
        result.stdout.fnmatch_lines(["*Quality check failed*", "*1 failed, 1 error*"])
