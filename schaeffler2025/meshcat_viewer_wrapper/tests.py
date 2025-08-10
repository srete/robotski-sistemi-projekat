import doctest

from supaero2025.meshcat_viewer_wrapper import colors


def load_tests(loader, tests, pattern):
    tests.addTests(doctest.DocTestSuite(colors))
    return tests
