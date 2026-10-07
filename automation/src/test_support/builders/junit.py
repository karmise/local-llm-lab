"""Named scenario preparation and test doubles."""

from test_support.assertions import values


def make_read_once_stub(path, raw, reads):
    def read_once(self):
        values.equal(self, path)
        reads.append(1)
        return raw if len(reads) == 1 else b'<testsuite><testcase name="changed"/></testsuite>'

    return read_once
