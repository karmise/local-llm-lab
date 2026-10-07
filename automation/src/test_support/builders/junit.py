"""Named scenario preparation and test doubles."""


def make_read_once_stub(path, raw, reads):
    def read_once(self):
        assert self == path
        reads.append(1)
        return raw if len(reads) == 1 else b'<testsuite><testcase name="changed"/></testsuite>'

    return read_once
