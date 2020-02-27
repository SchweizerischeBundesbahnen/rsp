# https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file
class multifile(object):
    """Allows teeing."""
    def __init__(self, files):
        self._files = files

    def __getattr__(self, attr, *args):
        return self._wrap(attr, *args)

    def _wrap(self, attr, *args):
        def g(*a, **kw):
            for f in self._files:
                res = getattr(f, attr, *args)(*a, **kw)
            return res

        return g
