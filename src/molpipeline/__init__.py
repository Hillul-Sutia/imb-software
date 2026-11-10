"""Reproducible molecular classification experiments."""


def __getattr__(name):
    # Older programmatic runs persisted this class at molpipeline.MTuneAdapter.
    if name == 'MTuneAdapter':
        from .training import MTuneAdapter
        return MTuneAdapter
    raise AttributeError(name)
