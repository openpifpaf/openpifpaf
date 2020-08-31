"""Plugins for OpenPifPaf.

A plugin is a module that starts with `openpifpaf_`.
The module has to contain a `register()` function.

Follows Flask-style plugin discovery:
https://packaging.python.org/guides/creating-and-discovering-plugins/
"""

import importlib
import logging
import pkgutil

LOG = logging.getLogger(__name__)

REGISTERED = set()


def register():
    discovered_plugins = {
        name: importlib.import_module(name)
        for finder, name, ispkg in pkgutil.iter_modules()
        if name.startswith('openpifpaf_')
    }
    LOG.debug('discoverd %d plugins', len(discovered_plugins))

    for name, module in discovered_plugins.items():
        if name in REGISTERED:
            continue
        LOG.info('registering %s', name)
        module.register()
        REGISTERED.add(name)
