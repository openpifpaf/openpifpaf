"""Plugins for OpenPifPaf.

A plugin is a module that starts with `openpifpaf_`.
The module has to contain a `register()` function.

Follows Flask-style plugin discovery:
https://packaging.python.org/guides/creating-and-discovering-plugins/
"""

import importlib
import logging
import pkgutil

from . import contrib

LOG = logging.getLogger(__name__)

REGISTERED = set()


def register():
    contrib_plugins = {
        'openpifpaf.contrib.{}'.format(name):
            importlib.import_module('openpifpaf.contrib.{}'.format(name))
        for finder, name, ispkg in pkgutil.iter_modules(contrib.__path__)
    }
    discovered_plugins = {
        name: importlib.import_module(name)
        for finder, name, ispkg in pkgutil.iter_modules()
        if name.startswith('openpifpaf_')
    }
    # print('{} contrib plugins. Discoverd {} plugins.'.format(
    #     len(contrib_plugins), len(discovered_plugins)))

    for name, module in dict(**contrib_plugins, **discovered_plugins).items():
        if name in REGISTERED:
            continue
        module.register()
        REGISTERED.add(name)
