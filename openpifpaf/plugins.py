"""Plugins for OpenPifPaf.

A plugin is a module that starts with `openpifpaf_`.
The module has to contain a `register()` function.

Follows Flask-style plugin discovery:
https://packaging.python.org/guides/creating-and-discovering-plugins/
"""

import importlib
import pkgutil

REGISTERED = {}


def register():
    from . import contrib  # pylint: disable=import-outside-toplevel,cyclic-import

    contrib_plugins = {
        'openpifpaf.contrib.{}'.format(name):
            importlib.import_module('openpifpaf.contrib.{}'.format(name))
        for finder, name, is_pkg in pkgutil.iter_modules(contrib.__path__)
    }
    discovered_plugins = {
        name: importlib.import_module(name)
        for finder, name, is_pkg in pkgutil.iter_modules()
        if name.startswith('openpifpaf_')
    }
    # This function is called before logging levels are configured.
    # Uncomment for debug:
    # print('{} contrib plugins. Discovered {} plugins.'.format(
    #     len(contrib_plugins), len(discovered_plugins)))

    for name, module in dict(**contrib_plugins, **discovered_plugins).items():
        if name in REGISTERED:
            continue
        module.register()
        REGISTERED[name] = module


def versions():
    return {name: getattr(m, '__version__', 'unknown')
            for name, m in REGISTERED.items()
            if not name.startswith('openpifpaf.contrib.')}
