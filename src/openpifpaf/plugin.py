"""Plugins for OpenPifPaf.

A plugin is a module that starts with `openpifpaf_`.
The module has to contain a `register()` function.

Follows Flask-style plugin discovery:
https://packaging.python.org/guides/creating-and-discovering-plugins/
"""

import sys
import importlib
import pkgutil

REGISTERED = {}


def register():
    from . import plugins  # pylint: disable=import-outside-toplevel,cyclic-import

    plugin_names = [
        'openpifpaf.plugins.{}'.format(name)
        for finder, name, is_pkg in pkgutil.iter_modules(plugins.__path__)
    ] + [
        name
        for finder, name, is_pkg in pkgutil.iter_modules()
        if name.startswith('openpifpaf_')
    ]

    for name in plugin_names:
        if name not in sys.modules:
            module = importlib.import_module(name)
        else:
            module = sys.modules[name]

        # partially imported modules do not have the register attribute
        if not hasattr(module, 'register'):
            continue

        module.register()
        REGISTERED[name] = module


def versions():
    return {name: getattr(m, '__version__', 'unknown')
            for name, m in REGISTERED.items()
            if not name.startswith('openpifpaf.plugins.')}
