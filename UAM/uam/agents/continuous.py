import glob
from importlib import import_module
from os.path import dirname, basename, isfile, join

# 文件内调用
from uam import logger
from uam.agents import __path__

__currentmodule__= import_module("uam.agents.continuous")

modules = glob.glob(join(dirname(__path__[0] + "/_continuous/"), "*.py"))
_CAGENTS = []
for f in modules:
    if isfile(f) and not f.endswith('__init__.py'):
        _CAGENTS.append(basename(f)[:-3])

logger.info("--- Loading Continuous Agents: ---")

for filename in _CAGENTS:
    module = import_module("uam.agents._continuous." + filename)
    cls_ = getattr(module, filename + "Agent")
    setattr(__currentmodule__, filename + "Agent", cls_)
    logger.info(f"Loading {filename}Agent from {module.__name__}")
