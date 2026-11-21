import os
import sys

# 把 src 目录加入 sys.path，保证 src/__init__.py 与 src 内部模块之间的绝对导入
# （如 from positionalencoding import *）在任何运行方式下都成立
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from positionalencoding import *
