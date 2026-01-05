# Models/__init__.py
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .EvoStructCLIP import EvoStructCLIP
from .MSABranch import MSABranch
from .VoxelBranch import VoxelBranch