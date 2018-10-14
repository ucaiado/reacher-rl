import sys
sys.path.append("../")

try:
    from models import Actor, Critic
    from utils import param_table
    from utils.make_env import make
except ModuleNotFoundError:
    from .models import Actor, Critic
    from drlnd.utils import param_table
    from drlnd.utils.make_env import make
