from sugarscape.config import SugarscapeConfig
from sugarscape.agent import SugarAgent
from sugarscape.environment import SugarEnvironment
from sugarscape.simulation import SugarSimulation
from sugarscape.welfare import WelfareCalculator

# Import plotting module only if matplotlib is available
try:
    from sugarscape.welfare_plots import WelfarePlotter
    __all__ = [
        "SugarscapeConfig",
        "SugarAgent",
        "SugarEnvironment",
        "SugarSimulation",
        "WelfareCalculator",
        "WelfarePlotter"
    ]
except ImportError:
    WelfarePlotter = None
    __all__ = [
        "SugarscapeConfig",
        "SugarAgent",
        "SugarEnvironment",
        "SugarSimulation",
        "WelfareCalculator"
    ]
