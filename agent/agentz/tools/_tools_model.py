from agentz.pydantic_models import ExperimentConfiguration
from agentz.ACI import OSWorldEnvironment

from .openai_api import GPTClientManager
from .os_inspector import OSInspector

class Tools():

    def __init__(self, settings : ExperimentConfiguration, env : OSWorldEnvironment):
        """
        Initialize `Tools` dependencies and runtime state.
        
        Parameters
        ----------
        settings : ExperimentConfiguration
            Runtime settings for this component.
        env : OSWorldEnvironment
            Environment interface used to execute actions.
        
        Returns
        -------
        None
            No return value.
        """
        self.settings = settings
        self.env = env


    def activate(self):
        """
        Activate the tool and register its metadata.
        
        Returns
        -------
        Any
            Function result.
        
        """
        self.gpt_client = GPTClientManager(self.settings)
        self.os_inspector = OSInspector(self.env)

    def reset(self, settings : ExperimentConfiguration):
        """
        Reset the runtime status of this tool.
        
        Parameters
        ----------
        settings : ExperimentConfiguration
            Component settings.
        
        Returns
        -------
        Any
            Function result.
        
        """
        self.settings = settings
