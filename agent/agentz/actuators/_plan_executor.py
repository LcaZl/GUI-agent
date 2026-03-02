import logging
from typing import Any

from agentz.ACI import OSWorldEnvironment
from agentz.pydantic_models import PlanExecutorSettings, Step


class PlanExecutor:
    def __init__(
        self,
        settings: PlanExecutorSettings,
        env: OSWorldEnvironment,
    ) -> None:
        """
        Initialize the plan executor.
        
        Parameters
        ----------
        settings : PlanExecutorSettings
            Executor runtime settings.
        env : OSWorldEnvironment
            Environment interface used to execute actions.
        
        Returns
        -------
        None
            No return value.
        """
        self.env = env
        self.settings = settings
        self.default_pause_sec = float(settings.default_pause_sec)
        self.observe_after_python = bool(settings.observe_after_python)
        self.logger = logging.getLogger("PlanExecutor")

    def _resolve_pause(self, pause: Any) -> float:
        """
        Resolve a pause value with safe defaults.
        
        Parameters
        ----------
        pause : Any
            Pause value requested by the planner step.
        
        Returns
        -------
        float
            Effective pause in seconds.
        """
        try:
            p = float(pause)
        except Exception:
            p = 0.0
        if p <= 0.0:
            return self.default_pause_sec
        return p

    def execute_step(self, step: Step) -> dict:
        """
        Execute one planner step and return the resulting perception payload.
        
        Parameters
        ----------
        step : Step
            Planner step to execute.
        
        Returns
        -------
        dict
            Environment response, optionally refreshed after Python commands.
        """
        action_type = str(step.action_type or "").strip()
        action = {
            "action_type": step.action_type,
            "command": step.command,
        }
        pause = self._resolve_pause(step.pause)

        perception = self.env.step(action=action, pause=pause)

        if self.observe_after_python and action_type.lower() == "python":
            try:
                refreshed_obs = self.env.observe()
                if isinstance(perception, dict) and "obs" in perception:
                    perception["obs"] = refreshed_obs
                else:
                    perception = {
                        "obs": refreshed_obs,
                        "reward": None,
                        "done": None,
                        "info": None,
                    }
            except Exception as exc:
                self.logger.warning(
                    "Post-python observe failed; using step observation. error=%s",
                    exc,
                )

        return perception
