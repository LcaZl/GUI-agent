import logging
from typing import Any

from agentz.memory import TRIMLLM, EpisodicMemory, HistoryManager, OnlineTMS
from agentz.utility import print_dict, visualize_ui_elements
from agentz.ACI import OSWorldEnvironment
from agentz.pydantic_models import ExecutedChunk, ExperimentConfiguration, Observation
from agentz.perception import PerceptionInterface
from agentz.tools import Tools
from agentz.planning import Planner
from agentz.actuators import PlanExecutor
from agentz.pydantic_models import Episode
from agentz.judge import Judge

import os
import time
import asyncio
import json
import hashlib
from agentz.memory.utils import append_metrics_csv
from agentz.constants import (
    TMS_GRID,
    TMS_MAX_ANCHORS_PER_OBS,
    TMS_MAX_NODES_IN_PROMPT,
    TRIM_GRID,
    TRIM_MAX_ANCHORS_IN_PROMPT_OVERRIDE,
    TRIM_MAX_NODES_IN_PROMPT,
)


class Agent:
    """
    Main Agent class.

    This class orchestrates the full agent lifecycle:
    environment setup, perception, planning, execution, evaluation,
    and memory management using a BDI-style control loop.
    """

    def __init__(self, name: str, settings: ExperimentConfiguration) -> None:
        """
        Initialize the Agent with a name and experiment settings.
        
        Args:
            name: Identifier for the agent instance.
            settings: Full experiment configuration object.
        """
        self.logger = logging.getLogger("Agent")
        self.name: str = name
        self.settings: ExperimentConfiguration = settings

        # Core subsystems (initialized asynchronously in create)
        self.env: OSWorldEnvironment = None
        self.tools: Tools = None
        self.perception: PerceptionInterface = None
        self.executor: PlanExecutor = None
        self.planner: Planner = None
        self.judge: Judge = None

        # Cache of the most recent task
        self._last_task: dict[str, Any] = None

    def stop(self):
        """
        Process stop.
        
        Returns
        -------
        Any
            Function result.
        
        """
        self.memory_manager.close()

    @classmethod
    async def create(cls, name: str, settings) -> "Agent":
        """
        Asynchronous factory method for creating and fully initializing an Agent.
        
        This method ensures that all async-dependent components (e.g. environment)
        are correctly initialized before the agent is returned.
        
        Args:
            name: Identifier for the agent instance.
            settings: Full experiment configuration object.
        
        Returns:
            A fully initialized Agent instance.
        """
        self = cls(name, settings)

        # Log experiment configuration for reproducibility
        self.logger.info("Experiment settings:")
        print_dict(settings.model_dump())

        # Create OSWorld environment and start async initialization
        self.env = OSWorldEnvironment(settings.osworld_settings)
        self.env.start_init()

        # Initialize episodic memory manager
        self.memory_manager = EpisodicMemory(self.settings.memory_settings)

        # Initialize tools that depend on the environment
        self.tools = Tools(settings, self.env)
        self.tools.activate()

        # Initialize perception pipeline
        self.perception = PerceptionInterface(
            settings.perception_settings,
            parallel=True,
            debug_visualizations=True
        )

        self.logger.info("Memory initialized. mem_root=%s", settings.memory_settings.root)

        # Initialize executor for low-level action execution
        self.executor = PlanExecutor(settings.plan_executor_settings, self.env)

        # Initialize planner for high-level decision making
        self.planner = Planner(
            settings=settings.planner_settings,
            gpt_client=self.tools.gpt_client,
            env=self.env,
            perception=self.perception,
            executor=self.executor,
            mem_root=str(settings.memory_settings.root)
        )

        # Initialize judge for post-execution evaluation
        self.judge = Judge(settings, self.tools)


        # Initialize history manager for structured interaction logs
        self.history_manager = HistoryManager()

        # Initialize online TMS (Task Memory Structure)
        self.tms = OnlineTMS(
            grid=TMS_GRID,
            max_nodes_in_prompt=TMS_MAX_NODES_IN_PROMPT,
            max_anchors_per_obs=TMS_MAX_ANCHORS_PER_OBS,
        )

        # Initialize TRIM LLM for memory updates
        self.trim = TRIMLLM(
            gpt_client=self.tools.gpt_client,
            grid=TRIM_GRID,
            max_nodes_in_prompt=TRIM_MAX_NODES_IN_PROMPT,
            max_anchors_in_prompt=TRIM_MAX_ANCHORS_IN_PROMPT_OVERRIDE,
        )

        # Ensure TMS starts from a clean state
        self.tms.reset()

        # Wait until the environment is fully ready
        await self.env.wait_ready(timeout=float(settings.osworld_settings.init_timeout_sec))
        self.logger.info("Agent initialized. env.ready=%s", self.env.ready)


        return self

    def _log_chunk_evaluation(self, chunk: ExecutedChunk):
        """
        Log a compact but detailed summary of a chunk evaluation.
        
        Args:
            chunk: ExecutedChunk containing steps and evaluation results.
        """
        self.logger.info(
            "Chunk result | success=%s | failure_type=%s | failing_step=%s",
            chunk.overall_success,
            chunk.failure_type,
            chunk.failing_step_index,
        )

        # Log per-step evaluation details
        for step, ev in zip(chunk.steps, chunk.steps_eval):
            status = "OK" if ev.success else "FAIL"
            self.logger.info(
                "  [Step %d] %s | %s | conf=%.2f",
                step.index,
                status,
                step.description,
                ev.confidence,
            )
            if not ev.success:
                self.logger.info(
                    "    reason=%s | fix=%s",
                    ev.failure_reason,
                    ev.fix_suggestion,
                )

    def run_task_bdi(
        self,
        task,
        max_cycles: int = 3,
        verbose: bool = False,
        close_memory_on_finish: bool = True,
        metrics_path: str = "data/experiments.csv",
        experiment_context: dict[str, Any] | None = None,
    ):
        """
        Execute run task bdi.
        
        Parameters
        ----------
        task : Any
            Task payload or metadata.
        max_cycles : Optional[int]
            Function argument.
        verbose : Optional[bool]
            Function argument.
        close_memory_on_finish : Optional[bool]
            Function argument.
        metrics_path : Optional[str]
            Filesystem path.
        experiment_context : Optional[dict[str, Any] | None]
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        """
        try:
            if task is None:
                raise ValueError(f"No task assigned to {self.name}")

            # --------------------------------------------------
            # Initialization
            # --------------------------------------------------

            # Initialize history manager for structured interaction logs
            self.history_manager = HistoryManager()

            # Initialize online TMS (Task Memory Structure)
            self.tms = OnlineTMS(
                grid=TMS_GRID,
                max_nodes_in_prompt=TMS_MAX_NODES_IN_PROMPT,
                max_anchors_per_obs=TMS_MAX_ANCHORS_PER_OBS,
            )

            # Initialize TRIM LLM for memory updates
            self.trim = TRIMLLM(
                gpt_client=self.tools.gpt_client,
                grid=TRIM_GRID,
                max_nodes_in_prompt=TRIM_MAX_NODES_IN_PROMPT,
                max_anchors_in_prompt=TRIM_MAX_ANCHORS_IN_PROMPT_OVERRIDE,
            )

            # Ensure TMS starts from a clean state
            self.tms.reset()

            self._last_task = task

            self.logger.info("Starting new task execution")
            self.logger.info("Task:\n%s", task["instruction"])

            # Reset environment and collect system information
            perception = self.env.reset(task)
            vm_info = self.tools.os_inspector.probe()
            os_info = vm_info["os"]

            # Create new episode metadata
            episode = Episode(
                episode_id=os.urandom(16).hex(),
                task=task,
                instruction=task.get("instruction"),
                status="STARTED",
                started_ts_ms=time.time_ns() // 1_000_000,
                os_name=os_info["pretty_name"],
                desktop_env=vm_info["desktop_environment"],
                display_server=vm_info["display_server"],
            )

            self.logger.info(
                "Episode started | episode_id=%s | task_id=%s",
                episode.episode_id,
                task.get("id"),
            )

            # --------------------------------------------------
            # Initial observation (belief bootstrap)
            # --------------------------------------------------

            last_observation = self.perception.process(perception)
            self.history_manager.update(last_observation, tags=["observation"])

            # Optional visualization of the initial UI state
            if verbose:
                visualize_ui_elements(
                    ui_dict=self.history_manager.last_observation.ui_elements,
                    screenshot=self.history_manager.last_observation.screenshot,
                    show_label=True,
                    include_label_text=False,
                    title="Initial UI Observation",
                )

            # --------------------------------------------------
            # Main BDI loop
            # --------------------------------------------------

            for cycle in range(int(max_cycles)):
                self.logger.info("=== Cycle %d / %d ===", cycle + 1, max_cycles)

                # ----------------------------
                # Planning (Desires)
                # ----------------------------

                action_chunk = self.planner.propose_next_steps(
                    task=task,
                    history_manager=self.history_manager,
                    system_info=vm_info,
                    tms=self.tms,
                    memory=self.memory_manager
                )

                # Log planner prompt and output
                self.history_manager.update(action_chunk, tags=["start_chunk"])

                self.logger.info(
                    "Planner produced chunk | macro_goal='%s' | decision=%s | steps=%d",
                    action_chunk.macro_goal,
                    action_chunk.decision,
                    len(action_chunk.steps),
                )

                # Stop execution if planner signals termination
                if action_chunk.decision in ("DONE", "FAIL"):
                    self.logger.info(
                        "Planner decided to %s. Ending loop.",
                        action_chunk.decision
                    )
                    break

                # ----------------------------
                # Execution (Intentions)
                # ----------------------------

                for k, step in enumerate(action_chunk.steps):
                    # Log each planned step before execution
                    self.history_manager.update(step, tags=["step"])

                    self.logger.debug(
                        "Executing step %d | %s | type=%s | pause=%.2f",
                        k,
                        step.description,
                        step.action_type,
                        step.pause,
                    )

                    # Execute the step in the environment
                    perception = self.executor.execute_step(step)

                    # Process resulting observation
                    last_observation = self.perception.process(perception)
                    self.history_manager.update(
                        last_observation,
                        tags=["observation", "observation_after_step"]
                    )

                    # Optional visualization after each step
                    if verbose:
                        visualize_ui_elements(
                            ui_dict=last_observation.ui_elements,
                            screenshot=last_observation.screenshot,
                            show_label=True,
                            include_label_text=False,
                            title=f"Observation after step {k + 1}/{len(action_chunk.steps)}",
                        )

                # ----------------------------
                # Evaluation (Judge)
                # ----------------------------

                self.logger.info("Evaluating executed chunk")
                last_evaluation = self.judge.evaluate_outcome(
                    self.history_manager
                )

                # Log judge prompt and evaluation

                self.history_manager.update(
                    {"chunk": action_chunk, "evaluation": last_evaluation},
                    tags=["end_chunk"]
                )

                # Log compact evaluation summary
                self._log_chunk_evaluation(self.history_manager.last_chunk)

                # ----------------------------
                # Memory update (TRIM + TMS)
                # ----------------------------

                self.logger.info("Running TRIM")
                trim_out = self.trim.run(
                    task_instruction=task.get("instruction", ""),
                    tms_nodes=self.tms.nodes(),
                    history_manager=self.history_manager,
                    current_observation=self.history_manager.last_observation,
                    chunk_digest=self.history_manager.last_chunk_digest_for_tms(),
                    cid=episode.episode_id,
                )
                self.history_manager.update(trim_out, tags=["trim_info"])

                # Apply TRIM output to update TMS
                self.logger.info("Updating TMS")
                self.tms.apply_trim_output(
                    trim_out,
                    self.history_manager.last_chunk,
                    self.history_manager.last_observation,
                )

                self.logger.info("Cycle %d completed\n", cycle + 1)

            # --------------------------------------------------
            # Final evaluation
            # --------------------------------------------------

            self.logger.info("Agent finished execution. Evaluating final result.")
            osworld_score = self.env.evaluate()

            # Store final episode score
            episode.score = {
                "status": osworld_score["status"],
                "metric": osworld_score["result"]["metric"],
                "success": osworld_score["result"]["success"],
            }
            episode.finished_ts_ms = time.time_ns() // 1_000_000
            episode.status = "DONE" if episode.score["success"] else "FAIL"

            # Compute and attach metrics (history-based)
            episode.score["stats"] = self.history_manager.compute_metrics(episode)
            settings_dump: dict[str, Any] = {}
            try:
                settings_dump = self.settings.model_dump()
            except Exception:
                pass

            settings_json = json.dumps(
                settings_dump,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            settings_hash = hashlib.sha1(settings_json.encode("utf-8")).hexdigest()

            run_context: dict[str, Any] = {
                "agent_name": self.name,
                "agent_settings_hash": settings_hash,
                "agent_settings_json": settings_json,
            }
            if experiment_context:
                run_context.update(experiment_context)

            append_metrics_csv(
                path=metrics_path,
                episode=episode,
                run_context=run_context,
            )

            # Persist episode and close memory manager
            self.memory_manager.ingest_end_of_episode(
                episode=episode,
                history_manager=self.history_manager,
                tms=self.tms
            )
            if close_memory_on_finish:
                self.memory_manager.close()

            self.logger.info("Final score: %s", episode.score)
            return episode

        except Exception:
            # Ensure memory is closed on failure and re-raise
            self.logger.error(
                "Error in loop catched. Interrupting execution."
            )
            try:
                if close_memory_on_finish and getattr(self, "memory_manager", None) is not None:
                    self.memory_manager.close()
            except Exception:
                pass
            raise
