import logging
from typing import Any

from agentz.pydantic_models import ActionChunk, ChunkEvaluation, Episode, ExecutedChunk, GPTClientRequest, Observation, Step, TRIMToolOutput, UIElement

from ..utils._formatters import (
    chunk_digest_for_tms,
    terminal_text_for_prompt,
    ui_elements_string,
    ui_elements_string_for_trim,
    ui_elements_string_full,
)
from ..utils._metrics import compute_episode_metrics, append_metrics_csv
from agentz.constants import TERMINAL_PROMPT_MAX_CHARS, TERMINAL_PROMPT_MAX_LINES, UI_FULL_MAX_ITEMS

class HistoryManager():

    def __init__(self):
        """
        Initialize `HistoryManager` dependencies and runtime state.
        
        Returns
        -------
        None
            No return value.
        """
        self.logger = logging.getLogger("HistoryManager")

        self.observations = {}
        
        self.chunks_history : list[ExecutedChunk]= []
        self.steps_history : list[Step] = []
        self.observations_history : list[Observation] = []

        # Alternate list [Observation, Step, Observation, ....]
        self.full_history : list[Step or Observation] = []
        self.trim_info : list[TRIMToolOutput] = []
        self.llm_requests : list[(str, GPTClientRequest)] = []
        self.active_chunk : ActionChunk = None
        self.active_chunk_first_observation : Observation = None

        self.chunk_window = (0,None)

        self.last_chunk : ExecutedChunk = None
        self.last_observation : Observation = None
        self.last_step : Step = None
        self.last_trim_info : TRIMToolOutput = None
        self.last_llm_request : GPTClientRequest = None

    def update(self, data : Any, tags : list[str]):

        """
        Update history state with the latest cycle artifacts.
        
        Parameters
        ----------
        data : Any
            Function argument.
        tags : list[str]
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
        """
        if "observation" in tags:
            
            obs : Observation = data
            self.last_observation = obs
            self.observations[obs.observation_id] = obs
            self.observations_history.append(obs)
            self.full_history.append(obs)

            if "observation_after_step" in tags:
                self.last_step.obs_after = self.last_observation.model_dump()

        elif "step" in tags:
            step : Step = data
            self.last_step = step
            self.last_step.obs_before = self.last_observation.model_dump()
            self.steps_history.append(step)
            self.full_history.append(step)
        
        elif "start_chunk" in tags:
            new_chunk : ActionChunk = data
            self.active_chunk = new_chunk
            self.active_chunk_first_observation = self.last_observation

            try:
                self.logger.info(
                    "Chunk start | macro_goal=%s | decision=%s | steps=%d",
                    new_chunk.macro_goal,
                    new_chunk.decision,
                    len(new_chunk.steps),
                )
            except Exception:
                pass
        
            # La last_observation è già stata pushata in full_history
            # e deve essere il primo elemento della history del chunk.
            self.chunk_start_idx = len(self.full_history) - 1


        elif "trim_info" in tags:
            trim_out : TRIMToolOutput = data
            self.last_trim_info = trim_out
            self.trim_info.append(trim_out)

        elif "llm_prompt" in tags:
            llm_req : GPTClientRequest = data["request"]
            entity : str = data["entity"]
            self.last_llm_request = llm_req
            self.llm_requests.append((entity, llm_req))
            
        elif "end_chunk" in tags:
            assert self.chunk_start_idx is not None, "end_chunk without start_chunk"

            executed_chunk : ActionChunk = data["chunk"]
            evaluation : ChunkEvaluation = data["evaluation"]

            steps = executed_chunk.steps
            steps_eval = evaluation.steps_eval


            self.last_chunk = ExecutedChunk(

                # From executed 
                macro_goal = executed_chunk.macro_goal,
                decision = executed_chunk.decision,
                steps = steps,
                steps_eval = steps_eval,
                
                # From evaluation
                overall_success = evaluation.overall_success,
                failing_step_index = evaluation.failing_step_index,
                planner_guidance = evaluation.planner_guidance,
                post_chunk_state = evaluation.post_chunk_state,
                failure_type = evaluation.failure_type,

                # First & Last observation
                first_observation = self.active_chunk_first_observation,
                last_observation = self.last_observation,
                
                history = self.full_history[self.chunk_start_idx : len(self.full_history)]
            )

            self.chunks_history.append(self.last_chunk)

            try:
                self.logger.info(
                    "Chunk end | success=%s | failure_type=%s | failing_step=%s",
                    evaluation.overall_success,
                    evaluation.failure_type,
                    evaluation.failing_step_index,
                )
            except Exception:
                pass
        
        else:
            raise ValueError(f"History update has invalid tags: {tags}")

    def get_active_chunk(self) -> ActionChunk | None:
        """
        Return get active chunk.
        
        Returns
        -------
        ActionChunk | None
            Function result.
        """
        return self.active_chunk

    def get_last_chunk(self) -> ExecutedChunk | None:
        """
        Return get last chunk.
        
        Returns
        -------
        ExecutedChunk | None
            Function result.
        """
        return self.last_chunk

    def get_last_observation(self) -> Observation | None:
        """
        Return get last observation.
        
        Returns
        -------
        Observation | None
            Function result.
        """
        return self.last_observation

    def compute_metrics(self, episode : Episode) -> dict[str, Any]:
        """
        Compute a compact metrics bundle for the completed episode.
        """
        return compute_episode_metrics(self, episode=episode)

    def chunks_digest(self):

        """
        Process chunks digest.
        
        Returns
        -------
        Any
            Function result.
        
        """
        digest : list[dict[str, Any]] = []
        lines : list[str] = []

        for i, ex_chunk in enumerate(self.chunks_history):

            lines.append(f"\n\nCHUNK {i+1}/{len(self.chunks_history)}:\n")
            lines.append(f"Goal: {ex_chunk.macro_goal}\n")
            lines.append(f"Task statu after chunk: {ex_chunk.decision}\n")
            lines.append(f"Success flag: {ex_chunk.overall_success}\n")
            lines.append(f"Failed step index: {ex_chunk.failing_step_index}\n")
            lines.append(f"Planner hints: {ex_chunk.planner_guidance}\n")
            lines.append(f"Chunk steps:\n")

            for i, (step, step_eval) in enumerate(zip(ex_chunk.steps, ex_chunk.steps_eval)):
                lines.append(f"- STEP {i} - {step.index} - {step_eval.index}")
                lines.append(f"  - Info pre execution")    
                lines.append(f"     description: {step.description}")    
                lines.append(f"     expected_outcome: {step.expected_outcome}")    
                lines.append(f"     action_type: {step.action_type}")    
                lines.append(f"     command: {step.command}")    
                lines.append(f"     pause: {step.pause}")    
                lines.append(f"  - Info post execution")    
                lines.append(f"     success: {step_eval.success}")    
                lines.append(f"     confidence in success: {step_eval.confidence}")    
                lines.append(f"     evidence: {step_eval.evidence}")    
                lines.append(f"     failure_reason: {step_eval.failure_reason}")    
                lines.append(f"     fix_suggestion: {step_eval.fix_suggestion}")    

            lines.append(f"UI Elements before chunk exection (screenshot n.1)")
            lines.append(self.ui_elements_string_for_trim(ex_chunk.first_observation.ui_elements))

            lines.append(f"UI Elements before chunk exection (screenshot n.2)")
            lines.append(self.ui_elements_string_for_trim(ex_chunk.last_observation.ui_elements))

        return "\n".join(lines)


    def ui_elements_string(self, ui_elements: dict[str, UIElement]) -> str:
        """
        Process ui elements string.
        
        Parameters
        ----------
        ui_elements : dict[str, UIElement]
            UI elements collection.
        
        Returns
        -------
        str
            Resulting string value.
        
        """
        return ui_elements_string(ui_elements)

    def ui_elements_string_full(self, ui_elements: dict[str, UIElement], max_items: int = UI_FULL_MAX_ITEMS) -> str:
        """
        Process ui elements string full.
        
        Parameters
        ----------
        ui_elements : dict[str, UIElement]
            UI elements collection.
        max_items : Optional[int]
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
        """
        return ui_elements_string_full(ui_elements, max_items=max_items)

    def ui_elements_string_for_trim(self, ui_elements: dict[str, UIElement]) -> str:
        """
        Process ui elements string for trim.
        
        Parameters
        ----------
        ui_elements : dict[str, UIElement]
            UI elements collection.
        
        Returns
        -------
        str
            Resulting string value.
        
        """
        return ui_elements_string_for_trim(ui_elements)

    def last_chunk_digest_for_tms(self) -> str:
        """
        Process last chunk digest for tms.
        
        Returns
        -------
        str
            Resulting string value.
        
        """
        return chunk_digest_for_tms(self.last_chunk)

    def terminal_text_for_prompt(
        self,
        terminal_content: Any,
        max_chars: int = TERMINAL_PROMPT_MAX_CHARS,
        max_lines: int = TERMINAL_PROMPT_MAX_LINES,
    ) -> str:
        """
        Process terminal text for prompt.
        
        Parameters
        ----------
        terminal_content : Any
            Function argument.
        max_chars : Optional[int]
            Maximum number of characters allowed.
        max_lines : Optional[int]
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
        """
        raw = "" if terminal_content is None else str(terminal_content)
        raw_len = len(raw.strip())
        norm = terminal_text_for_prompt(terminal_content, max_chars=max_chars, max_lines=max_lines)
        norm_len = len((norm or "").strip())
        if raw_len > 0 and norm_len == 0:
            self.logger.info(
                "History terminal | raw_len=%d | normalized_empty=True",
                raw_len,
            )
        return norm

