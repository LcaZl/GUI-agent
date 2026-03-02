from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

class BBCoords(BaseModel):
    x_1: float
    y_1: float
    x_2: float
    y_2: float
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class CenterCoords(BaseModel):
    x: float
    y: float
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class UIElement(BaseModel):
    """
    Compact UI element used by the planner prompt.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(..., description="Symbolic UI id (e.g., ui_12)")
    kind: Optional[str] = Field(default=None, description="Element kind: node|text|icon|...")
    label: str = Field(default="", description="Human-facing label/content")
    value: Optional[str] = Field(default=None, description="Optional value/text payload (when provided by a11y)")
    source: Optional[Literal["a11y", "vision", "fusion"]] = Field(
        default=None,
        description="Origin of the element: a11y|vision|fusion",
    )
    a11y_role: Optional[str] = Field(default=None, description="Accessibility role if available")

    visible: Optional[bool] = Field(default=None, description="From a11y_visible when present")
    enabled: Optional[bool] = Field(default=None, description="From a11y_enabled when present")
    actionable: Optional[bool] = Field(default=None, description="Heuristic: clickable/actionable")

    focused: Optional[bool] = Field(default=None, description="A11y focused state if available")
    selected: Optional[bool] = Field(default=None, description="A11y selected state if available")
    checked: Optional[bool] = Field(default=None, description="A11y checked state if available")
    expanded: Optional[bool] = Field(default=None, description="A11y expanded state if available")

    actions: Optional[List[str]] = Field(default=None, description="Short list of available a11y actions (trimmed)")
    states: Optional[List[str]] = Field(default=None, description="Short list of a11y states (trimmed)")

    a11y_id: Optional[str] = Field(default=None, description="Stable a11y id if present")
    a11y_node_id: Optional[str] = Field(default=None, description="Extractor-local node id")
    a11y_parent_id: Optional[str] = Field(default=None, description="Extractor-local parent id")
    a11y_depth: Optional[int] = Field(default=None, description="Depth in a11y tree (root=0)")
    a11y_child_index: Optional[int] = Field(default=None, description="Index among siblings")

    score: Optional[float] = Field(default=None, description="Generic score (if any)")
    vision_score: Optional[float] = Field(default=None, description="Vision confidence (OCR/YOLO)")
    fusion_score: Optional[float] = Field(default=None, description="Vision<->a11y match score (if matched)")
    fusion_matched: Optional[bool] = Field(default=None, description="Whether this row had a fusion match")

    app_name: Optional[str] = Field(default=None, description="Owning application name (from a11y ancestors)")
    window_name: Optional[str] = Field(default=None, description="Owning window/frame name (from a11y ancestors)")
    window_active: Optional[bool] = Field(default=None, description="Whether owning window/frame is active")

    bb_coords: "BBCoords" = Field(..., description="Bounding box coordinates")
    center_coords: "CenterCoords" = Field(..., description="Bounding box center coordinates")

class Observation(BaseModel):
    """
    Schema dell'osservazione percepita (output di PerceptionInterface.process).
    Tutti opzionali tranne screenshot e ui_elements.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    observation_id: Optional[int] = Field(default=None, description="ID univoco incrementale dell'osservazione")
    screen_info: Optional[Any] = None
    ally_info: Optional[Any] = None
    a11y_raw: Optional[Any] = None
    fused_data: Optional[Any] = None

    ui_elements: Dict[str, UIElement] = Field(..., description="Dizionario di UIElement indicizzati per id")
    screenshot: Any = Field(..., description="Screenshot dell'osservazione (array/bytes/PIL/etc.)")
    terminal_content : str = Field("", description="Content of the terminal windows, if opened on the virtual machine.")
    
    reward: Optional[float] = None
    info: Optional[Any] = None
    done: Optional[bool] = None

## EVALUATION

class FailureType(str, Enum):
    ACTION_INEFFECTIVE = "ACTION_INEFFECTIVE"
    WRONG_TARGET = "WRONG_TARGET"
    UI_NOT_READY = "UI_NOT_READY"
    ENV_LIMITATION = "ENV_LIMITATION"
    UNCLEAR = "UNCLEAR"

class StepEvaluation(BaseModel):
    index: int = Field(..., ge=0, description="0-based index of the step in the chunk.")
    success: bool = Field(..., description="Whether the step achieved its expected outcome.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Judge confidence based on available evidence.")
    evidence: str = Field(..., min_length=1, description="Compact evidence grounded in before/after screenshots and UI elements (no speculation).")
    failure_reason: Optional[str] = Field(default=None, description="If failed: precise reason (what did not happen / what unexpected thing happened).")
    fix_suggestion: Optional[str] = Field(default=None, description="If failed: precise next action(s) to recover, written as planner-ready guidance.")
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class ChunkEvaluation(BaseModel):
    overall_success: bool = Field(..., description="True only if the chunk achieved its intended immediate goal.")
    steps_eval: List[StepEvaluation] = Field(default_factory=list)
    failing_step_index: Optional[int] = Field(default=None, description="If not SUCCESS: 0-based index of the first step that likely failed. Null if unclear.")
    planner_guidance: str = Field(..., min_length=1, description="Planner-ready guidance for the next call. If success: what state we are in now. If failure: what to do next to recover and avoid repeating the same mistake.")
    post_chunk_state: str = Field(..., min_length=1, description="Single concise sentence describing what is now true about the UI or environment state after the chunk.")
    failure_type: Optional[FailureType] = Field(default=None, description="If overall_success is false: coarse classification of the primary failure cause.")

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

## EXECUTION

class Step(BaseModel):
    """Single step: executable intent + compact evidence pointers (optional)."""
    index: int = Field(..., ge=0, description="0-based index of the step in the chunk.")
    description: str = Field(..., min_length=1, description="Short, precise action description (prompt-ready).")
    expected_outcome: str = Field(..., min_length=1, description="Short expected result after execution (prompt-ready).")
    action_type: str = Field(default=None, description="Action type (WAIT, DONE, FAIL, python).")
    command: Optional[str] = Field(default=None, description="Python code to execute when kind=python.")
    pause: float = Field(default=0, ge=0.0, description="Seconds to wait AFTER the step. Used for python steps and WAIT steps.")
    obs_before: SkipJsonSchema[Optional[Observation]] = None
    obs_after: SkipJsonSchema[Optional[Observation]] = None

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class ActionChunk(BaseModel):
    """Next 1..K steps to execute now."""
    macro_goal: str = Field(..., min_length=1, description="Immediate macro objective (prompt-ready).")
    decision: Literal["CONTINUE", "DONE", "FAIL"] = Field(..., description="Planner decision after this chunk.")
    steps: List[Step] = Field(default_factory=list, description="Planned next steps.")
    
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

## STORAGE
class ExecutedChunk(BaseModel):

    macro_goal: str = Field(..., min_length=1, description="Immediate macro objective (prompt-ready).")
    decision: Literal["CONTINUE", "DONE", "FAIL"] = Field(..., description="Planner decision after this chunk.")
    steps: List[Step] = Field(default_factory=list, description="Planned next steps.")
    steps_eval: List[StepEvaluation] = Field(default_factory=list)
    
    overall_success: bool = Field(..., description="True only if the chunk achieved its intended immediate goal.")
    failing_step_index: Optional[int] = Field(default=None, description="If not SUCCESS: 0-based index of the first step that likely failed. Null if unclear.")
    planner_guidance: str = Field(..., min_length=1, description="Planner-ready guidance for the next call. If success: what state we are in now. If failure: what to do next to recover and avoid repeating the same mistake.")
    post_chunk_state: str = Field(..., min_length=1, description="Single concise sentence describing what is now true about the UI or environment state after the chunk.")
    failure_type: Optional[FailureType] = Field(default=None, description="If overall_success is false: coarse classification of the primary failure cause.")
    
    first_observation : Observation 
    last_observation : Observation

    history: List[Union[Observation, Step]]


    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class Episode(BaseModel):
    episode_id: str = Field(..., min_length=1, description="Unique episode id.") 
    task: Dict[str, Any] = Field(default=None, description="Task identifier if available.") 
    instruction: str = Field(..., min_length=1, description="User instruction.") 
    status: Literal["STARTED", "DONE", "FAIL", "ABORT"] = Field(default="STARTED", description="Episode termination.") 
    started_ts_ms: int = Field(..., ge=0, description="Start timestamp (ms).") 
    os_name: str 
    desktop_env: str 
    display_server: str 
    finished_ts_ms: Optional[int] = Field(default=None, ge=0, description="Finish timestamp (ms).") 

    score : Optional[dict[str, Any]] = None
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
