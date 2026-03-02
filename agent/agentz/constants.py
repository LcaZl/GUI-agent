"""Centralized constants used across agentz modules."""


from pathlib import Path

# Agent / runtime
TMS_GRID = 20  # Spatial anchor quantization grid (px); lower = finer anchors but noisier.
TMS_MAX_NODES_IN_PROMPT = 3  # Planner context size from TMS; keep small to avoid prompt bloat.
TMS_MAX_ANCHORS_PER_OBS = 64  # Cap anchors per observation; higher captures more UI detail.
TRIM_GRID = 20  # Anchor grid for TRIM; should match TMS grid for consistency.
TRIM_MAX_NODES_IN_PROMPT = 5  # Nodes shown to TRIM; too high increases confusion.
TRIM_MAX_ANCHORS_IN_PROMPT_OVERRIDE = 40  # Agent override for TRIM anchors; balance signal vs noise.
ENV_WAIT_READY_TIMEOUT_SEC = 180  # Max seconds to wait for env readiness before failing.

# ACI / OSWorld
OBS_SNIPPET_LEN = 400  # Max chars of observation text in error logs.
STEP_DEFAULT_PAUSE_SEC = 1.0  # Default pause after env.step.
STEP_DEFAULT_TIMEOUT_SEC = 30  # Default timeout for env.step.

# Judge
UI_DELTA_MAX_LABELS = 30  # Max labels shown in UI delta summary.
JUDGE_TEMPERATURE = 0.0  # Deterministic judge evaluation.

# Planning
EPISODIC_PER_QUERY_LIMIT = 15  # Max hits per query when retrieving episodic memory.
PLANNER_LAST_CHUNK_EVAL_MAX_CHARS = 800  # Max chars for last chunk evaluation injected into planner prompt.
DEFAULT_PRIORITY_FALLBACK = 99  # Sort fallback priority when query type is unknown.
PATTERN_POOL_CAP_MIN = 5  # Minimum pattern candidates to consider.
PATTERN_POOL_CAP_PER_CASE = 3  # Extra pattern pool per max_cases.
CHUNK_POOL_CAP_MIN = 3  # Minimum chunk candidates to consider.
CHUNK_POOL_CAP_PER_CASE = 2  # Extra chunk pool per max_cases.
MEMORY_UI_REF_FLOOR = 0.40  # Drop pattern candidates with weak UI similarity reference.
MEMORY_ADAPTIVE_STRICT_BEST_MIN = 0.65  # If best UI ref is >= this value, use strict memory gating.
MEMORY_ADAPTIVE_BALANCED_BEST_MIN = 0.45  # If best UI ref is >= this value, use balanced gating.
MEMORY_ADAPTIVE_STRICT_CLOSE_DELTA = 0.12  # In strict mode, keep only cases close to best UI ref.
MEMORY_ADAPTIVE_BALANCED_CLOSE_DELTA = 0.20  # In balanced mode, keep a wider neighborhood around best UI ref.
MEMORY_ADAPTIVE_STRICT_FAIL_CAP = 1  # Max fail-pattern hints in strict mode.
MEMORY_ADAPTIVE_BALANCED_FAIL_CAP = 1  # Max fail-pattern hints in balanced mode.
MEMORY_ADAPTIVE_WEAK_FAIL_CAP = 1  # Max fail-pattern hints in weak mode.
PLANNER_RETRIEVAL_PATTERN_LIMIT = 3  # Planner retrieval cap for pattern hits per query.
PLANNER_RETRIEVAL_CHUNK_LIMIT = 2  # Planner retrieval cap for chunk hits per query.
PLANNER_RETRIEVAL_STEP_LIMIT = 1  # Planner retrieval cap for step hits per query.
PLANNER_RETRIEVAL_EPISODE_LIMIT = 1  # Planner retrieval cap for episode hits per query.
PLANNER_RETRIEVAL_UI_LIMIT = 0  # Planner retrieval cap for UI hits per query.
MEMORY_PATTERN_SUCCESS_MIN_SEEN = 2  # Min seen_count for success patterns under normal transfer.
MEMORY_PATTERN_FAIL_MIN_SEEN = 2  # Min seen_count for fail patterns under normal transfer.
MEMORY_PATTERN_SUCCESS_HIGH_SIM_OVERRIDE = 0.78  # Accept singleton success pattern only above this UI ref.
MEMORY_PATTERN_FAIL_HIGH_SIM_OVERRIDE = 0.82  # Accept singleton fail pattern only above this UI ref.

# Memory: core
UI_FULL_MAX_ITEMS = 120  # Judge UI digest cap / full UI dump cap.
TERMINAL_PROMPT_MAX_CHARS = 600  # Default terminal chars for prompts.
TERMINAL_PROMPT_MAX_LINES = 40  # Default terminal lines for prompts.

# Memory: utils
MIN_WORD_LEN = 3  # Minimum token length for word-based Jaccard.
DEFAULT_MAX_VALUE_CHARS = None  # No truncation by default; rely on global prompt budgets.
PROJECT_MAX_VALUE_CHARS = None  # No truncation by default; rely on global prompt budgets.
TRIM_MAX_VALUE_CHARS = None  # No truncation by default; rely on global prompt budgets.
DEFAULT_MAX_ANCHORS = None  # No anchor cap by default; rely on global prompt budgets.
MIN_LABEL_LEN = None  # No minimum label length; keep short but informative labels.
APP_NAME_MAX_CHARS = None  # No app name truncation.
WINDOW_NAME_MAX_CHARS = None  # No window name truncation.
UI_MAX_ITEMS = 80  # Planner UI digest cap.
UI_TRIM_MAX_ITEMS = 80  # TRIM UI digest cap.
UI_TRIM_QUANTIZE_PX = 10.0  # Quantization step (px) for TRIM UI positions.
MAX_SIGNATURE_LEN = 8000  # Max chars for raw UI signature strings.
MAX_STABLE_ITEMS = 400  # Cap stable signature tokens.
MAX_TOKEN_LEN = 60  # Truncate individual tokens in stable signatures.
MAX_SIGNATURE_TOKENS = 400  # Cap token set size for similarity.
SHA1_SHORT_LEN = 10  # Length of short SHA1 prefix.
HASH_MIN_LEN = 10  # Min length for hex-like token scrubbing.

# Memory: episodic
FINGERPRINT_SIZE = 16  # Downsample size for image fingerprinting.
GRAY_R = 0.299  # Luma weight for R channel.
GRAY_G = 0.587  # Luma weight for G channel.
GRAY_B = 0.114  # Luma weight for B channel.
PNG_CLIP_MIN = 0  # Min clip for uint8 conversion.
PNG_CLIP_MAX = 255  # Max clip for uint8 conversion.
OBS_UI_COMPACT_MAX_ITEMS = 60  # UI items stored per observation snapshot.
SCRUB_MAX_WORDS = None  # No scrubbing limit by default.
HEX_HASH_MIN_LEN = 10  # Min hex length to scrub as hash-like token.
STABLE_UI_SIG_MAX_ITEMS = 50  # Max items in stable UI signatures.
STABLE_ACTION_MAX_STEPS = 12  # Max steps in action skeleton signature.
STEPS_TEXT_MAX_CHARS = None  # No truncation by default.
EPISODIC_SIGNATURE_MAX_LEN = None  # No truncation by default.
MEMORY_PATTERN_INGEST_MIN_STEPS = 1  # Skip pattern ingestion for chunks without executable steps.
MEMORY_PATTERN_INGEST_MIN_UI_TOKENS = 8  # Skip pattern ingestion when pre-UI signature is too sparse.

# Memory: episodic retrieval
FTS_MAX_TOKENS = 24  # Max tokens used in safe FTS query.
DEFAULT_SEARCH_LIMIT = 6  # Default number of hits to return.
QUERY_SNIPPET_LEN = 80  # Log snippet length for queries.
KIND_LIMIT_PATTERN_MIN = 2  # Minimum pattern hits.
KIND_LIMIT_PATTERN_MAX = 4  # Maximum pattern hits.
KIND_LIMIT_CHUNK_MIN = 2  # Minimum chunk hits.
KIND_LIMIT_CHUNK_MAX = 3  # Maximum chunk hits.
KIND_LIMIT_STEP_MIN = 2  # Minimum step hits.
KIND_LIMIT_STEP_MAX = 3  # Maximum step hits.
KIND_LIMIT_EPISODE_MIN = 1  # Minimum episode hits.
KIND_LIMIT_EPISODE_MAX = 2  # Maximum episode hits.
KIND_LIMIT_UI_MIN = 0  # Minimum UI hits.
KIND_LIMIT_UI_MAX = 2  # Maximum UI hits.
KIND_LIMIT_UI_DIVISOR = 5  # UI quota scaling factor.

# Memory: TMS / TRIM
ANCHOR_LABEL_MAX_CHARS = None  # No truncation; rely on global prompt budgets.
ANCHOR_ROLE_MAX_CHARS = None  # No truncation; rely on global prompt budgets.
TRIM_DEFAULT_GRID = 20  # Anchor grid for TRIM; should match TMS grid.
TRIM_MAX_ANCHORS_IN_PROMPT = 100  # Cap anchors shown to TRIM.
TRIM_PROJECT_MAX_VALUE_CHARS = None  # No truncation; rely on global prompt budgets.
TRIM_PROJECT_MAX_ANCHORS = None  # No cap; rely on global prompt budgets.
TRIM_NODES_MAX_VALUE_CHARS = None  # No truncation; rely on global prompt budgets.
TMS_TOKEN_BUDGET_CHARS = 2400  # Approx prompt budget for TMS context.
TMS_NODE_ID_LEN = 10  # Length of generated node ids (hex).
TMS_ANCHOR_CAP = 120  # Max anchors stored per node.
TMS_INACTIVE_KEEP_SCORE = 0.65  # Keep inactive nodes if relevance >= threshold.
TMS_FIND_NODE_SIM_THRESHOLD = 0.85  # Title similarity threshold for dedup.
TMS_DIVERSITY_TITLE_SIM = 0.85  # Title similarity for diversity filtering.
TMS_DIVERSITY_VALUE_SIM = 0.85  # Value similarity for diversity filtering.
TMS_EST_SIZE_OVERHEAD = 80  # Rough overhead per node for budget estimation.
TMS_RECENCY_DECAY = 0.35  # Recency decay factor (higher = faster decay).
TMS_SCORE_LEXICAL_W = 0.55  # Weight of lexical similarity in relevance.
TMS_SCORE_ANCHOR_W = 0.35  # Weight of anchor overlap in relevance.
TMS_SCORE_RECENCY_W = 0.10  # Weight of recency in relevance.
TMS_INACTIVE_PENALTY = 0.65  # Penalty applied to inactive node scores.
TMS_INACTIVATE_MIN_FAIL_STREAK = 2  # Min consecutive failures before INACTIVATE is applied.
TMS_ANCHOR_MAX_ITEMS = 8  # Max anchors shown in planner context.
TMS_PROJECT_MAX_VALUE_CHARS = 220  # Truncation cap for node values in planner context.
TMS_PROJECT_MAX_ANCHORS = 12  # Cap anchors per node in planner context.
TMS_JACCARD_MIN_LEN = 3  # Min token length for Jaccard comparisons.

# Perception
VISIBLE_BONUS = 100  # Rank boost for visible elements.
AREA_NORM_WEIGHT = 10  # Weight for normalized area contribution.
DEPTH_NORM_PENALTY = 0.01  # Penalty for deep a11y nodes.
SRC_PREF_WEIGHT = 1_000_000  # Weight for source preference.
INTERACTIVE_WEIGHT_SCHEMA = 10_000  # Weight for interactive elements (schema scoring).
ENABLED_WEIGHT_SCHEMA = 1_000  # Weight for enabled elements (schema scoring).
AREA_DENOM_FLOOR = 1.0  # Avoid div-by-zero in normalization.
DEPTH_DEFAULT = 1e9  # Default depth when missing.
SCORE_DEFAULT = -1.0  # Default score when missing.
DEFAULT_MATCH_COLUMN = "name_norm"  # Default column for matching.
MIN_FUSION_SCORE = 0.55  # Minimum fusion score to accept a match.
MIN_IOU_CANDIDATE = 0.10  # IoU threshold for candidate pairing.
MAX_CENTER_DIST_PX = 60.0  # Max center distance for positional match.
MIN_POS_ONLY_SCORE = 0.35  # Minimum positional-only score.
MIN_TXT_ONLY_SCORE = 0.82  # Minimum text-only similarity when a11y geometry is missing.
DEDUP_IOU_SAME_LABEL = 0.90  # Dedup threshold for same-label boxes.
UI_DEDUP_MAX_PER_LABEL = 4  # Max kept elements per normalized label after dedup.
VISION_A11Y_SUPPRESS_IOU = 0.70  # Drop vision-only rows if IoU with an a11y-backed row is above this threshold.
VISION_A11Y_SUPPRESS_MIN_OVERLAP = 0.85  # Drop vision-only rows if one box is mostly contained in an a11y-backed box.
VISION_A11Y_SUPPRESS_MIN_TARGET_COVER = 0.40  # For containment suppression, require overlap to also cover a meaningful portion of the a11y target.
VISION_A11Y_SUPPRESS_CENTER_DIST_PX = 18.0  # Drop vision-only rows when centers are nearly coincident with a11y-backed rows.
INTERACTIVE_WEIGHT_FUSION = 10.0  # Weight for a11y interactivity.
ENABLED_WEIGHT_FUSION = 2.0  # Weight for enabled state.
SHOWING_WEIGHT = 1.0  # Weight for showing/visible state.
FUSION_WEIGHT = 1.0  # Weight for existing fusion score.
VISION_WEIGHT = 0.2  # Weight for vision confidence.
POS_WEIGHT = 0.6  # Weight for position similarity.
TXT_WEIGHT = 0.4  # Weight for text similarity.
VISION_SCORE_MIN_ICON_TEXT = 0.55  # Min vision score for icon/text elements.
STATE_TOKENS_MAX_N = 12  # Max a11y state tokens retained.
ROUND_COORDS_NDIGITS = 0  # Rounding precision for coords (0 = integer).
DEFAULT_DEBUG_DIR = Path("../data/debug/processed_screenshots")  # Default debug output path.
DEFAULT_VISION_PREFIX = "vision"  # Prefix for vision debug files.
DEFAULT_FUSED_PREFIX = "fused"  # Prefix for fused debug files.
OCR_MIN_CONF = 0.70  # Minimum OCR confidence to keep text.
OCR_MIN_LEN = 3  # Minimum OCR text length.
OCR_MAX_LEN = 80  # Maximum OCR text length.
CAPTION_ICON_MIN_CONF = 0.55  # Minimum icon confidence for captioning.
CAPTION_MAX_CHARS = None  # No caption length limit by default.
CAPTION_MAX_WORDS = None  # No caption word limit by default.
IOU_EPS_DEFAULT = 1e-9  # Epsilon to stabilize IoU computation.
ICON_COLOR = (0, 255, 0)  # Box color for icons in debug viz.
TEXT_COLOR = (255, 0, 0)  # Box color for text in debug viz.
NAME_MIN_LEN = 1  # Minimum length for a11y names to keep.
A11Y_TEXT_ROLE_MIN_LEN = 1  # Min length for non-interactive label/text roles.
NODE_ID_PREFIX = "node"  # Prefix for synthetic a11y node ids.

# Tools: OpenAI client
ALLOWED_OVERRIDES = {  # Supported per-request override keys.
    "model",
    "temperature",
    "reasoning",
    "max_retries",
    "bind_tools_kwargs",
}
AZURE_ENDPOINT = "https://rt-bdi-zanolo.openai.azure.com/"  # Azure OpenAI endpoint.
AZURE_API_VERSION = "2024-12-01-preview"  # Azure API version for requests.
DEFAULT_IMAGE_FORMAT = "JPEG"  # Default image format for numpy conversion.
DEFAULT_JPEG_QUALITY = 90  # JPEG quality for numpy images.
MAX_VISION_IMAGES = 10  # Max images per vision request (Azure limit).
DEFAULT_MAX_TOKENS = 4096  # Default max_tokens for chat calls.
MODEL_PROMPT_RATIO_DEFAULT = (85, 170)  # Base/tile tokens for default vision model.
MODEL_PROMPT_MULTIPLIERS = {  # Patch-model multipliers by family.
    "gpt-5-mini": 1.62,
    "gpt-5-nano": 2.46,
    "gpt-4.1-mini": 1.62,
    "gpt-4.1-nano": 2.46,
    "o4-mini": 1.72,
}
PATCH_SIZE = 32  # Patch size for patch-based image tokenization.
MAX_PATCHES = 1536  # Max patches for patch-based tokenization.
MAX_DIM_STAGE1 = 2048.0  # Stage-1 max dimension for high detail.
MAX_SHORTEST_DIM = 768.0  # Stage-2 shortest side target for high detail.
TILE_SIZE = 512.0  # Tile size for tile-based tokenization.
BASE_TILE_GPT5 = (70, 140)  # Base/tile tokens for GPT-5 family.
BASE_TILE_4O_MINI = (2833, 5667)  # Base/tile tokens for 4o-mini.
BASE_TILE_O1 = (75, 150)  # Base/tile tokens for o1/o3 family.
BASE_TILE_COMPUTER_USE = (65, 129)  # Base/tile tokens for computer-use.
CLIP_MIN = 0  # Min pixel clamp for numpy images.
CLIP_MAX = 255  # Max pixel clamp for numpy images.
FLOAT_NORM_MAX = 1.0  # Max float value considered normalized.

# Tools: OpenAI interaction logging
LOG_FIELDNAMES = [  # CSV column order for interaction logs.
    "timestamp",
    "prompt",
    "tool_name",
    "tool_args",
    "model_name",
    "token_count_prompt",
    "token_count_completion",
    "success",
    "error_message",
    "elapsed_time_sec",
    # Vision/cost tracking
    "image_count",
    "image_detail",
    "image_total_bytes",
    "image_est_tokens_low",
    "image_est_tokens_high",
    # API usage (if provided by service)
    "api_prompt_tokens",
    "api_completion_tokens",
    "api_total_tokens",
    "api_cached_tokens",
]
PROMPT_SNIPPET_LEN = 500  # Max chars saved for prompt logging.
ELAPSED_TIME_ROUND_DIGITS = 3  # Rounding precision for elapsed time.
API_TOKENS_DEFAULT = -1  # Default when API usage is missing.
ELAPSED_TIME_DEFAULT = -1.0  # Default when elapsed time is missing.

# Utility: output / visualization
DF_PRINT_FIGSIZE = (12, 8)  # Default figure size for dataframes.
DF_DEFAULT_LIMIT = 30  # Default row limit in dataframe prints.
DF_PREVIEW_LINES = 30  # Preview lines for accessibility logs.
DRAW_MAX_BOXES = 50  # Max boxes drawn for debug screenshots.
TERMINAL_PREVIEW_MAX_CHARS = 500  # Terminal preview truncation length.
TITLE_FONTSIZE = 12  # Default title font size.
MAX_LABEL_CHARS = 22  # Max label length for UI annotations.
BOX_LINEWIDTH = 2.5  # Line width for UI boxes.
FIGSIZE_MIN_W = 8  # Minimum figure width.
FIGSIZE_MIN_H = 6  # Minimum figure height.
FIGSIZE_DENOM = 140  # Size scaling denominator for dynamic figsize.
TRANSITION_FIGSIZE = (12, 10)  # Default before/after figure size.
TRUNC_MAX_STR = 180  # Max length for text truncation in history.
SUMMARY_MAX_LEN = 120  # Max length for summary strings.
PRINT_DIVIDER_LEN = 88  # Divider length for history display.
PRINT_DIVIDER_LEN_LARGE = 90  # Divider length for banners.
ITEM_TRUNC_LEN = 300  # Max length for item repr truncation.
PAD_DEFAULT = 2.0  # Padding for label backgrounds.
ALPHA_DEFAULT = 0.90  # Alpha for label backgrounds.
TEXT_OFFSET_X = 1.0  # Label x offset for readability.
TEXT_OFFSET_Y = 1.0  # Label y offset for readability.
SHOW_IMG_FIGSIZE = (10, 6)  # Default figure size for images.
OUTLINE_COLOR = (255, 0, 0)  # Default outline color for boxes.
RECT_LINEWIDTH = 1  # Rectangle line width for debug boxes.
PIXEL_MIN = 0.0  # Minimum pixel coordinate.
PIXEL_MAX_OFFSET = 1.0  # Max offset to keep boxes inside image.
BOX_MIN_SIZE = 1.0  # Minimum width/height for label boxes.
LABEL_TRUNC_LEN = 60  # Max length for label text in summaries.
LARGE_CONTAINER_LEN = 20  # Threshold to summarize large containers.
