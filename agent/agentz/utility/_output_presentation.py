from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import os, io, numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except Exception:
    def display(obj):
        """
        Display rich output when notebook rendering is available.
        
        Parameters
        ----------
        obj : Any
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
        """
        print(obj)

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec


import numpy as np
import pandas as pd
from agentz.constants import (
    ALPHA_DEFAULT,
    BOX_LINEWIDTH,
    BOX_MIN_SIZE,
    DF_DEFAULT_LIMIT,
    DF_PREVIEW_LINES,
    DF_PRINT_FIGSIZE,
    DRAW_MAX_BOXES,
    FIGSIZE_DENOM,
    FIGSIZE_MIN_H,
    FIGSIZE_MIN_W,
    ITEM_TRUNC_LEN,
    LABEL_TRUNC_LEN,
    LARGE_CONTAINER_LEN,
    MAX_LABEL_CHARS,
    OUTLINE_COLOR,
    PAD_DEFAULT,
    PIXEL_MAX_OFFSET,
    PIXEL_MIN,
    PRINT_DIVIDER_LEN,
    PRINT_DIVIDER_LEN_LARGE,
    RECT_LINEWIDTH,
    SHOW_IMG_FIGSIZE,
    SUMMARY_MAX_LEN,
    TERMINAL_PREVIEW_MAX_CHARS,
    TEXT_OFFSET_X,
    TEXT_OFFSET_Y,
    TITLE_FONTSIZE,
    TRANSITION_FIGSIZE,
    TRUNC_MAX_STR,
)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from agentz.pydantic_models import (
    TRIMToolOutput,
    TRIMSubtaskDecision,
    TRIMIntent,
    TMSOp,
    NodeStatus,
)

from agentz.memory import OnlineTMS

import matplotlib.pyplot as plt

def show_screenshot(screenshot):
    """
    Show screenshot.
        
        Parameters
        ----------
        screenshot : Any
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
    """
    plt.figure(figsize=DF_PRINT_FIGSIZE)  # increase display size in inches
    plt.imshow(screenshot)
    plt.axis("off")
    plt.title("Desktop frame (display only)")
    plt.show()

def print_list(lis, title = ""):
    """
    Print a list with a title 
    """
    print(title)
    if len(lis) == 0:
        print(" -- Empty list.")
    else:
        for i, el in enumerate(lis, 1):
            print(f" - {i} - {el}")
 
def print_map(config_map, title = None):
    """
    Print a dict with a title 
    """
    if title:
        print(title)
        
    for i, (key, value) in enumerate(config_map.items(), 1):
        print(f"{i} - {key} : {value}")
        
def print_dataframe(data, title=None, limit=DF_DEFAULT_LIMIT, sort_by=None, ascending=True, show_index=True):
    
    """
    Print dataframe.
        
        Parameters
        ----------
        data : Any
            Function argument.
        title : Optional[Any]
            Function argument.
        limit : Optional[Any]
            Function argument.
        sort_by : Optional[Any]
            Function argument.
        ascending : Optional[Any]
            Function argument.
        show_index : Optional[Any]
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
    """
    if title is not None:
        print(title)
    pd.set_option('display.float_format', '{:.6f}'.format)
    
    # Check if data is a Series and convert to DataFrame
    if isinstance(data, pd.Series):
        
        # If the Series has a name, use it as the column name; otherwise, default to a generic column name.
        column_name = data.name if data.name is not None else 'Value'
        # Convert Series to DataFrame for uniform handling
        data = data.to_frame(name=column_name)

    if sort_by is not None and isinstance(data, pd.DataFrame):
        data = data.sort_values(by=sort_by, ascending=ascending)
    
    print(tabulate(data[:limit], headers='keys', tablefmt='simple_grid', showindex=show_index))
    print('\n')
    
from typing import Union, Dict, Any, List, Optional

def print_dict(
    d: Union[Dict[Any, Any], List[Any], str],
    title: str = '',
    avoid_keys: Optional[List[str]] = None,
    indent: int = 0
) -> None:
    """
    Recursively prints a formatted dictionary or list with indentation and optional title.
    """
    if avoid_keys is None:
        avoid_keys = []

    indent_str = '    ' * indent

    # Title formatting
    if title:
        line = f"\n{indent_str}== {title.strip()} =="
        print(line)

    if isinstance(d, dict):
        for key, value in d.items():
            if key in avoid_keys:
                continue
            if isinstance(value, dict):
                print(f"{indent_str}- {key}:")
                print_dict(value, avoid_keys=avoid_keys, indent=indent + 1)
            elif isinstance(value, list):
                print(f"{indent_str}- {key}:")
                for item in value:
                    if isinstance(item, (dict, list)):
                        print_dict(item, avoid_keys=avoid_keys, indent=indent + 1)
                    else:
                        cleaned_item = str(item).replace('\n', ' ').replace('\r', ' ').strip()
                        print(f"{indent_str}    • {cleaned_item}")
            else:
                cleaned_value = str(value).replace('\n', ' ').replace('\r', ' ').strip()
                print(f"{indent_str}- {key}: {cleaned_value}")
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, (dict, list)):
                print_dict(item, avoid_keys=avoid_keys, indent=indent)
            else:
                cleaned_item = str(item).replace('\n', ' ').replace('\r', ' ').strip()
                print(f"{indent_str}• {cleaned_item}")
    else:
        cleaned_data = str(d).replace('\n', ' ').replace('\r', ' ').strip()
        print(f"{indent_str}{cleaned_data}")

def show_and_store_prepared_data(data, base_path):

    """
    Show and store prepared data.
        
        Parameters
        ----------
        data : Any
            Function argument.
        base_path : Any
            Filesystem path.
        
        Returns
        -------
        Any
            Function result.
        
    """
    print(f"prepared data actual keys: {data.keys()}")


    tsv = data.get("a11y_tsv", "") or ""
    boxes = data.get("a11y_bboxes", []) or []
    rows = [r.split("\t") for r in tsv.splitlines()[1:]] if tsv else []  # skip header

    print(f"Rows in TSV: {len(rows)} | BBoxes: {len(boxes)}")
    assert len(rows) == len(boxes), "Mismatch: TSV rows != number of bounding boxes"

    for i, (r, bb) in enumerate(zip(rows, boxes)):
        if i >= DF_PREVIEW_LINES: break
        tag, name, text = r[1], r[2], r[3] if len(r) > 3 else ""
        print(f"[{i}] tag={tag} | name={name!r} | text={text!r} | box={tuple(bb)}")


    # Robust keys: prefer ACI TSV, fallback to original XML
    s = data.get("screenshot")
    a11y = data.get("a11y_tsv") or data.get("accessibility_tree")
    bboxes = data.get("a11y_bboxes")
    top_app = data.get("top_app")
    t = data.get("terminal")
    ins = data.get("instruction")

    # ---- Screenshot info + save
    if s is not None:
        if isinstance(s, (bytes, bytearray, memoryview)):
            s = np.array(Image.open(io.BytesIO(s)).convert("RGB"))
        elif isinstance(s, Image.Image):
            s = np.array(s.convert("RGB"))
        if isinstance(s, np.ndarray):
            print(f"[screenshot] shape={s.shape} dtype={s.dtype}")
            Image.fromarray(s).save(f"{base_path}/debug_screenshot.png")
            print("Saved: debug_screenshot.png")

    # ---- Accessibility preview (first 30 lines)
    if a11y:
        print("\n[accessibility] preview:")
        lines = a11y.splitlines()
        for ln in lines[:DF_PREVIEW_LINES]:
            print(ln)
        if len(lines) > DF_PREVIEW_LINES:
            print(f"... ({len(lines) - DF_PREVIEW_LINES} more lines)")
    else:
        print("\n[accessibility] none")

    # ---- Draw up to 50 boxes (if available) and save
    if isinstance(s, np.ndarray) and bboxes:
        im = Image.fromarray(s.copy())
        dr = ImageDraw.Draw(im)
        for i, (x1, y1, x2, y2) in enumerate(bboxes[:DRAW_MAX_BOXES]):
            # Use a thin outline to avoid clutter
            dr.rectangle([x1, y1, x2, y2], outline=OUTLINE_COLOR, width=RECT_LINEWIDTH)
        show_screenshot(im)
        im.save(f"{base_path}/debug_screenshot_with_boxes.png")
        print("Saved: debug_screenshot_with_boxes.png")
        print(f"[a11y_bboxes] count={len(bboxes)} (showing up to {DRAW_MAX_BOXES})")

    # ---- Top app (if provided)
    if top_app:
        print(f"\n[top_app] {top_app}")

    # ---- Terminal (truncate to 500 chars)
    if t:
        ts = str(t)
        print("\n[terminal]")
        print(ts if len(ts) <= TERMINAL_PREVIEW_MAX_CHARS else ts[:TERMINAL_PREVIEW_MAX_CHARS] + " ...")

    # ---- Instruction
    if ins:
        print("\n[instruction]")
        print(ins)


def visualize_ui_elements(
    ui_dict: dict,
    screenshot: np.ndarray,
    title: str = "UI Elements (a11y)",
    title_fontsize: int = TITLE_FONTSIZE,
    show_label: bool = True,
    include_label_text: bool = False,
    max_label_chars: int = MAX_LABEL_CHARS,
    figsize=None,
    box_color: str = "#00FF66",
    label_bg_color: str = "#00FF66",
    text_color: str = "#00FF66",
    box_linewidth: float = BOX_LINEWIDTH,
) -> pd.DataFrame:
    """
    Visualize ui elements.
        
        Parameters
        ----------
        ui_dict : dict
            Function argument.
        screenshot : np.ndarray
            Function argument.
        title : Optional[str]
            Function argument.
        title_fontsize : Optional[int]
            Function argument.
        show_label : Optional[bool]
            Function argument.
        include_label_text : Optional[bool]
            Function argument.
        max_label_chars : Optional[int]
            Function argument.
        figsize : Optional[Any]
            Function argument.
        box_color : Optional[str]
            Function argument.
        label_bg_color : Optional[str]
            Function argument.
        text_color : Optional[str]
            Function argument.
        box_linewidth : Optional[float]
            Function argument.
        
        Returns
        -------
        pd.DataFrame
            Function result.
        
    """
    if screenshot is None:
        raise ValueError("screenshot is None")
    if not isinstance(screenshot, np.ndarray):
        raise TypeError("screenshot must be a numpy ndarray")
    if ui_dict is None or len(ui_dict) == 0:
        raise ValueError("ui_dict is empty")

    img = screenshot
    if img.ndim == 2:
        pass
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        pass
    else:
        raise ValueError(f"Unexpected screenshot shape: {img.shape}")

    h, w = img.shape[0], img.shape[1]
    if figsize is None:
        figsize = (max(FIGSIZE_MIN_W, w / FIGSIZE_DENOM), max(FIGSIZE_MIN_H, h / FIGSIZE_DENOM))

    df = pd.DataFrame([el.model_dump() for el in ui_dict.values()])
    if df.empty:
        raise ValueError("No elements had bb_coords; dataframe is empty")

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    for _, r in df.iterrows():

        bb = r["bb_coords"]

        x1, y1, x2, y2 = float(bb["x_1"]), float(bb["y_1"]), float(bb["x_2"]), float(bb["y_2"])

        x1c = max(PIXEL_MIN, min(w - PIXEL_MAX_OFFSET, x1))
        y1c = max(PIXEL_MIN, min(h - PIXEL_MAX_OFFSET, y1))
        x2c = max(PIXEL_MIN, min(w - PIXEL_MAX_OFFSET, x2))
        y2c = max(PIXEL_MIN, min(h - PIXEL_MAX_OFFSET, y2))

        if x2c < x1c:
            x1c, x2c = x2c, x1c
        if y2c < y1c:
            y1c, y2c = y2c, y1c

        ax.add_patch(
            Rectangle(
                (x1c, y1c),
                x2c - x1c,
                y2c - y1c,
                fill=False,
                linewidth=box_linewidth,
                edgecolor=box_color,
                zorder=4,
            )
        )
        if show_label:
            # --- build label text ---
            text = str(r["id"])
            if include_label_text:
                lbl = str(r.get("label") or "").replace("\n", " ").strip()
                if len(lbl) > max_label_chars:
                    lbl = lbl[: max_label_chars - 1] + "…"
                if lbl:
                    text = f"{text} {lbl}"

            # --- helpers: measure text size in DATA coords (same units as image pixels) ---
            def _measure_text_size_data(ax, fig, s, fontsize=7):
                """
                Process measure text size data.
                                
                                Parameters
                                ----------
                                ax : Any
                                    Function argument.
                                fig : Any
                                    Function argument.
                                s : Any
                                    Function argument.
                                fontsize : Optional[Any]
                                    Function argument.
                                
                                Returns
                                -------
                                Any
                                    Function result.
                                
                """
                ttmp = ax.text(0, 0, s, fontsize=fontsize, ha="left", va="top", alpha=0.0)
                fig.canvas.draw()
                rend = fig.canvas.get_renderer()
                bb = ttmp.get_window_extent(renderer=rend)  # display coords
                inv = ax.transData.inverted()
                (dx0, dy0) = inv.transform((bb.x0, bb.y0))
                (dx1, dy1) = inv.transform((bb.x1, bb.y1))
                ttmp.remove()
                return abs(dx1 - dx0), abs(dy1 - dy0)

            label_w, label_h = _measure_text_size_data(ax, fig, text, fontsize=7)

            pad = PAD_DEFAULT             # padding background
            margin = 2.0                  # distanza label <-> bbox (puoi aumentare)
            need_w = label_w + 2 * pad
            need_h = label_h + 2 * pad

            # --- available space around bbox ---
            space_left  = x1c
            space_right = (w - 1) - x2c
            space_up    = y1c
            space_down  = (h - 1) - y2c

            # --- choose horizontal side ---
            if space_right >= need_w + margin:
                lx = x2c + margin
                ha = "left"     # label grows to the right
            elif space_left >= need_w + margin:
                lx = x1c - margin
                ha = "right"    # label grows to the left
            else:
                # fallback: keep inside image as much as possible
                # place it near x1c but clamp to fit
                ha = "left"
                lx = min(max(PIXEL_MIN, x1c), (w - 1) - need_w)

            # --- choose vertical side ---
            # (imshow default is origin='upper': smaller y is "up")
            if space_up >= need_h + margin:
                ly = y1c - margin
                va = "bottom"   # text extends upward from ly
            elif space_down >= need_h + margin:
                ly = y2c + margin
                va = "top"      # text extends downward from ly
            else:
                # fallback: keep inside image
                va = "top"
                ly = min(max(PIXEL_MIN, y1c), (h - 1) - need_h)

            # --- draw text at chosen anchor ---
            t = ax.text(
                lx,
                ly,
                text,
                fontsize=7,
                va=va,
                ha=ha,
                color=text_color,
                zorder=7,
            )

            # --- draw background rectangle tightly around rendered text (+pad) ---
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            bbox_disp = t.get_window_extent(renderer=rend)
            inv = ax.transData.inverted()
            (dx0, dy0) = inv.transform((bbox_disp.x0, bbox_disp.y0))
            (dx1, dy1) = inv.transform((bbox_disp.x1, bbox_disp.y1))

            x0 = min(dx0, dx1) - pad
            x1 = max(dx0, dx1) + pad
            y0 = min(dy0, dy1) - pad
            y1 = max(dy0, dy1) + pad

            # clamp background inside image
            x0 = max(PIXEL_MIN, x0)
            y0 = max(PIXEL_MIN, y0)
            x1 = min(float(w - 1), x1)
            y1 = min(float(h - 1), y1)

            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    max(BOX_MIN_SIZE, x1 - x0),
                    max(BOX_MIN_SIZE, y1 - y0),
                    fill=True,
                    linewidth=0,
                    facecolor=label_bg_color,
                    alpha=ALPHA_DEFAULT,
                    zorder=6,
                )
            )


    plt.show()
    print_dataframe(df[["id", "kind", "label", "source", "a11y_role", "states", "score", "vision_score", "fusion_score", "center_coords"]], limit=200)
    return df


@dataclass(frozen=True)
class UIVisualizationResult:
    fig: plt.Figure
    df: pd.DataFrame


def visualize_ui_elements2(
    ui_dict: dict,
    screenshot: np.ndarray,
    title: str = "UI Elements (a11y)",
    title_fontsize: int = TITLE_FONTSIZE,
    show_label: bool = True,
    include_label_text: bool = False,
    max_label_chars: int = MAX_LABEL_CHARS,
    figsize=None,
    box_color: str = "#00FF66",
    label_bg_color: str = "#00FF66",
    text_color: str = "#00FF66",
    box_linewidth: float = BOX_LINEWIDTH,
    # --- NEW: table rendering options ---
    table_max_rows: Optional[int] = 30,          # None => tutte le righe (attenzione a df grandi)
    table_columns: Optional[Sequence[str]] = None,  # None => tutte le colonne
    table_fontsize: int = 7,
    table_row_height: float = 1.2,
    table_header_color: str = "#F0F0F0",
    show: bool = True,
) -> UIVisualizationResult:
    """
    Visualize ui elements2.
        
        Parameters
        ----------
        ui_dict : dict
            Function argument.
        screenshot : np.ndarray
            Function argument.
        title : Optional[str]
            Function argument.
        title_fontsize : Optional[int]
            Function argument.
        show_label : Optional[bool]
            Function argument.
        include_label_text : Optional[bool]
            Function argument.
        max_label_chars : Optional[int]
            Function argument.
        figsize : Optional[Any]
            Function argument.
        box_color : Optional[str]
            Function argument.
        label_bg_color : Optional[str]
            Function argument.
        text_color : Optional[str]
            Function argument.
        box_linewidth : Optional[float]
            Function argument.
        table_max_rows : Optional[int]
            Function argument.
        table_columns : Optional[Sequence[str]]
            Function argument.
        table_fontsize : Optional[int]
            Function argument.
        table_row_height : Optional[float]
            Function argument.
        table_header_color : Optional[str]
            Function argument.
        show : Optional[bool]
            Function argument.
        
        Returns
        -------
        UIVisualizationResult
            Function result.
        
    """
    if screenshot is None:
        raise ValueError("screenshot is None")
    if not isinstance(screenshot, np.ndarray):
        raise TypeError("screenshot must be a numpy ndarray")
    if ui_dict is None or len(ui_dict) == 0:
        raise ValueError("ui_dict is empty")

    img = screenshot
    if img.ndim == 2:
        pass
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        pass
    else:
        raise ValueError(f"Unexpected screenshot shape: {img.shape}")

    h, w = img.shape[0], img.shape[1]
    if figsize is None:
        figsize = (max(FIGSIZE_MIN_W, w / FIGSIZE_DENOM), max(FIGSIZE_MIN_H, h / FIGSIZE_DENOM))

    df = pd.DataFrame([el.model_dump() for el in ui_dict.values()])
    if df.empty:
        raise ValueError("No elements had bb_coords; dataframe is empty")

    # ---------- Build a single exportable figure: image (top) + table (bottom)
    # Height ratios: more room to image, some to table. Tune as you like.
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[3.5, 1.5], figure=fig)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0])

    # --- IMAGE AXIS ---
    ax_img.imshow(img)
    ax_img.axis("off")
    if title:
        ax_img.set_title(title, fontsize=title_fontsize)

    # Draw bboxes + labels
    for _, r in df.iterrows():
        bb = r["bb_coords"]
        x1, y1, x2, y2 = float(bb["x_1"]), float(bb["y_1"]), float(bb["x_2"]), float(bb["y_2"])

        x1c = max(PIXEL_MIN, min(w - PIXEL_MAX_OFFSET, x1))
        y1c = max(PIXEL_MIN, min(h - PIXEL_MAX_OFFSET, y1))
        x2c = max(PIXEL_MIN, min(w - PIXEL_MAX_OFFSET, x2))
        y2c = max(PIXEL_MIN, min(h - PIXEL_MAX_OFFSET, y2))

        if x2c < x1c:
            x1c, x2c = x2c, x1c
        if y2c < y1c:
            y1c, y2c = y2c, y1c

        ax_img.add_patch(
            Rectangle(
                (x1c, y1c),
                x2c - x1c,
                y2c - y1c,
                fill=False,
                linewidth=box_linewidth,
                edgecolor=box_color,
                zorder=4,
            )
        )

        if show_label:
            text = str(r["id"])
            if include_label_text:
                lbl = str(r.get("label") or "").replace("\n", " ").strip()
                if len(lbl) > max_label_chars:
                    lbl = lbl[: max_label_chars - 1] + "…"
                if lbl:
                    text = f"{text} {lbl}"

            t = ax_img.text(
                x1c,
                y1c,
                text,
                fontsize=7,
                va="top",
                ha="left",
                color=text_color,
                zorder=7,
            )

            fig.canvas.draw()
            bbox_disp = t.get_window_extent(renderer=fig.canvas.get_renderer())
            inv = ax_img.transData.inverted()
            (bx0, by0) = inv.transform((bbox_disp.x0, bbox_disp.y0))
            (bx1, by1) = inv.transform((bbox_disp.x1, bbox_disp.y1))

            pad = PAD_DEFAULT
            lx0 = x1c
            ly0 = y1c
            lx1 = min(float(w - 1), bx1 + pad)
            ly1 = min(float(h - 1), by1 + pad)

            ax_img.add_patch(
                Rectangle(
                    (lx0, ly0),
                    max(BOX_MIN_SIZE, lx1 - lx0),
                    max(BOX_MIN_SIZE, ly1 - ly0),
                    fill=True,
                    linewidth=0,
                    facecolor=label_bg_color,
                    alpha=ALPHA_DEFAULT,
                    zorder=6,
                )
            )

            t.set_position((lx0 + TEXT_OFFSET_X, ly0 + (ly1 - ly0) - TEXT_OFFSET_Y))
            t.set_zorder(7)

    # --- TABLE AXIS ---
    ax_tbl.axis("off")

    df_tbl = df
    if table_columns is not None:
        missing = [c for c in table_columns if c not in df_tbl.columns]
        if missing:
            raise ValueError(f"table_columns contains missing columns: {missing}")
        df_tbl = df_tbl.loc[:, list(table_columns)]

    if table_max_rows is not None and len(df_tbl) > table_max_rows:
        df_tbl = df_tbl.head(table_max_rows)

    # Convert to strings for safe rendering (dicts, lists, etc.)
    cell_text = df_tbl.astype(str).values.tolist()
    col_labels = [str(c) for c in df_tbl.columns.tolist()]

    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(table_fontsize)
    tbl.scale(1.0, table_row_height)

    # Style header row
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(table_header_color)
            cell.set_text_props(weight="bold")

    fig.tight_layout()

    if show:
        plt.show()

    return UIVisualizationResult(fig=fig, df=df)


def show_img(img, title=None):
    """
    Show img.
        
        Parameters
        ----------
        img : Any
            Function argument.
        title : Optional[Any]
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
    """
    plt.figure(figsize=SHOW_IMG_FIGSIZE)
    plt.imshow(img)  # shape (1080, 1920, 3), range 0-255
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

def show_transition(before_img, after_img, title=None, figsize=TRANSITION_FIGSIZE):
    """
    Show transition.
        
        Parameters
        ----------
        before_img : Any
            Function argument.
        after_img : Any
            Function argument.
        title : Optional[Any]
            Function argument.
        figsize : Optional[Any]
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    axes[0].imshow(before_img)
    axes[0].axis("off")
    axes[0].set_title("Before")

    axes[1].imshow(after_img)
    axes[1].axis("off")
    axes[1].set_title("After")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def print_history(
    history,
    max_ui=8,
    max_str=TRUNC_MAX_STR,
    show_steps=True,          # <--- abilita/disabilita show_transition
    transition_figsize=TRANSITION_FIGSIZE,
    transition_title=True,    # <--- se True, mette un titolo utile
):
    """
    Compact pretty-printer for a `history` that alternates Observation (dict) and Step-like objects.
    - Prints "Observation" instead of "dict".
    - Never dumps raw screenshot arrays; prints only shape/dtype.
    - Skips the `obs` field entirely (common in Step objects and often very large/redundant).
    - If show_steps=True, calls show_transition with screenshots before/after each step.
    """

    def trunc(x, n=max_str):
        """
        Truncate trunc.
        
        Parameters
        ----------
        x : Any
            Function argument.
        n : Optional[Any]
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
        """
        s = str(x).replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"

    def is_array(x):
        """
        Return whether is array.
        
        Parameters
        ----------
        x : Any
            Function argument.
        
        Returns
        -------
        Any
            True when the condition is satisfied, otherwise False.
        """
        return hasattr(x, "shape") and hasattr(x, "dtype")

    def dump_step(x):
        """
        Dump step.
        
        Parameters
        ----------
        x : Any
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
        """
        if hasattr(x, "model_dump"):
            try:
                return x.model_dump()
            except Exception:
                return None
        if hasattr(x, "dict"):
            try:
                return x.dict()
            except Exception:
                return None
        return None

    def get_obs_screenshot(obs):
        """
        Ritorna lo screenshot dall'Observation se presente, altrimenti None.
        """
        if isinstance(obs, dict) and "screenshot" in obs:
            return obs["screenshot"]
        return None

    def find_prev_screenshot(idx):
        """
        Cerca all'indietro la più recente Observation con screenshot.
        """
        for j in range(idx - 1, -1, -1):
            if isinstance(history[j], dict) and "screenshot" in history[j]:
                return history[j]["screenshot"]
        return None

    def find_next_screenshot(idx):
        """
        Cerca in avanti la prima Observation con screenshot.
        """
        for j in range(idx + 1, len(history)):
            if isinstance(history[j], dict) and "screenshot" in history[j]:
                return history[j]["screenshot"]
        return None

    def step_title(i, step_obj):
        """
        Process step title.
        
        Parameters
        ----------
        i : Any
            Function argument.
        step_obj : Any
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
        """
        if not transition_title:
            return None
        # titolo “informativo” ma non troppo verboso
        data = dump_step(step_obj)
        action = None
        if isinstance(data, dict):
            # prova a pescare un campo ragionevole se esiste
            for k in ("action", "tool", "name", "type"):
                if k in data:
                    action = data[k]
                    break
        base = f"[{i:03d}] {type(step_obj).__name__}"
        return f"{base}" + (f" | {action}" if action is not None else "")

    def print_observation(obs: dict):
        # Only print useful, human-readable summaries (no raw pixel dumps)
        """
        Print observation.
        
        Parameters
        ----------
        obs : dict
            Function argument.
        
        Returns
        -------
        Any
            Function result.
        
        """
        if "screenshot" in obs:
            sc = obs["screenshot"]
            if is_array(sc):
                print(f"screenshot: shape={sc.shape}, dtype={sc.dtype}")
            else:
                print(f"screenshot: <{type(sc).__name__}>")

        if "ui_elements" in obs and isinstance(obs["ui_elements"], dict):
            ui = obs["ui_elements"]
            print(f"ui_elements: {len(ui)}")
            for k, el in list(ui.items())[:max_ui]:
                kind = getattr(el, "kind", None)
                label = getattr(el, "label", None)
                role = getattr(el, "a11y_role", None)
                vis = getattr(el, "visible", None)
                en = getattr(el, "enabled", None)
                bb = getattr(el, "bb_coords", None)

                bb_s = None
                if bb is not None and all(hasattr(bb, a) for a in ("x_1", "y_1", "x_2", "y_2")):
                    bb_s = f"bb=({bb.x_1:.1f},{bb.y_1:.1f})->({bb.x_2:.1f},{bb.y_2:.1f})"

                parts = [
                    f"{k}",
                    f"kind={kind}" if kind is not None else None,
                    f"label={trunc(label, LABEL_TRUNC_LEN)}" if label is not None else None,
                    f"role={role}" if role is not None else None,
                    f"visible={vis}" if vis is not None else None,
                    f"enabled={en}" if en is not None else None,
                    bb_s,
                ]
                print("  - " + ", ".join(p for p in parts if p))

            if len(ui) > max_ui:
                print(f"  … (+{len(ui) - max_ui} more)")

        # Print any other small fields (but skip noisy/heavy keys)
        for k, v in obs.items():
            if k in ("screenshot", "ui_elements", "obs"):
                continue
            if is_array(v):
                print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, (dict, list, tuple)) and len(v) > LARGE_CONTAINER_LEN:
                print(f"{k}: <{type(v).__name__} len={len(v)}>")
            else:
                print(f"{k}: {trunc(v)}")

    for i, item in enumerate(history):
        print("\n" + "=" * PRINT_DIVIDER_LEN)

        # Observation
        if isinstance(item, dict):
            print(f"[{i:03d}] Observation")
            print_observation(item)
            continue

        # Step-like
        print(f"[{i:03d}] {type(item).__name__}")

        # ---- INTEGRAZIONE show_transition: prima/dopo ciascuno step ----
        if show_steps:
            before_sc = find_prev_screenshot(i)
            after_sc = find_next_screenshot(i)

            # chiama show_transition solo se ho entrambi e sembrano array/immagini
            if before_sc is not None and after_sc is not None:
                try:
                    show_transition(
                        before_sc,
                        after_sc,
                        title=step_title(i, item),
                        figsize=transition_figsize
                    )
                except Exception as e:
                    # non bloccare il print se plotting fallisce
                    print(f"(show_transition skipped: {type(e).__name__}: {e})")
            else:
                # se vuoi silenziare questo, puoi rimuovere la riga sotto
                print("(show_transition skipped: missing before/after screenshot)")

        # dump step content
        data = dump_step(item)
        if isinstance(data, dict):
            for k, v in data.items():
                if k == "obs":
                    continue
                if k == "screenshot" and is_array(v):
                    print(f"{k}: shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, (dict, list, tuple)) and len(v) > LARGE_CONTAINER_LEN:
                    print(f"{k}: <{type(v).__name__} len={len(v)}>")
                else:
                    print(f"{k}: {trunc(v)}")
        else:
            print(trunc(repr(item), ITEM_TRUNC_LEN))




def _edge_type(e) -> str:
    # robust across different field names
    """
    Process edge type.
        
        Parameters
        ----------
        e : Any
            Function argument.
        
        Returns
        -------
        str
            Resulting string value.
        
    """
    return getattr(e, "relation", None) or getattr(e, "edge_type", None) or getattr(e, "type", None) or "UNKNOWN"


def banner(title: str) -> None:
    """
    Render banner.
        
        Parameters
        ----------
        title : str
            Function argument.
        
        Returns
        -------
        None
            No return value.
        
    """
    print("\n" + "=" * PRINT_DIVIDER_LEN_LARGE)
    print(title)
    print("=" * PRINT_DIVIDER_LEN_LARGE)

def show_trim_output(out: TRIMToolOutput) -> None:
    """
    Show trim output.
        
        Parameters
        ----------
        out : TRIMToolOutput
            Function argument.
        
        Returns
        -------
        None
            No return value.
        
    """
    print("\n[TRIM OUTPUT]")
    if out.global_notes:
        print(f"  global_notes: {out.global_notes}")
    for i, d in enumerate(out.decisions):
        print(f"  - decision[{i}]: op={d.op} intent={d.intent}")
        print(f"      subtask: {d.subtask}")
        if d.target_node_id:
            print(f"      target_node_id: {d.target_node_id}")
        if d.proposed_title:
            print(f"      proposed_title: {d.proposed_title}")
        if d.proposed_value:
            print(f"      proposed_value: {d.proposed_value}")
        if d.depends_on:
            print(f"      depends_on: {d.depends_on}")
        if d.rollback_to_rev is not None:
            print(f"      rollback_to_rev: {d.rollback_to_rev}")
        if d.rationale:
            print(f"      rationale: {d.rationale}")

def show_graph(tms: OnlineTMS) -> None:
    """
    Run show graph for the current workflow step.
    
    Parameters
    ----------
    tms : OnlineTMS
        Online task-memory graph used for contextual retrieval.
    
    Returns
    -------
    None
        No return value.
    """
    print("\n[TMS GRAPH SNAPSHOT]")
    nodes = tms.nodes()
    edges = tms.edges()
    print(f"  step: {tms.step}")
    print(f"  nodes: {len(nodes)} | edges: {len(edges)}")
    for n in nodes:
        val = (n.value or "").replace("\n", " ")
        val = val[:SUMMARY_MAX_LEN] + ("..." if len(val) > SUMMARY_MAX_LEN else "")
        print(f"  - node_id={n.node_id} | status={n.status} | title={n.title!r}")
        print(f"      revisions={len(n.revisions)} last_rev={n.revisions[-1].rev_id if n.revisions else None}")
        print(f"      anchors={len(n.anchors)} value='{val}'")
    for e in edges:
        print(f" - edge: {e.parent_id} -> {e.child_id} ({_edge_type(e)})")
