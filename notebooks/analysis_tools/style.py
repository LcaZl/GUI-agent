from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


APTOS_FONT_DIR = Path(r"C:\Users\Luca\Downloads\Microsoft Aptos Fonts")
PREFERRED_FONT_FAMILY = "Aptos Display"

SUPTITLE_FONTSIZE = 30
PANEL_TITLE_FONTSIZE = 24
AXIS_LABEL_FONTSIZE = 22
XTICK_FONTSIZE = 20
YTICK_FONTSIZE = 20
LEGEND_FONTSIZE = 18
LEGEND_TITLE_FONTSIZE = 18
ANNOTATION_FONTSIZE = 18
HEATMAP_CELL_FONTSIZE = 16


def configure_matplotlib_font(font_dir=APTOS_FONT_DIR, preferred_family=PREFERRED_FONT_FAMILY):
    font_dir = Path(font_dir)
    available_names = []
    if font_dir.exists():
        for font_path in sorted(font_dir.glob("*.ttf")) + sorted(font_dir.glob("*.otf")):
            font_manager.fontManager.addfont(str(font_path))
            try:
                available_names.append(font_manager.FontProperties(fname=str(font_path)).get_name())
            except Exception:
                continue

    chosen = None
    if preferred_family in available_names:
        chosen = preferred_family
    else:
        for name in available_names:
            if preferred_family.lower() in name.lower():
                chosen = name
                break
    if chosen is None:
        for name in available_names:
            if "aptos" in name.lower():
                chosen = name
                break

    if chosen:
        plt.rcParams["font.family"] = chosen
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"
    return chosen
