"""
Terminal color palette matching the paper's LaTeX method colors.

Use these in Rich styles so CLI output is consistent with paper figures.
Primary palette: blue, orange, green, red (method1–4). Rich accepts rgb(r,g,b).
"""
# Paper colors (RGB) as Rich rgb(r,g,b) strings
# LaTeX: \definecolor{method1color}{RGB}{31, 119, 180} etc.
PAPER_BLUE = "rgb(31,119,180)"      # method1
PAPER_ORANGE = "rgb(255,127,14)"    # method2
PAPER_GREEN = "rgb(44,160,44)"      # method3
PAPER_RED = "rgb(214,39,40)"        # method4

# Semantic roles for CLI (only blue, orange, green, red)
CLI_APP = PAPER_BLUE
CLI_SECTION = PAPER_BLUE             # sections use blue like app
CLI_SUCCESS = PAPER_GREEN
CLI_WARNING = PAPER_ORANGE
CLI_ERROR = PAPER_RED
CLI_DIM = f"dim {PAPER_BLUE}"        # secondary text = dim blue
