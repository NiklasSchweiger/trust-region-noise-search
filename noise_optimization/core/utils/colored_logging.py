"""
Colored logging utility using Rich for beautiful terminal output.

Provides styled print functions that automatically color-code different log levels
and component prefixes for better readability.
"""

from __future__ import annotations

from typing import Optional
import builtins

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.text import Text
    RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    RICH_AVAILABLE = False
    _console = None

# Paper color palette: blue, orange, green, red (matches LaTeX and CLI)
try:
    from .terminal_colors import PAPER_BLUE, PAPER_ORANGE, PAPER_GREEN, PAPER_RED
except ImportError:
    PAPER_BLUE = "blue"
    PAPER_ORANGE = "yellow"
    PAPER_GREEN = "green"
    PAPER_RED = "red"

LEVEL_COLORS = {
    "INFO": PAPER_BLUE,
    "DEBUG": "dim white",
    "WARNING": PAPER_ORANGE,
    "ERROR": PAPER_RED,
    "SUCCESS": PAPER_GREEN,
}

COMPONENT_COLORS = {
    "T2I": PAPER_BLUE,
    "CoSyNE": PAPER_BLUE,
    "TRS": PAPER_BLUE,
    "Cosyne": PAPER_BLUE,
    "REWARD": PAPER_BLUE,
    "MODEL": PAPER_BLUE,
    "BENCHMARK": PAPER_BLUE,
    "SOLVER": PAPER_BLUE,
    "PIPELINE": PAPER_BLUE,
    "DrawBench": PAPER_BLUE,
    "DrawBenchBenchmark": PAPER_BLUE,
    "ScalingLogger": f"dim {PAPER_BLUE}",
    "REWARD_FACTORY": PAPER_BLUE,
}


def _get_component_color(prefix: str) -> str:
    """Get color for a component prefix."""
    # Try exact match first
    if prefix in COMPONENT_COLORS:
        return COMPONENT_COLORS[prefix]
    # Try case-insensitive match
    prefix_lower = prefix.lower()
    for key, color in COMPONENT_COLORS.items():
        if key.lower() == prefix_lower:
            return color
    # Check for partial matches (e.g., "Cosyne.__init__" should match "Cosyne")
    for key, color in COMPONENT_COLORS.items():
        if key.lower() in prefix_lower or prefix_lower in key.lower():
            return color
    return PAPER_BLUE


def colored_print(message: str, level: Optional[str] = None, component: Optional[str] = None) -> None:
    """Print a message with automatic color styling.
    
    Args:
        message: The message to print
        level: Optional log level (INFO, DEBUG, WARNING, ERROR, SUCCESS)
        component: Optional component prefix (T2I, CoSyNE, TRS, etc.)
    """
    if not RICH_AVAILABLE or _console is None:
        builtins.print(message)
        return
    
    # Try to parse the message format: [PREFIX] message
    msg_stripped = message.strip()
    if msg_stripped.startswith("["):
        end_bracket = msg_stripped.find("]")
        if end_bracket > 0:
            prefix = msg_stripped[1:end_bracket]
            rest = msg_stripped[end_bracket + 1:].strip()
            
            parsed_level = level
            parsed_component = component
            
            # Check if prefix is a log level
            if prefix in LEVEL_COLORS:
                parsed_level = prefix
                # Check if there's a component after the level
                if rest.startswith("["):
                    comp_end = rest.find("]")
                    if comp_end > 0:
                        parsed_component = rest[1:comp_end]
                        rest = rest[comp_end + 1:].strip()
            else:
                # Treat as component prefix - try to extract component name
                # Handle cases like "Cosyne.__init__", "T2I Benchmark Summary", etc.
                component_candidate = prefix.split(".")[0].split()[0]  # Take first part before dot or space
                # Check if it matches a known component
                component_match = None
                for comp_key in COMPONENT_COLORS.keys():
                    if comp_key.lower() in component_candidate.lower() or component_candidate.lower() in comp_key.lower():
                        component_match = comp_key
                        break
                parsed_component = component_match if component_match else component_candidate
            
            # Create styled output
            text = Text()
            if parsed_level:
                text.append("[", style="dim")
                text.append(parsed_level, style=LEVEL_COLORS.get(parsed_level, "white"))
                text.append("]", style="dim")
                if parsed_component or rest:
                    text.append(" ", style="dim")
            
            if parsed_component:
                text.append("[", style="dim")
                text.append(parsed_component, style=_get_component_color(parsed_component))
                text.append("]", style="dim")
                if rest:
                    text.append(" ", style="dim")
            
            text.append(rest, style="white")
            _console.print(text)
            return
    
    # Fallback: print as-is with Rich (or fallback to builtin if Rich fails)
    try:
        _console.print(message)
    except Exception:
        builtins.print(message)


def info(message: str, component: Optional[str] = None) -> None:
    """Print an INFO level message."""
    if component:
        colored_print(f"[INFO] [{component}] {message}")
    else:
        colored_print(f"[INFO] {message}")


def debug(message: str, component: Optional[str] = None) -> None:
    """Print a DEBUG level message."""
    if component:
        colored_print(f"[DEBUG] [{component}] {message}")
    else:
        colored_print(f"[DEBUG] {message}")


def warning(message: str, component: Optional[str] = None) -> None:
    """Print a WARNING level message."""
    if component:
        colored_print(f"[WARNING] [{component}] {message}")
    else:
        colored_print(f"[WARNING] {message}")


def error(message: str, component: Optional[str] = None) -> None:
    """Print an ERROR level message."""
    if component:
        colored_print(f"[ERROR] [{component}] {message}")
    else:
        colored_print(f"[ERROR] {message}")


def success(message: str, component: Optional[str] = None) -> None:
    """Print a SUCCESS level message."""
    if component:
        colored_print(f"[SUCCESS] [{component}] {message}")
    else:
        colored_print(f"[SUCCESS] {message}")


# Global print override functionality
_original_print = builtins.print
_use_colored = False


def enable_colored_printing(enable: bool = True) -> None:
    """Enable or disable colored printing globally by overriding built-in print().
    
    WARNING: This modifies the global print function. Use with caution.
    """
    global _use_colored
    _use_colored = enable
    
    if enable and RICH_AVAILABLE:
        def colored_print_wrapper(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False):
            """Wrapper that intercepts print calls and applies coloring."""
            # Don't intercept if printing to a file
            if file is not None:
                _original_print(*args, sep=sep, end=end, file=file, flush=flush)
                return
            
            if args:
                message = sep.join(str(arg) for arg in args)
                # Only apply coloring if message matches our patterns (starts with [)
                if message.strip().startswith("["):
                    try:
                        colored_print(message)
                        # Handle custom end characters (colored_print always ends with newline)
                        if end and end != "\n":
                            _original_print(end, end="", file=file, flush=flush)
                        elif flush and _console:
                            try:
                                _console.file.flush()
                            except Exception:
                                pass
                        return
                    except Exception as e:
                        # Fall back to original print if coloring fails
                        _original_print(*args, sep=sep, end=end, file=file, flush=flush)
                        return
            # Fall back to original print for other cases
            _original_print(*args, sep=sep, end=end, file=file, flush=flush)
        
        builtins.print = colored_print_wrapper
    else:
        builtins.print = _original_print


def disable_colored_printing() -> None:
    """Disable colored printing and restore original print()."""
    enable_colored_printing(False)
