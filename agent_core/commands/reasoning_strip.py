"""Strip LLM reasoning tokens and leaked chain-of-thought from workflow output."""
import re

THINK_OPEN = chr(60) + 'think' + chr(62)
THINK_CLOSE = chr(60) + '/think' + chr(62)
TOOL_OPEN = chr(60) + 'tool_call' + chr(62)
TOOL_CLOSE = chr(60) + '/tool_call' + chr(62)
CHECKMARK = chr(9989)

_REASONING_LINE_PREFIXES = (
    "Ah,", "Wait,", "Let's", "Let me",
    "I see", "I will", "I'll", "I've", "I am",
    "Now,", "Hmm", "Actually", "Okay",
    "Draft", "All good", "All constraints", "Ready.",
    "Proceeds.", "Output matches", "Self-Correction",
    "[Output", "[Self", "Output Generation",
    "Check constraints", "I need to", "I should",
    "First,", "Second,", "Third,",
    "I'll just", "I'll provide", "I'll output",
    "One minor", "One detail",
    "Draft looks", "Structure:", "End with exactly",
    "one of the lines", "one of these formats",
    "Check bullet", "No intro text", "Code block only",
    "Constraint:", "Constraints:", "Check:",
    "I'll verify", "I'll ensure", "I'll keep",
    "I'll stick", "I'll put", "I'll add",
    "I will strictly", "I will structure",
    "I will ensure", "I will just",
    "My task:", "Critique points", "This looks like",
    "Let me verify line", "Let me count",
    "Let me look", "Let me trace",
    "I will craft", "I will produce",
    "Ready to produce", "Ready,",
    "Focus on:", "I will list",
)

# Build the think/tool_call tag regex at runtime to avoid source-level stripping
_re_think = re.compile(
    re.escape(THINK_OPEN) + r'.*?' + re.escape(THINK_CLOSE)
    + r'|' + re.escape(THINK_OPEN) + r'.*$'
    + r'|' + re.escape(TOOL_OPEN) + r'.*?' + re.escape(TOOL_CLOSE)
    + r'|' + re.escape(TOOL_OPEN) + r'.*$'
    , re.DOTALL | re.MULTILINE
)
_re_arrow = re.compile(r'^\s*->\s*\*?Proceeds\*?\.?\s*$', re.MULTILINE | re.IGNORECASE)
_re_markers = re.compile(r'^\s*\[(?:Output|Self)[^\]]*\]\s*$', re.MULTILINE | re.IGNORECASE)
_re_multi_blank = re.compile(r'\n{3,}', re.DOTALL)


def _is_reasoning_line(stripped: str) -> bool:
    """Check if a stripped line is reasoning leak."""
    check = re.sub(r'^\d+\.\s+', '', stripped)
    check = check.lstrip('*').strip()
    check_lower = check.lower()
    for prefix in _REASONING_LINE_PREFIXES:
        if check_lower.startswith(prefix.lower()):
            return True
    if CHECKMARK in check:
        return True
    if 'Output matches response' in check:
        return True
    if '-> *Proceeds*' in check:
        return True
    if check.endswith('Proceeds.') and len(check) < 30:
        return True
    if check.endswith('Yes.') and len(check) < 40 and '?' in check:
        return True
    if check == 'Checked.':
        return True
    if check.startswith('End with exactly'):
        return True
    if check_lower.startswith('let me'):
        return True
    if check_lower.startswith('this looks like'):
        return True
    if check_lower.startswith('my task:'):
        return True
    if check_lower.startswith('critique points'):
        return True
    if check.lower().startswith('i will craft'):
        return True
    if check.lower().startswith('focus on:'):
        return True
    if re.match(r'^\d+\.\s+`', stripped):
        return True
    if stripped.startswith('## Analysis to critique:'):
        return True
    if stripped.startswith('Wait, let'):
        return True
    if stripped.startswith('Ah, I see'):
        return True
    if stripped.startswith('And then'):
        return True
    if stripped.startswith('```'):
        return True
    if stripped == '...':
        return True
    if stripped.startswith('## Analysis to critique:'):
        return True
    if re.match(r'^\d+\.\s+(Sandbox|LLM|Missing|Success|Prompt)', stripped):
        return True
    if stripped.lower().startswith('draft points to refine:'):
        return True
    if stripped.lower().startswith('output matches response'):
        return True
    if stripped.startswith(chr(96)):  # ASCII backtick
        return True
    if 'wait, no.' in check_lower or 'wait, no' in check_lower:
        return True
    if 'let me re-read' in check_lower:
        return True
    if check_lower.startswith('and then immediately'):
        return True
    if check_lower.startswith('this looks like'):
        return True
    if re.match(r'^\d+\.\s+(Shell|Path|The sanitizer)', stripped):
        return True
    if check_lower == 'the prompt says':
        return True
    if check_lower.startswith('looks like a draft'):
        return True
    if check_lower.startswith('the prompt says') or check_lower.startswith('the prompt actually'):
        return True
    if check_lower.startswith('the provided analysis says'):
        return True
    if check_lower.startswith('check tone:'):
        return True
    if check_lower.startswith('format:'):
        return True
    if check_lower.startswith('- the provided'):
        return True
    if re.match(r'^\d+\.\s+(Shell|Path|The sanitizer|Context|Async)', stripped):
        return True
    if stripped.startswith('- "No explicit') or stripped.startswith('- "Sandbox'):
        return True
    if '-> True, but misses' in check:
        return True
    if check_lower.startswith('i will list') or check_lower.startswith('i will produce') or check_lower.startswith('i will craft'):
        return True
    if check_lower.startswith('ready.') or check_lower.startswith('all set.') or check_lower.startswith('ready,'):
        return True
    if check_lower.startswith('also,'):
        return True
    if check_lower.startswith('this string') or check_lower.startswith('this is'):
        return True
    if check.startswith('- **)Verdict**)') or check.startswith('- **)Verdict Shallow'):
        return True
    if re.match(r'^- \*\*(\w+(?:\s+\w+){0,5})\*\*\s*$', stripped):
        # Lines that are only a short bold label like "- **Verdict**" with no content
        return True
    if re.match(r'^- \*\*(?:Missed|Overlooked|Missing|Verdict)\b', stripped):
        # Standalone section labels from response structure plan
        return True
    if re.match(r'^- Missing metrics:', stripped):
        return True
    if check_lower.startswith('does enforce boundary') or check_lower.startswith("don't call it"):
        return True
    if re.match(r'^\s+(timeout_match|if timeout_match|run_args)\b', stripped):
        # Embedded Python code snippet from reasoning process
        return True
    if re.match(r'^\s{8,}(if |return |def |path |filename |import )', stripped):
        # Indented code snippet from reasoning (8+ spaces of code indent)
        return True
    # Check code-snippet patterns against the stripped content (no leading spaces)
    code_patterns = [
        r'^(timeout_match|timeout|run_args|cmd_to_run)\s*=',
        r'^if\s+\w+:',
        r'^if\s+path\s*\.startswith',
        r'^filename\s*=\s*query',
        r'^return\s+self\.',
        r'^return\s+str\(normalize',
        r'^path\s*=\s*path\[2:\]',
        r'^return\s+\(',
    ]
    for pat in code_patterns:
        if re.match(pat, check):
            return True
    if stripped.startswith('def safe_path') or stripped.startswith('def _safe_path') or stripped.startswith('def _normalize_path'):
        return True
    if stripped.startswith('from agent_core') and 'import' in stripped and len(stripped) < 120:
        return True
    if check_lower.startswith('so `') and 'uses' in check_lower:
        return True
    if check_lower.startswith('however, `') and 'do not' in check_lower:
        return True
    if check == 'So `agent.py` actually uses the secure one.':
        return True
    if re.match(r'^path\s*=\s*path\[', stripped):
        return True
    return False


def _clean_analysis_sections(text: str) -> str:
    """Extract only the valid analysis sections, removing reasoning drafts.

    The LLM often writes a partial draft with reasoning before the clean version.
    We keep only content from the first '## 1.' header onwards, plus the
    header lines and refinement/verification sections.
    """
    lines = text.split('\n')

    # Find header (first lines that start with '# ' (not '##') or '|')
    header_lines = []
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    # Only keep top-level title '# ' and meta '| ' lines before the sections
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith('# ') or s.startswith('|'):
            header_lines.append(lines[i])
            i += 1
        else:
            break

    # Find first '## 1.' section header at column 0 (no leading whitespace)
    first_section = -1
    for j, line in enumerate(lines):
        if re.match(r'^##\s+1\.', line):  # no strip() - must start at col 0
            first_section = j
            break

    if first_section == -1:
        # No section headers found, return text with header + line-by-line cleaned
        return text

    # Find end of clean sections (before '## Refinement' or '## Verification')
    refinement_start = -1
    verification_start = -1
    for j in range(first_section, len(lines)):
        stripped = lines[j].strip()
        if stripped.startswith('## Refinement'):
            refinement_start = j
            break
        if stripped.startswith('## Verification'):
            verification_start = j
            break

    # Build clean sections: header + sections 1-8 + BLOCKED
    section_lines = header_lines
    if first_section >= 0:
        end = refinement_start if refinement_start >= 0 else (verification_start if verification_start >= 0 else len(lines))
        section_lines.append('')
        section_lines.extend(lines[first_section:end])

    # Add refinement section if it exists, cleaned
    if refinement_start >= 0:
        refinement_end = verification_start if verification_start >= 0 else len(lines)
        section_lines.append('')
        for rl in lines[refinement_start:refinement_end]:
            if not _is_reasoning_line(rl.strip()):
                section_lines.append(rl)

    # Add verification section if it exists
    if verification_start >= 0:
        section_lines.append('')
        section_lines.extend(lines[verification_start:])

    return '\n'.join(section_lines)


def strip_reasoning(text: str) -> str:
    """Remove LLM reasoning tokens and leaked chain-of-thought from text."""
    if not text:
        return text

    out = _re_think.sub('', text)
    out = _re_arrow.sub('', out)
    out = _re_markers.sub('', out)

    # Try section-based extraction for analysis files
    if '## 1.' in out and ('## Refinement' in out or '## Verification' in out):
        out = _clean_analysis_sections(out)
    else:
        # Line-by-line reasoning removal
        lines_out = []
        for line in out.split('\n'):
            stripped = line.strip()
            if not _is_reasoning_line(stripped):
                lines_out.append(line)
        out = '\n'.join(lines_out)

    out = _re_multi_blank.sub('\n\n', out)
    return out.strip() + '\n'
