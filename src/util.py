from datetime import datetime
import os

from metrics import PRETTY_METRIC_NAMES


def get_time():
    """
    Pretty print current time for labeling training runs
    """
    return datetime.now().strftime("%m-%d-%H-%M-%S")


def extract_prefixes(strings):
    """
    Get the unique prefixes of a list of strings

    ["apple_suffix", "banana_suffix", "cherry_suffix"] => ['apple', 'banana', 'cherry']
    """
    common_suffix = ""
    str = [s[::-1] for s in strings]
    for i in range(len(min(strings, key=len))):
        if len(set([s[i] for s in str])) == 1:
            common_suffix += str[0][i]
        else:
            break
    common_suffix = common_suffix[::-1]
    return [s[:-len(common_suffix)] for s in strings], common_suffix


def create_results_dir(parent_dir, run_label):
    """
    Create results directory for subset evaluation
    """
    dir_name = os.path.join(parent_dir, run_label)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def render_table(results, minimal=False):
    """
    Converts a nested dict to a LaTeX table.

    Input:
        {
            "Column 1": {
                "Row 1": ...,
                "Row 2": ...,
                ...
            },
            "Column 2": {
                "Row 1": ...,
                "Row 2": ...,
                ...
            },
            ...
        }
    
    Output:
        | Column 1 | Column 2 | ...      |
        |----------|----------|----------|
        | Row 1    | Row 1    | ...      |
        | Row 2    | Row 2    | ...      |
        | ...      | ...      | ...      |

    """
    if not (isinstance(results, dict) and all(isinstance(value, dict) and set(value.keys()) == set(next(iter(results.values())).keys()) for value in results.values())):
        raise ValueError("Input must be a nested dict with matching inner keys")

    headers = list(results[list(results.keys())[0]].keys())
    headers = [h for h in headers if not isinstance(results[list(results.keys())[0]][h], list)]

    header_spec = ''.join(['C{1.5cm}' for _ in headers])
    header_text = [PRETTY_METRIC_NAMES.get(h, h) for h in headers]
    header_text = ' & '.join([f'\\textbf{{{key}}}' for key in header_text])

    result_text = '\n'
    for entry_key, entry_item in results.items():
        entry_text = f'{entry_key} & '
        entry_text += ' & '.join([str(round(entry_item[h], 2)) if isinstance(entry_item[h], float) else str(entry_item[h]) for h in headers])
        entry_text += ' \\\\\n'
        result_text += entry_text

    if not minimal:
        table = f"""\\begin{{tabular}}{{p{{3cm}}{header_spec}}}
\\toprule
\\textbf{{Method}} & {header_text} \\\\
\\midrule
{result_text}
\\bottomrule
\\end{{tabular}}"""
    else:
        table = f"""\\textbf{{Method}} & {header_text} \\\\
\\midrule
{result_text}"""

    return table