### Used Sonnet 4.6 to help generate this script to convert the tables in the notebooks to the format we use
### Namely because reformatting tables took too much time 
### Manually reviewed all data to ensure it doesn't chang the actual values of the output


## To import from the code to the paper
## Copy the format_table output
## Rename columns for space
## Replace '\n' with an actual new line and ' '
## Replace '\\' with '\'

import re

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_tabular(latex: str):
    """Return (col_spec, rows) where rows is a list of lists of strings."""
    spec_match = re.search(r'\\begin\{tabular\}\{([^}]+)\}', latex)
    col_spec = spec_match.group(1) if spec_match else ""

    lines = latex.split('\n')
    rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if any(line.startswith(cmd) for cmd in [
            r'\begin', r'\end', r'\toprule', r'\bottomrule', r'\midrule'
        ]):
            continue
        if '\\\\' in line:
            line = line.rsplit('\\\\', 1)[0].strip()
            cells = [c.strip() for c in line.split('&')]
            rows.append(cells)
    return col_spec, rows


def build_col_spec(orig_spec: str) -> str:
    """Build {l|lll...} — vertical bar after the first column."""
    cols = re.sub(r'[^a-zA-Z]', '', orig_spec)
    ncols = len(cols)
    return 'l|' + 'l' * (ncols - 1)


def is_pvalue_row(first_cell: str) -> bool:
    return (
        'p-value' in first_cell.lower().replace(' ', '') or
        'pvalue' in first_cell.lower().replace(' ', '')
    )

def wrap_header_cell(cell: str) -> str:
    return '{' + cell + '}' if cell.strip() else cell


def wrap_pvalue_cells(cells: list) -> list:
    result = [cells[0]]
    for c in cells[1:]:
        try:
            if float(c) < 0.05:
                result.append('\\maxf{' + c + '}')
            else:
                result.append(c)
        except ValueError:
            result.append('\\maxf{' + c + '}')
    return result


def format_table(latex: str) -> str:
    """Return a reformatted LaTeX tabular wrapped in adjustbox."""
    orig_spec, rows = parse_tabular(latex)
    if not rows:
        return latex

    spec = build_col_spec(orig_spec)
    ncols = max(len(r) for r in rows)

    out = []
    out.append('\\begin{adjustbox}{width=\\columnwidth,center}')
    out.append(f'\\begin{{tabular}}{{{spec}}}')

    for row_idx, row in enumerate(rows):
        padded = row + [''] * (ncols - len(row))

        # wrap header cells in {}
        if row_idx == 0:
            padded = [wrap_header_cell(c) for c in padded]

        # wrap p-value data cells in \maxf{}
        if is_pvalue_row(padded[0]):
            padded = wrap_pvalue_cells(padded)

        line = ' & '.join(padded) + ' \\\\'
        out.append(line)

        # \hline under header row
        if row_idx == 0:
            out.append('\\hline')
        # \hline under p-value rows — but never after the very last row
        elif is_pvalue_row(padded[0]) and row_idx != len(rows) - 1:
            out.append('\\hline')

    out.append('\\end{tabular}')
    out.append('\\end{adjustbox}')
    return '\n '.join(out)


# ── table definitions ─────────────────────────────────────────────────────────

TABLE1 = r"""
\begin{tabular}{llllllll}
\toprule
 & SSW & HSN & PER & UHH & SNE & POW & NES \\
Method &  &  &  &  &  &  &  \\
\midrule
Omnibus: & 161.910 & 274.258 & 74.154 & 183.905 & 181.834 &  4.850 & 10.974 \\
Prob(Omnibus): &  0.000 &  0.000 &  0.000 &  0.000 &  0.000 &  0.088 &  0.004 \\
Skew: &  0.580 &  1.286 & -0.367 &  0.650 &  0.849 &  0.129 &  0.184 \\
Kurtosis: &  6.118 &  4.763 &  2.369 &  6.443 &  4.992 &  3.149 &  2.756 \\
Durbin-Watson: &    2.043 &    1.944 &    2.021 &    1.993 &    1.943 &    1.973 &    2.086 \\
Jarque-Bera (JB): &  590.211 &  516.826 &   50.022 &  717.407 &  365.533 &    4.758 &   10.402 \\
Prob(JB): & 6.87e-129 & 5.92e-113 & 1.37e-11 & 1.65e-156 & 4.22e-80 &   0.0926 &  0.00551 \\
Cond. No. &     3.68 &     3.21 &     2.34 &     3.45 &     4.24 &     3.36 &     4.86 \\
R-Square & 0.029000 & 0.203000 & 0.051000 & 0.026000 & 0.042000 & 0.298000 & 0.041000 \\
HC & 0.033236 & 0.389477 & -0.531993 & 1.452170 & 1.612055 & -0.570869 & -1.455542 \\
Prob(HC) & 0.973492 & 0.696989 & 0.594824 & 0.146702 & 0.107198 & 0.568189 & 0.145766 \\
QG & 0.969345 & 0.890881 & 0.966891 & 1.010882 & 0.976174 & 0.971717 & 0.882332 \\
Prob(QG) & 0.695196 & 0.146671 & 0.671778 & 0.892060 & 0.761533 & 0.718059 & 0.115273 \\
\bottomrule
\end{tabular}
"""

TABLE2 = r"""
\begin{tabular}{lrrrrr}
\toprule
 & entropy & complexity & acoustic_complexity & acoustic_diversity & bioacoustic_index \\
\midrule
SSW mean & 0.128658 & 0.011975 & 0.080662 & 0.076547 & 0.060563 \\
SSW p-value & 0.000000 & 0.230000 & 0.000000 & 0.000000 & 0.030000 \\
HSN mean & 0.515735 & 0.243675 & 0.238571 & 0.126380 & 0.063726 \\
HSN p-value & 0.000000 & 0.000000 & 0.000000 & 0.000000 & 0.000000 \\
PER mean & 0.283323 & 0.229406 & 0.145114 & 0.431127 & 0.081671 \\
PER p-value & 0.000000 & 0.000000 & 0.000000 & 0.000000 & 0.000000 \\
UHH mean & 0.020474 & 0.017151 & 0.262665 & 0.054517 & 0.107054 \\
UHH p-value & 0.060000 & 0.310000 & 0.000000 & 0.000000 & 0.000000 \\
SNE mean & 0.227538 & 0.090867 & 0.183407 & 0.206336 & 0.080270 \\
SNE p-value & 0.000000 & 0.000000 & 0.000000 & 0.000000 & 0.000000 \\
POW mean & 0.233137 & 0.541327 & 0.167056 & 0.587305 & 0.342548 \\
POW p-value & 0.000000 & 0.000000 & 0.000000 & 0.000000 & 0.000000 \\
NES mean & 0.129252 & 0.298772 & 0.184891 & 0.419490 & 0.128498 \\
NES p-value & 0.000000 & 0.000000 & 0.000000 & 0.000000 & 0.000000 \\
\bottomrule
\end{tabular}
"""

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t1 = format_table(TABLE1)
    t2 = format_table(TABLE2)
    result = t1 + '\n\n' + t2
    print(result)

    with open('/mnt/user-data/outputs/formatted_tables.txt', 'w') as f:
        f.write(result)
    print("\n\n[Saved to formatted_tables.txt]")