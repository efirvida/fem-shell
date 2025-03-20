import numpy as np


def print_matrix(matrix: np.ndarray, max_size: int = 8) -> None:
    """
    Print a 2D array as a bordered table string with truncation.

    Parameters
    ----------
    matrix : np.ndarray
        Input array to format.
    max_size : int, optional
        Maximum number of rows/columns to show, by default 8
    """
    matrix = np.array(matrix, dtype=object)
    nrows, ncols = matrix.shape
    cell_width = 14
    ellipsis_str = f"{'...':^{cell_width}}"

    def trunc_indices(total: int):
        if total <= max_size:
            return list(range(total)), []
        n_head = max_size // 2
        n_tail = max_size - n_head - 1
        return list(range(n_head)) + list(range(total - n_tail, total)), [n_head]

    row_idx, row_cuts = trunc_indices(nrows)
    col_idx, col_cuts = trunc_indices(ncols)
    truncated = matrix[np.ix_(row_idx, col_idx)]

    formatted = []
    for i, row in enumerate(truncated):
        if i in row_cuts:
            formatted.append([ellipsis_str] * len(col_idx))
        formatted_row = []
        for j, val in enumerate(row):
            if j in col_cuts:
                formatted_row.append(ellipsis_str)
            else:
                try:
                    num = float(val)
                    formatted_row.append(
                        f"{num:{cell_width}.3e}" if abs(num) > 1e-10 else " " * cell_width
                    )
                except Exception:
                    formatted_row.append(f"{str(val):^{cell_width}}")
        formatted.append(formatted_row)

    ncols_final = len(formatted[0])
    border = "+" + "+".join(["-" * (cell_width + 2)] * ncols_final) + "+"
    table_lines = [border]
    for row in formatted:
        table_lines.append("| " + " | ".join(row) + " |")
        table_lines.append(border)
    print("\n".join(table_lines))
