import { ArrowDown, ArrowUp, ChevronRight } from "lucide-react";
import type { ReactNode } from "react";
import { useQueryState } from "../lib/url";
import { cx } from "./ui";

export type Column<T> = {
  key: string;
  header: ReactNode;
  cell: (row: T) => ReactNode;
  /** Sort value; omit to make the column unsortable. */
  sort?: (row: T) => string | number | null;
  align?: "left" | "right";
  className?: string;
  /** Cell-level drill: clicking the cell inspects this dimension instead of the whole row. */
  onCell?: (row: T) => void;
  /** Per-row guard for onCell (e.g. nothing to inspect). */
  cellEnabled?: (row: T) => boolean;
  cellTitle?: string;
};

/** Compact table. Row click = whole entity; drillable cell click = that dimension.
    Sort state lives in the URL (?<id>Sort=key:dir) so it survives drill-down and Back. */
export function DataTable<T>({ id, columns, rows, rowKey, onRowClick, rowTitle, empty }: {
  id: string;
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  onRowClick?: (row: T) => void;
  rowTitle?: string;
  empty?: ReactNode;
}) {
  const [params, setParams] = useQueryState();
  const sortParam = params.get(`${id}Sort`);
  const [sortKey, sortDir] = sortParam ? sortParam.split(":") : [null, null];
  const sortColumn = columns.find((column) => column.key === sortKey && column.sort);
  const sorted = sortColumn
    ? [...rows].sort((a, b) => {
        const left = sortColumn.sort!(a);
        const right = sortColumn.sort!(b);
        if (left === right) return 0;
        if (left === null) return 1;
        if (right === null) return -1;
        return (left < right ? -1 : 1) * (sortDir === "desc" ? -1 : 1);
      })
    : rows;

  const toggleSort = (key: string) => {
    const next = sortKey !== key ? `${key}:asc` : sortDir === "asc" ? `${key}:desc` : null;
    setParams({ [`${id}Sort`]: next }, { replace: true });
  };

  if (!rows.length) return <>{empty}</>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-line">
            {columns.map((column) => (
              <th key={column.key} scope="col"
                className={cx("whitespace-nowrap px-3 py-2 text-xs font-medium text-ink-3",
                  column.align === "right" ? "text-right" : "text-left", column.className)}>
                {column.sort ? (
                  <button type="button" onClick={() => toggleSort(column.key)}
                    className={cx("focus-ring inline-flex items-center gap-1 rounded hover:text-ink",
                      column.align === "right" && "flex-row-reverse")}>
                    {column.header}
                    {sortColumn?.key === column.key && (sortDir === "desc" ? <ArrowDown size={12} /> : <ArrowUp size={12} />)}
                  </button>
                ) : column.header}
              </th>
            ))}
            {onRowClick && <th className="w-6" aria-hidden />}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={rowKey(row)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              onKeyDown={onRowClick ? (event) => { if (event.key === "Enter") onRowClick(row); } : undefined}
              tabIndex={onRowClick ? 0 : undefined}
              title={rowTitle}
              className={cx("group border-b border-line last:border-0",
                onRowClick && "focus-ring cursor-pointer hover:bg-surface-2")}>
              {columns.map((column) => (
                <td key={column.key}
                  className={cx("px-3 py-1.5 align-middle", column.align === "right" ? "text-right" : "text-left", column.className)}>
                  {column.onCell && (column.cellEnabled?.(row) ?? true) ? (
                    <button type="button" title={column.cellTitle}
                      onClick={(event) => { event.stopPropagation(); column.onCell!(row); }}
                      onKeyDown={(event) => event.stopPropagation()}
                      className="drill num">
                      {column.cell(row)}
                    </button>
                  ) : column.cell(row)}
                </td>
              ))}
              {onRowClick && (
                <td className="pr-2 text-ink-3"><ChevronRight size={14} className="opacity-40 group-hover:opacity-100" /></td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
