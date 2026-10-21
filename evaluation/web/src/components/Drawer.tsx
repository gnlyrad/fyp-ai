import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import type { ReactNode } from "react";

/** Right-side detail sheet for intermediate drill-downs. Open state is owned by the URL.
    `inspector` makes it a non-modal side panel: no dimming, the page behind stays readable and clickable, and only Escape or the
    close button closes it. */
export function Drawer({ open, onClose, title, subtitle, children, inspector = false }: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  subtitle?: ReactNode;
  children: ReactNode;
  inspector?: boolean;
}) {
  return (
    <Dialog.Root open={open} modal={!inspector} onOpenChange={(next) => { if (!next) onClose(); }}>
      <Dialog.Portal>
        {!inspector && <Dialog.Overlay className="fixed inset-0 z-30 bg-black/20" />}
        <Dialog.Content
          className={`fixed right-0 z-40 flex w-full flex-col border-l border-line bg-surface shadow-xl outline-none ${
            inspector ? "bottom-0 top-[var(--stt-header-h,48px)] max-w-[440px]" : "inset-y-0 max-w-[560px]"}`}
          aria-describedby={undefined}
          onInteractOutside={inspector ? (event) => event.preventDefault() : undefined}
          // Focus the drawer itself, not its first control: otherwise opening a metric drawer
          // focuses the title's info icon and pops its tooltip open.
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            (event.currentTarget as HTMLElement).focus();
          }}>
          <header className="flex items-start justify-between gap-3 border-b border-line px-5 py-3">
            <div className="min-w-0">
              <Dialog.Title className="truncate text-base font-semibold text-ink">{title}</Dialog.Title>
              {subtitle && <div className="mt-0.5 text-xs text-ink-2">{subtitle}</div>}
            </div>
            <Dialog.Close className="focus-ring rounded p-1 text-ink-3 hover:bg-surface-2 hover:text-ink" aria-label="Close">
              <X size={16} />
            </Dialog.Close>
          </header>
          <div className="flex-1 space-y-5 overflow-y-auto px-5 py-4">{children}</div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
