"use client";
import { useRef, useState } from "react";

export function Uploader({
    sessionId,
    onUploaded,
}: {
    sessionId: string;
    onUploaded: () => void;
}) {
    const api = process.env.NEXT_PUBLIC_API_BASE!;
    const ref = useRef<HTMLInputElement>(null);
    const [busy, setBusy] = useState(false);
    const [status, setStatus] = useState("");

    async function upload(selected?: File | null) {
        const picked = selected instanceof File ? selected : null;
        const f = picked || ref.current?.files?.[0] || null;
        if (!f) {
            setStatus("Please choose a file first");
            return;
        }
        setStatus(`Uploading ${f.name}...`);
        setBusy(true);
        let timeout: ReturnType<typeof setTimeout> | undefined;
        try {
            const ctrl = new AbortController();
            timeout = setTimeout(() => ctrl.abort(), 30000);
            const fd = new FormData();
            fd.append("session_id", sessionId);
            fd.append("f", f);
            const r = await fetch(`${api}/api/upload`, {
                method: "POST",
                body: fd,
                signal: ctrl.signal,
            });
            if (!r.ok) throw new Error(`upload failed: ${r.status}`);
            onUploaded();
            if (ref.current) ref.current.value = "";
            setStatus(`Upload complete: ${f.name}`);
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e);
            setStatus(`Upload failed: ${msg}`);
        } finally {
            if (timeout) clearTimeout(timeout);
            setBusy(false);
        }
    }

    return (
        <div className="flex gap-2 items-center">
            <input
                ref={ref}
                type="file"
                className="hidden"
                accept="image/*,.pdf,.txt,.md,.doc,.docx"
                onChange={(e) => {
                    const picked = e.target.files?.[0] || null;
                    if (picked) {
                        void upload(picked);
                    }
                }}
            />
            <button
                onClick={() => {
                    ref.current?.click();
                }}
                disabled={busy}
                className="material-btn-secondary"
            >
                {busy ? "Uploading…" : "Upload File"}
            </button>
            {status && <span className="text-xs text-slate-500">{status}</span>}
        </div>
    );
}
