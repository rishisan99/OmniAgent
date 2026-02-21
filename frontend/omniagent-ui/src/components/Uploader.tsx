"use client";
import { useEffect, useRef, useState } from "react";

type Attachment = {
    id: string;
    kind: string;
    name?: string;
    mime?: string;
};

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
    const [attachments, setAttachments] = useState<Attachment[]>([]);

    async function refreshAttachments() {
        try {
            const r = await fetch(`${api}/api/uploads/${sessionId}`);
            if (!r.ok) throw new Error(`list failed: ${r.status}`);
            const data = (await r.json()) as { attachments?: Attachment[] };
            setAttachments(Array.isArray(data.attachments) ? data.attachments : []);
        } catch {
            // no-op to avoid noisy UX
        }
    }

    useEffect(() => {
        void refreshAttachments();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [sessionId]);

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
            await refreshAttachments();
            if (ref.current) ref.current.value = "";
            setStatus("");
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e);
            setStatus(`Upload failed: ${msg}`);
        } finally {
            if (timeout) clearTimeout(timeout);
            setBusy(false);
        }
    }

    async function removeAttachment(id: string) {
        setBusy(true);
        setStatus("Removing from context...");
        try {
            const r = await fetch(`${api}/api/uploads/${sessionId}/${id}`, {
                method: "DELETE",
            });
            if (!r.ok) throw new Error(`remove failed: ${r.status}`);
            await refreshAttachments();
            setStatus("Removed from context");
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e);
            setStatus(`Remove failed: ${msg}`);
        } finally {
            setBusy(false);
        }
    }

    return (
        <div className="flex flex-col items-start gap-2">
            <div className="flex flex-col items-start gap-1">
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
                        {busy ? "Working…" : "Upload File"}
                    </button>
                </div>
                {status && status.toLowerCase().startsWith("upload failed") && (
                    <span className="text-xs text-slate-500">{status}</span>
                )}
            </div>
            {attachments.length > 0 && (
                <div className="flex flex-wrap gap-2">
                    {attachments.map((a) => (
                        <div
                            key={a.id}
                            className="inline-flex items-center gap-2 rounded-full border border-slate-300 bg-slate-50 px-2 py-1 text-xs"
                        >
                            <span className="text-slate-700">
                                {a.kind}: {a.name || a.id}
                            </span>
                            <button
                                type="button"
                                disabled={busy}
                                onClick={() => {
                                    void removeAttachment(a.id);
                                }}
                                aria-label={`Remove ${a.name || a.id}`}
                                className="text-red-500 font-semibold disabled:text-slate-400"
                            >
                                X
                            </button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
