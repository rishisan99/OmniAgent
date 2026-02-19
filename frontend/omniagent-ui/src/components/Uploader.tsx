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

    async function upload() {
        const file = ref.current?.files?.[0];
        if (!file) return;
        setBusy(true);
        try {
            const fd = new FormData();
            fd.append("session_id", sessionId);
            fd.append("f", file);
            const r = await fetch(`${api}/api/upload`, {
                method: "POST",
                body: fd,
            });
            if (!r.ok) throw new Error(`upload failed: ${r.status}`);
            onUploaded();
            if (ref.current) ref.current.value = "";
        } finally {
            setBusy(false);
        }
    }

    return (
        <div className="flex gap-2 items-center">
            <input ref={ref} type="file" className="text-sm max-w-[220px]" />
            <button
                onClick={upload}
                disabled={busy}
                className="material-btn-secondary"
            >
                {busy ? "Uploading…" : "Upload"}
            </button>
        </div>
    );
}
