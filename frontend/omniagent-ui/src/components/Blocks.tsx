"use client";
import { useEffect, useState } from "react";
import { apiBase } from "@/lib/api";

type Block = { block_id: string; title?: string; kind?: string; payload?: unknown };

function esc(s: string): string {
    return s
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function allowClickable(url: string): boolean {
    try {
        const u = new URL(url);
        const host = u.hostname.toLowerCase();
        if (host === "example.com" || host.endsWith(".example.com")) return false;
        return u.protocol === "http:" || u.protocol === "https:";
    } catch {
        return false;
    }
}

function mdToHtml(src: string): string {
    const raw = (src || "").replace(/\r\n/g, "\n").trim();
    const fenced = raw.match(/^```(?:markdown|md)?\n([\s\S]*?)\n```$/i);
    const normalized = (fenced ? fenced[1] : raw)
        .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, "$1")
        .replace(/\[[^\]]+\]\(sandbox:[^)]+\)/gi, "");
    const lines = normalized.split("\n");
    let inCode = false;
    const out: string[] = [];
    for (const raw of lines) {
        const line = raw ?? "";
        const ltrim = line.trimStart();
        if (ltrim.startsWith("```")) {
            inCode = !inCode;
            out.push(inCode ? "<pre><code>" : "</code></pre>");
            continue;
        }
        if (inCode) {
            out.push(`${esc(line)}\n`);
            continue;
        }
        if (ltrim.startsWith("### ")) out.push(`<h3>${esc(ltrim.slice(4))}</h3>`);
        else if (ltrim.startsWith("## ")) out.push(`<h2>${esc(ltrim.slice(3))}</h2>`);
        else if (ltrim.startsWith("# ")) out.push(`<h1>${esc(ltrim.slice(2))}</h1>`);
        else if (ltrim.startsWith("- ") || ltrim.startsWith("* ")) out.push(`<li>${esc(ltrim.slice(2))}</li>`);
        else if (line.trim() === "") out.push("<br/>");
        else out.push(`<p>${esc(line)}</p>`);
    }
    const html = out.join("");
    return html
        .replace(/<li>/g, "<ul><li>")
        .replace(/<\/li>/g, "</li></ul>")
        .replace(/<\/ul><ul>/g, "")
        .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_m, label: string, url: string) => {
            if (!allowClickable(url)) return esc(label);
            return `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)}</a>`;
        })
        .replace(/(^|[\s>(])(https?:\/\/[^\s<)]+)/g, (_m, prefix: string, url: string) => {
            if (!allowClickable(url)) return `${prefix}${esc(url)}`;
            return `${prefix}<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(url)}</a>`;
        })
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/`(.+?)`/g, "<code>$1</code>");
}

function PendingCountdown({ kind }: { kind?: string }) {
    const totalSec = kind === "image_gen" ? 60 : kind === "doc" ? 15 : 0;
    const [remainingMs, setRemainingMs] = useState(totalSec * 1000);

    useEffect(() => {
        if (totalSec <= 0) return;
        const start = Date.now();
        const timer = window.setInterval(() => {
            const elapsed = Date.now() - start;
            const next = Math.max(0, totalSec * 1000 - elapsed);
            setRemainingMs(next);
            if (next <= 0) {
                window.clearInterval(timer);
            }
        }, 50);
        return () => {
            window.clearInterval(timer);
        };
    }, [totalSec]);

    const label =
        kind === "image_gen"
            ? "Generating Image"
            : kind === "doc"
                ? "Generating Document"
                : "Generating";
    if (totalSec <= 0) {
        return <>{`< ${label} ... />`}</>;
    }
    return <>{`< ${label} ... ${Math.max(0, remainingMs / 1000).toFixed(2)} sec />`}</>;
}

export function Blocks({
    blocks,
    order,
}: {
    blocks: Record<string, Block>;
    order?: string[];
}) {
    const api = apiBase();
    const showKnowledgeBlocks = process.env.NEXT_PUBLIC_SHOW_KNOWLEDGE_BLOCKS === "true";
    const actionBtnClass =
        "inline-flex items-center rounded-md border border-slate-300 bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-800 transition hover:bg-slate-200";

    async function forceDownload(url: string, filename: string) {
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`Download failed: ${res.status}`);
            const blob = await res.blob();
            const obj = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = obj;
            a.download = filename || "download";
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(obj);
        } catch {
            // Fallback if blob download fails
            const a = document.createElement("a");
            a.href = url;
            a.download = filename || "download";
            document.body.appendChild(a);
            a.click();
            a.remove();
        }
    }

    const ids = order !== undefined ? order : Object.keys(blocks);
    const list = ids
        .map((id) => blocks[id])
        .filter(Boolean)
        .filter(
            (b) =>
                showKnowledgeBlocks
                || (b.kind !== "web" && b.kind !== "rag" && b.kind !== "kb_rag" && b.kind !== "vision"),
        );

    return (
        <div className="space-y-2">
            {list.map((b) => {
                const payloadObj = (b.payload as { data?: Record<string, unknown> } | undefined) || {};
                const p = (payloadObj.data || payloadObj) as Record<string, unknown>;
                const url = typeof p.url === "string" ? `${api}${p.url}` : null;
                const mime = typeof p.mime === "string" ? p.mime : "";
                const filename = typeof p.filename === "string" ? p.filename : "document";
                const isPending = !b.payload;
                const md = typeof p.text === "string" ? p.text : "";
                const isMeta = b.kind === "meta_initial" || b.kind === "meta_conclusion";

                return (
                    <div
                        key={b.block_id}
                        className={
                            isMeta
                                ? "rounded-xl px-3 py-2 bg-slate-50 border border-slate-200"
                                : "border border-slate-200 rounded-xl p-3 bg-white"
                        }
                    >
                        {!isMeta && b.kind !== "text" && (
                            <div className="text-sm font-semibold">
                                {b.title || b.block_id}
                            </div>
                        )}

                        {isPending && (
                            <div className="mt-2 text-sm text-slate-500">
                                <PendingCountdown kind={b.kind} />
                            </div>
                        )}

                        {b.kind !== "doc" && mime === "text/markdown" && md && (
                            <div
                                className="mt-2 markdown-body"
                                dangerouslySetInnerHTML={{ __html: mdToHtml(md) }}
                            />
                        )}
                        {url && typeof mime === "string" && mime.startsWith("image/") && (
                            <div className="mt-2">
                                {/* eslint-disable-next-line @next/next/no-img-element */}
                                <img
                                    src={url}
                                    alt={b.title || "image"}
                                    className="w-full max-w-[360px] rounded-lg border border-slate-100"
                                />
                                <div className="mt-2 flex items-center gap-3">
                                    <button
                                        type="button"
                                        onClick={() => void forceDownload(url, filename)}
                                        className={actionBtnClass}
                                    >
                                        Download
                                    </button>
                                    <a
                                        href={url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className={actionBtnClass}
                                    >
                                        View
                                    </a>
                                </div>
                            </div>
                        )}

                        {url && typeof mime === "string" && mime.startsWith("audio/") && (
                            <audio controls className="mt-2 w-full" src={url} />
                        )}

                        {b.kind === "doc" && url && (
                            <div className="mt-2 flex items-center gap-3">
                                <span className="text-sm text-slate-700">{filename}</span>
                                <button
                                    type="button"
                                    onClick={() => void forceDownload(url, filename)}
                                    className={actionBtnClass}
                                >
                                    Download
                                </button>
                                <a
                                    href={url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className={actionBtnClass}
                                >
                                    View
                                </a>
                            </div>
                        )}

                        {b.kind !== "doc" && url && typeof mime === "string" && mime.startsWith("text/") && (
                            <a href={url} target="_blank" className="mt-2 inline-block text-sm text-blue-600 underline">
                                Open generated document
                            </a>
                        )}

                        {!url && !isPending && mime !== "text/markdown" && b.kind !== "web" && b.kind !== "vision" && (
                            <pre className="mt-2 text-xs whitespace-pre-wrap break-words bg-slate-50 p-2 rounded">
                                {JSON.stringify(b.payload, null, 2)}
                            </pre>
                        )}
                    </div>
                );
            })}
        </div>
    );
}
