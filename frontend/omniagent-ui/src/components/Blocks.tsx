"use client";

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

export function Blocks({
    blocks,
    order,
}: {
    blocks: Record<string, Block>;
    order?: string[];
}) {
    const api = process.env.NEXT_PUBLIC_API_BASE!;

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

    const ids = order?.length ? order : Object.keys(blocks);
    const list = ids
        .map((id) => blocks[id])
        .filter(Boolean)
        .filter((b) => b.kind !== "web");

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
                const label =
                    b.kind === "image_gen"
                        ? "Image"
                        : b.kind === "tts"
                          ? "Audio"
                          : b.kind === "doc"
                            ? "Document"
                            : b.kind === "rag"
                              ? "Document Context"
                              : b.kind === "web"
                                ? "Web Results"
                                : "Result";

                return (
                    <div key={b.block_id} className="border border-slate-200 rounded-xl p-3 bg-white">
                        <div className="text-sm font-semibold">
                            {b.title || b.block_id}
                        </div>

                        {isPending && (
                            <div className="mt-2 text-sm text-slate-500">
                                {`< Generating ${label} ... />`}
                            </div>
                        )}

                        {b.kind !== "doc" && mime === "text/markdown" && md && (
                            <div
                                className="mt-2 markdown-body"
                                dangerouslySetInnerHTML={{ __html: mdToHtml(md) }}
                            />
                        )}

                        {url && typeof mime === "string" && mime.startsWith("image/") && (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img
                                src={url}
                                alt={b.title || "image"}
                                className="mt-2 w-full max-w-[360px] rounded-lg border border-slate-100"
                            />
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
                                    className="inline-block text-sm text-blue-600 underline"
                                >
                                    Download
                                </button>
                                <a
                                    href={url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-block text-sm text-blue-600 underline"
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

                        {!url && !isPending && mime !== "text/markdown" && (
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
