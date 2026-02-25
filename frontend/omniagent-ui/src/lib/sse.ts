export type SSEMessage = {
    type: string;
    data?: Record<string, unknown>;
    run_id?: string;
    trace_id?: string;
    ts_ms?: number;
};

export async function* ssePost(
    url: string,
    body: Record<string, unknown>,
    opts?: { signal?: AbortSignal },
): AsyncGenerator<SSEMessage> {
    const res = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
            "Cache-Control": "no-cache",
        },
        cache: "no-store",
        body: JSON.stringify(body),
        signal: opts?.signal,
    });
    if (!res.ok || !res.body) throw new Error(`SSE failed: ${res.status}`);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });

        const parts = buf.split(/\r?\n\r?\n/);
        buf = parts.pop() || "";

        for (const chunk of parts) {
            const json = chunk
                .split(/\r?\n/)
                .filter((l) => l.startsWith("data:"))
                .map((l) => l.replace(/^data:\s*/, ""))
                .join("\n");
            if (!json) continue;
            yield JSON.parse(json) as SSEMessage;
        }
    }

    // Some proxies may close streams without trailing blank line; parse tail once.
    const tail = buf.trim();
    if (!tail) return;
    const json = tail
        .split(/\r?\n/)
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.replace(/^data:\s*/, ""))
        .join("\n");
    if (!json) return;
    yield JSON.parse(json) as SSEMessage;
}
