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
): AsyncGenerator<SSEMessage> {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) throw new Error(`SSE failed: ${res.status}`);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });

        const parts = buf.split("\n\n");
        buf = parts.pop() || "";

        for (const chunk of parts) {
            const line = chunk.split("\n").find((l) => l.startsWith("data: "));
            if (!line) continue;
            const json = line.replace(/^data:\s*/, "");
            yield JSON.parse(json) as SSEMessage;
        }
    }
}
