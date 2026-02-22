"use client";
import { useEffect, useRef, useState } from "react";
import { ssePost, SSEMessage } from "@/lib/sse";
import { ModelPicker } from "@/components/ModelPicker";
import { Uploader } from "@/components/Uploader";
import { Blocks } from "@/components/Blocks";
import { apiUrl } from "@/lib/api";

type Block = {
    block_id: string;
    title?: string;
    kind?: string;
    payload?: unknown;
};
type ChatMessage = {
    id: string;
    role: "user" | "assistant";
    text: string;
    blocks: Record<string, Block>;
    blockOrder: string[];
};

const SESSION_KEY = "omniagent_session_id";
const SERVER_BOOT_KEY = "omniagent_server_boot_id";
const GREETING_SEEN_KEY = "omniagent_greeting_seen";
const GREETING_TEXT =
    "Greetings, I'm OmniAgent, an Agentic Multi-Model System, I can generate text, images, audio and docs, maybe all at once. Try it out :)";

function createSessionId(): string {
    return `s_${Math.random().toString(16).slice(2, 10)}`;
}

export default function Page() {
    const [sessionId, setSessionId] = useState("");
    const [provider, setProvider] = useState("openai");
    const [model, setModel] = useState("gpt-4o-mini");

    const [input, setInput] = useState("");
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [busy, setBusy] = useState(false);
    const [activeAssistantId, setActiveAssistantId] = useState<string | null>(
        null,
    );
    const [showGreeting, setShowGreeting] = useState(false);
    const [greetingText, setGreetingText] = useState("");
    const tokenBufferRef = useRef<Record<string, string>>({});
    const flushTimerRef = useRef<number | null>(null);

    function onChange(p: string, m: string) {
        setProvider(p);
        setModel(m);
    }

    useEffect(() => {
        if (typeof window === "undefined") return;
        const v = window.localStorage.getItem(SESSION_KEY);
        if (v) {
            setSessionId(v);
            return;
        }
        const id = createSessionId();
        window.localStorage.setItem(SESSION_KEY, id);
        setSessionId(id);
    }, []);

    useEffect(() => {
        if (!sessionId || typeof window === "undefined") return;
        const saved = window.localStorage.getItem(
            `omniagent_chat_${sessionId}`,
        );
        if (!saved) return;
        try {
            setMessages(JSON.parse(saved));
        } catch {
            window.localStorage.removeItem(`omniagent_chat_${sessionId}`);
        }
    }, [sessionId]);

    useEffect(() => {
        if (!sessionId || typeof window === "undefined") return;
        let cancelled = false;
        async function syncWithServerBoot() {
            try {
                const r = await fetch(apiUrl("/api/session/meta"));
                if (!r.ok) return;
                const data = (await r.json()) as { server_boot_id?: string };
                const bootId = String(data.server_boot_id || "");
                if (!bootId || cancelled) return;
                const prevBoot = window.localStorage.getItem(SERVER_BOOT_KEY);
                if (prevBoot && prevBoot !== bootId) {
                    window.localStorage.removeItem(
                        `omniagent_chat_${sessionId}`,
                    );
                    setMessages([]);
                    const nextSession = createSessionId();
                    window.localStorage.setItem(SESSION_KEY, nextSession);
                    window.localStorage.setItem(SERVER_BOOT_KEY, bootId);
                    setSessionId(nextSession);
                    return;
                }
                if (!prevBoot) {
                    window.localStorage.setItem(SERVER_BOOT_KEY, bootId);
                }
            } catch {
                // Ignore transient server/meta errors.
            }
        }
        void syncWithServerBoot();
        return () => {
            cancelled = true;
        };
    }, [sessionId]);

    useEffect(() => {
        if (typeof window === "undefined") return;
        const seen = window.localStorage.getItem(GREETING_SEEN_KEY) === "1";
        setShowGreeting(true);
        if (seen) {
            setGreetingText(GREETING_TEXT);
            return;
        }
        setGreetingText("");
        let idx = 0;
        const timer = window.setInterval(() => {
            idx += 1;
            setGreetingText(GREETING_TEXT.slice(0, idx));
            if (idx >= GREETING_TEXT.length) {
                window.clearInterval(timer);
                window.localStorage.setItem(GREETING_SEEN_KEY, "1");
            }
        }, 18);
        return () => {
            window.clearInterval(timer);
        };
    }, []);

    useEffect(() => {
        if (!sessionId || typeof window === "undefined") return;
        window.localStorage.setItem(
            `omniagent_chat_${sessionId}`,
            JSON.stringify(messages),
        );
    }, [messages, sessionId]);

    function flushTokenBuffer() {
        const buffered = tokenBufferRef.current;
        const ids = Object.keys(buffered);
        if (ids.length === 0) return;

        tokenBufferRef.current = {};
        setMessages((m) =>
            m.map((x) => {
                const delta = buffered[x.id];
                if (!delta) return x;
                return { ...x, text: x.text + delta };
            }),
        );
    }

    function scheduleTokenFlush() {
        if (flushTimerRef.current !== null) return;
        flushTimerRef.current = window.setTimeout(() => {
            flushTokenBuffer();
            flushTimerRef.current = null;
        }, 40);
    }

    async function send() {
        if (!sessionId || !input.trim() || busy) return;
        const userText = input.trim();
        setBusy(true);
        const body = { session_id: sessionId, provider, model, text: userText };
        const userId = `m_${Date.now().toString(16)}_u`;
        const assistantId = `m_${Date.now().toString(16)}_a`;
        setMessages((m) => [
            ...m,
            {
                id: userId,
                role: "user",
                text: userText,
                blocks: {},
                blockOrder: [],
            },
            {
                id: assistantId,
                role: "assistant",
                text: "",
                blocks: {},
                blockOrder: [],
            },
        ]);
        setActiveAssistantId(assistantId);
        setInput("");

        try {
            for await (const msg of ssePost(apiUrl("/api/chat/stream"), body)) {
                handle(msg, assistantId);
            }
        } catch (e: unknown) {
            flushTokenBuffer();
            const err = e instanceof Error ? e.message : String(e);
            setMessages((m) =>
                m.map((x) =>
                    x.id === assistantId
                        ? { ...x, text: `${x.text}\n\n[error] ${err}` }
                        : x,
                ),
            );
        } finally {
            flushTokenBuffer();
            setActiveAssistantId(null);
            setBusy(false);
        }
    }

    async function clearChat() {
        if (!sessionId || busy) return;
        const current = sessionId;
        try {
            await fetch(apiUrl("/api/session/clear"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: current }),
            });
        } catch {
            // Best effort server clear; local clear still applies.
        }
        if (typeof window !== "undefined") {
            window.localStorage.removeItem(`omniagent_chat_${current}`);
            const nextSession = createSessionId();
            window.localStorage.setItem(SESSION_KEY, nextSession);
            setSessionId(nextSession);
        }
        setMessages([]);
        setInput("");
        setActiveAssistantId(null);
        tokenBufferRef.current = {};
    }

    function handle(msg: SSEMessage, assistantId: string) {
        if (msg.type === "token") {
            tokenBufferRef.current[assistantId] =
                `${tokenBufferRef.current[assistantId] || ""}${String(msg.data?.text || "")}`;
            scheduleTokenFlush();
            return;
        }
        if (msg.type === "block_start") {
            flushTokenBuffer();
            const rawId = msg.data?.block_id;
            if (typeof rawId !== "string" || rawId.length === 0) return;
            const id = rawId;
            const rawTitle = msg.data?.title;
            const rawKind = msg.data?.kind;
            const title = typeof rawTitle === "string" ? rawTitle : undefined;
            const kind = typeof rawKind === "string" ? rawKind : undefined;
            return setMessages((m) =>
                m.map((x) => {
                    if (x.id !== assistantId) return x;
                    return {
                        ...x,
                        blocks: {
                            ...x.blocks,
                            [id]: {
                                block_id: id,
                                title,
                                kind,
                            },
                        },
                        blockOrder: x.blockOrder.includes(id)
                            ? x.blockOrder
                            : [...x.blockOrder, id],
                    };
                }),
            );
        }
        if (msg.type === "block_end") {
            flushTokenBuffer();
            const rawId = msg.data?.block_id;
            if (typeof rawId !== "string" || rawId.length === 0) return;
            const id = rawId;
            return setMessages((m) =>
                m.map((x) => {
                    if (x.id !== assistantId) return x;
                    return {
                        ...x,
                        blocks: {
                            ...x.blocks,
                            [id]: {
                                ...(x.blocks[id] || { block_id: id }),
                                payload: msg.data?.payload,
                            },
                        },
                        blockOrder: x.blockOrder.includes(id)
                            ? x.blockOrder
                            : [...x.blockOrder, id],
                    };
                }),
            );
        }
        if (msg.type === "block_token") {
            flushTokenBuffer();
            const rawId = msg.data?.block_id;
            if (typeof rawId !== "string" || rawId.length === 0) return;
            const id = rawId;
            const tok = String(msg.data?.text || "");
            return setMessages((m) =>
                m.map((x) => {
                    if (x.id !== assistantId) return x;
                    const prev = x.blocks[id];
                    const prevPayload =
                        (prev?.payload as
                            | { data?: { text?: string; mime?: string } }
                            | undefined) || {};
                    const prevData = prevPayload.data || {};
                    const nextText = `${prevData.text || ""}${tok}`;
                    return {
                        ...x,
                        blocks: {
                            ...x.blocks,
                            [id]: {
                                ...(prev || { block_id: id }),
                                payload: {
                                    ...(prevPayload || {}),
                                    data: {
                                        ...prevData,
                                        text: nextText,
                                        mime: "text/markdown",
                                    },
                                },
                            },
                        },
                        blockOrder: x.blockOrder.includes(id)
                            ? x.blockOrder
                            : [...x.blockOrder, id],
                    };
                }),
            );
        }
        if (msg.type === "error") {
            flushTokenBuffer();
            return setMessages((m) =>
                m.map((x) =>
                    x.id === assistantId
                        ? {
                              ...x,
                              text: `${x.text}\n[error] ${msg.data?.error || "unknown"}\n`,
                          }
                        : x,
                ),
            );
        }
    }

    useEffect(() => {
        return () => {
            if (flushTimerRef.current !== null) {
                window.clearTimeout(flushTimerRef.current);
            }
        };
    }, []);

    return (
        <main className="mx-auto p-4 md:p-8 max-w-5xl">
            <div className="material-card p-4 md:p-5 mb-4">
                <div className="flex items-center justify-between gap-3 flex-wrap">
                    <div className="text-xl font-semibold tracking-tight">
                        OmniAgent
                    </div>
                    <ModelPicker
                        provider={provider}
                        model={model}
                        onChange={onChange}
                    />
                </div>
                <div className="flex items-start justify-between gap-3 mt-3 flex-wrap">
                    <div className="text-sm text-slate-600">
                        Session: {sessionId || "..."}
                    </div>
                    <div className="flex items-start gap-2">
                        {sessionId && (
                            <Uploader
                                sessionId={sessionId}
                                onUploaded={() => {}}
                            />
                        )}
                        <button
                            type="button"
                            onClick={() => void clearChat()}
                            disabled={!sessionId || busy}
                            className="material-btn-secondary"
                        >
                            Clear Chat
                        </button>
                    </div>
                </div>
            </div>

            <section className="material-card p-4 md:p-5 min-h-[360px]">
                <div className="space-y-4">
                    {messages.length === 0 && (
                        <div className="text-slate-500 text-sm">
                            {showGreeting ? greetingText : ""}
                        </div>
                    )}

                    {messages.map((m) => (
                        <article
                            key={m.id}
                            className={
                                m.role === "user"
                                    ? "chat-bubble user-bubble"
                                    : "chat-bubble assistant-bubble"
                            }
                        >
                            <div className="text-xs uppercase tracking-wide text-slate-500 mb-2">
                                {m.role === "user" ? "You" : "Assistant"}
                            </div>
                            {(() => {
                                const initialIds = m.blockOrder.filter(
                                    (id) =>
                                        m.blocks[id]?.kind === "meta_initial",
                                );
                                const conclusionIds = m.blockOrder.filter(
                                    (id) =>
                                        m.blocks[id]?.kind ===
                                        "meta_conclusion",
                                );
                                const otherIds = m.blockOrder.filter((id) => {
                                    const k = m.blocks[id]?.kind;
                                    return (
                                        k !== "meta_initial" &&
                                        k !== "meta_conclusion"
                                    );
                                });
                                return (
                                    <>
                                        <Blocks
                                            blocks={m.blocks}
                                            order={initialIds}
                                        />
                                        {m.text && (
                                            <Blocks
                                                blocks={{
                                                    text: {
                                                        block_id: "text",
                                                        title: "Text",
                                                        kind: "text",
                                                        payload: {
                                                            data: {
                                                                text: m.text,
                                                                mime: "text/markdown",
                                                            },
                                                        },
                                                    },
                                                }}
                                                order={["text"]}
                                            />
                                        )}
                                        <Blocks
                                            blocks={m.blocks}
                                            order={otherIds}
                                        />
                                        <Blocks
                                            blocks={m.blocks}
                                            order={conclusionIds}
                                        />
                                    </>
                                );
                            })()}
                            {m.role === "assistant" &&
                                m.id === activeAssistantId &&
                                !m.text &&
                                m.blockOrder.length === 0 && (
                                    <div className="text-sm text-slate-500">
                                        Generating...
                                    </div>
                                )}
                        </article>
                    ))}
                </div>
            </section>

            <div className="material-card p-3 md:p-4 mt-4">
                <div className="flex gap-2">
                    <input
                        className="material-input flex-1"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Type a message…"
                        disabled={!sessionId}
                        onKeyDown={(e) => (e.key === "Enter" ? send() : null)}
                    />
                    <button
                        onClick={send}
                        disabled={busy || !sessionId}
                        className="material-btn"
                    >
                        {busy ? "..." : "Send"}
                    </button>
                </div>
            </div>
            <footer className="mt-5 text-center text-xs text-slate-500">
                Developed by M. Santhosh Rishi © 2026
            </footer>
        </main>
    );
}
