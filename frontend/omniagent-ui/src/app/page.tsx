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
type Artifact = { id?: string; url?: string; text?: string };
type SessionStateResp = {
    artifact_memory?: {
        image?: Artifact | null;
        audio?: Artifact | null;
        doc?: Artifact | null;
    };
};
type ArtifactBaseline = {
    image?: string;
    audio?: string;
    doc?: string;
};
type MediaExpect = {
    image: boolean;
    audio: boolean;
    doc: boolean;
};

const SESSION_KEY = "omniagent_session_id";
const SERVER_BOOT_KEY = "omniagent_server_boot_id";
const GREETING_SEEN_KEY = "omniagent_greeting_seen";
const GREETING_TEXT =
    "Greetings, I'm OmniAgent, an Agentic Multi-Model System, I can generate text, images, audio and docs, maybe all at once. Try it out :)";

function createSessionId(): string {
    return `s_${Math.random().toString(16).slice(2, 10)}`;
}

function mimeFromUrl(url: string, fallback: string): string {
    const u = (url || "").toLowerCase();
    if (u.endsWith(".png")) return "image/png";
    if (u.endsWith(".jpg") || u.endsWith(".jpeg")) return "image/jpeg";
    if (u.endsWith(".webp")) return "image/webp";
    if (u.endsWith(".gif")) return "image/gif";
    if (u.endsWith(".mp3")) return "audio/mpeg";
    if (u.endsWith(".wav")) return "audio/wav";
    if (u.endsWith(".m4a")) return "audio/mp4";
    if (u.endsWith(".pdf")) return "application/pdf";
    if (u.endsWith(".md") || u.endsWith(".txt")) return "text/markdown";
    return fallback;
}

function timestampFromAssetName(name: string): number | null {
    const m = String(name || "").match(/_(\d{13})(?:\.[A-Za-z0-9]+)?$/);
    if (!m) return null;
    const n = Number(m[1]);
    return Number.isFinite(n) ? n : null;
}

function assetTimestampMs(a: Artifact | null | undefined): number {
    if (!a) return 0;
    const idTs = a.id ? timestampFromAssetName(a.id) : null;
    if (idTs) return idTs;
    const url = String(a.url || "");
    const base = url.split("/").pop() || "";
    const urlTs = timestampFromAssetName(base);
    return urlTs || 0;
}

function artifactIdentity(a: Artifact | null | undefined): string {
    if (!a) return "";
    const id = String(a.id || "").trim();
    if (id) return id;
    const u = String(a.url || "").trim();
    if (!u) return "";
    return u.split("/").pop() || "";
}

function expectedMediaFromText(text: string): MediaExpect {
    const t = (text || "").toLowerCase();
    const image = /\b(generate|create|make)\b[\s\S]{0,40}\b(image|picture|photo)\b/.test(t)
        || /\b(image|picture|photo)\s+of\b/.test(t);
    const audio = /\b(generate|create|make)\b[\s\S]{0,40}\b(audio|voice|tts)\b/.test(t)
        || /\b(read aloud|narrate|speak)\b/.test(t);
    const doc = /\b(generate|create|make|write|export)\b[\s\S]{0,40}\b(pdf|document|docx|text file|txt|markdown)\b/.test(t);
    return { image, audio, doc };
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
    const reconcileTimerRef = useRef<number | null>(null);
    const reconcileDeadlineRef = useRef<number>(0);
    const messagesRef = useRef<ChatMessage[]>([]);

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

    useEffect(() => {
        messagesRef.current = messages;
    }, [messages]);

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

    function stopReconcileWatch() {
        if (reconcileTimerRef.current !== null) {
            window.clearInterval(reconcileTimerRef.current);
            reconcileTimerRef.current = null;
        }
        reconcileDeadlineRef.current = 0;
    }

    async function readArtifactBaseline(): Promise<ArtifactBaseline> {
        if (!sessionId) return {};
        try {
            const r = await fetch(
                `${apiUrl("/api/session/state")}?session_id=${encodeURIComponent(sessionId)}`,
            );
            if (!r.ok) return {};
            const data = (await r.json()) as SessionStateResp;
            const mem = data.artifact_memory || {};
            return {
                image: artifactIdentity(mem.image || null),
                audio: artifactIdentity(mem.audio || null),
                doc: artifactIdentity(mem.doc || null),
            };
        } catch {
            return {};
        }
    }

    async function reconcilePendingMediaBlocks(
        assistantId: string,
        baseline: ArtifactBaseline = {},
        expected: MediaExpect = { image: true, audio: true, doc: true },
    ) {
        if (!sessionId) return;
        try {
            const r = await fetch(
                `${apiUrl("/api/session/state")}?session_id=${encodeURIComponent(sessionId)}`,
            );
            if (!r.ok) return;
            const data = (await r.json()) as SessionStateResp;
            const mem = data.artifact_memory || {};
            const image = mem.image || null;
            const audio = mem.audio || null;
            const doc = mem.doc || null;
            const imageUrl = String(image?.url || "");
            const audioUrl = String(audio?.url || "");
            const docUrl = String(doc?.url || "");
            const imageName = String(image?.id || "image.png");
            const audioName = String(audio?.id || "audio.mp3");
            const docName = String(doc?.id || "document");
            const docText = String(doc?.text || "");
            const imageId = artifactIdentity(image);
            const audioId = artifactIdentity(audio);
            const docId = artifactIdentity(doc);
            const imageTs = assetTimestampMs(image);
            const audioTs = assetTimestampMs(audio);
            const docTs = assetTimestampMs(doc);
            const imageFresh = expected.image && Boolean(imageUrl && imageId && imageId !== String(baseline.image || ""));
            const audioFresh = expected.audio && Boolean(audioUrl && audioId && audioId !== String(baseline.audio || ""));
            const docFresh = expected.doc && Boolean(docUrl && docId && docId !== String(baseline.doc || ""));

            setMessages((all) =>
                all.map((m) => {
                    if (m.id !== assistantId) return m;
                    const nextBlocks = { ...m.blocks };
                    let nextOrder = [...m.blockOrder];
                    for (const id of m.blockOrder) {
                        const b = nextBlocks[id];
                        if (!b || b.payload) continue;
                        if (b.kind === "image_gen" && imageFresh) {
                            nextBlocks[id] = {
                                ...b,
                                payload: {
                                    ok: true,
                                    kind: "image_gen",
                                    data: {
                                        url: imageUrl,
                                        filename: imageName,
                                        mime: mimeFromUrl(imageUrl, "image/png"),
                                    },
                                },
                            };
                        } else if (b.kind === "tts" && audioFresh) {
                            nextBlocks[id] = {
                                ...b,
                                payload: {
                                    ok: true,
                                    kind: "tts",
                                    data: {
                                        url: audioUrl,
                                        filename: audioName,
                                        mime: mimeFromUrl(audioUrl, "audio/mpeg"),
                                    },
                                },
                            };
                        } else if (b.kind === "doc" && docFresh) {
                            nextBlocks[id] = {
                                ...b,
                                payload: {
                                    ok: true,
                                    kind: "doc",
                                    data: {
                                        url: docUrl,
                                        filename: docName,
                                        text: docText,
                                        mime: mimeFromUrl(docUrl, "text/markdown"),
                                    },
                                },
                            };
                        }
                    }
                    const hasKind = (k: string) =>
                        nextOrder.some((id) => String(nextBlocks[id]?.kind || "") === k);
                    if (imageFresh && !hasKind("image_gen")) {
                        const id = `recon_image_${imageTs || Date.now()}`;
                        nextBlocks[id] = {
                            block_id: id,
                            title: "Generated Image",
                            kind: "image_gen",
                            payload: {
                                ok: true,
                                kind: "image_gen",
                                data: {
                                    url: imageUrl,
                                    filename: imageName,
                                    mime: mimeFromUrl(imageUrl, "image/png"),
                                },
                            },
                        };
                        nextOrder = [...nextOrder, id];
                    }
                    if (audioFresh && !hasKind("tts")) {
                        const id = `recon_audio_${audioTs || Date.now()}`;
                        nextBlocks[id] = {
                            block_id: id,
                            title: "Generated Audio",
                            kind: "tts",
                            payload: {
                                ok: true,
                                kind: "tts",
                                data: {
                                    url: audioUrl,
                                    filename: audioName,
                                    mime: mimeFromUrl(audioUrl, "audio/mpeg"),
                                },
                            },
                        };
                        nextOrder = [...nextOrder, id];
                    }
                    if (docFresh && !hasKind("doc")) {
                        const id = `recon_doc_${docTs || Date.now()}`;
                        nextBlocks[id] = {
                            block_id: id,
                            title: "Generated Document",
                            kind: "doc",
                            payload: {
                                ok: true,
                                kind: "doc",
                                data: {
                                    url: docUrl,
                                    filename: docName,
                                    text: docText,
                                    mime: mimeFromUrl(docUrl, "text/markdown"),
                                },
                            },
                        };
                        nextOrder = [...nextOrder, id];
                    }
                    return { ...m, blocks: nextBlocks, blockOrder: nextOrder };
                }),
            );
        } catch {
            // Best-effort reconciliation if stream dropped block_end events.
        }
    }

    function startReconcileWatch(assistantId: string, baseline: ArtifactBaseline, expected: MediaExpect) {
        stopReconcileWatch();
        reconcileDeadlineRef.current = Date.now() + 30_000;
        reconcileTimerRef.current = window.setInterval(() => {
            void (async () => {
                await reconcilePendingMediaBlocks(assistantId, baseline, expected);
                const msg = messagesRef.current.find((m) => m.id === assistantId);
                if (!msg) {
                    stopReconcileWatch();
                    return;
                }
                const mediaKinds = new Set(["image_gen", "tts", "doc"]);
                let hasPending = false;
                let hasResolved = false;
                for (const id of msg.blockOrder) {
                    const b = msg.blocks[id];
                    if (!b || !mediaKinds.has(String(b.kind || ""))) continue;
                    if (b.payload) hasResolved = true;
                    else hasPending = true;
                }
                if (!hasPending && hasResolved) {
                    stopReconcileWatch();
                    setActiveAssistantId((cur) => (cur === assistantId ? null : cur));
                    setBusy(false);
                    return;
                }
                if (Date.now() > reconcileDeadlineRef.current) {
                    stopReconcileWatch();
                }
            })();
        }, 2500);
    }

    async function send() {
        if (!sessionId || !input.trim() || busy) return;
        const userText = input.trim();
        stopReconcileWatch();
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
        const baseline = await readArtifactBaseline();
        const expected = expectedMediaFromText(userText);
        const expectsMedia = expected.image || expected.audio || expected.doc;

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
            await reconcilePendingMediaBlocks(assistantId, baseline, expected);
            const unresolved = messagesRef.current.some((m) => {
                if (m.id !== assistantId) return false;
                return m.blockOrder.some((id) => {
                    const b = m.blocks[id];
                    const k = String(b?.kind || "");
                    return (k === "image_gen" || k === "tts" || k === "doc") && !b?.payload;
                });
            });
            if (unresolved) {
                // Start post-run recovery only if media blocks are still pending.
                startReconcileWatch(assistantId, baseline, expected);
            } else if (expectsMedia) {
                // For media-intent turns, keep short recovery polling in case stream
                // dropped block events; scope it to expected media kinds only.
                startReconcileWatch(assistantId, baseline, expected);
            }
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
        if (msg.type === "task_result") {
            const taskId = msg.data?.task_id;
            const kind = msg.data?.kind;
            const ok = msg.data?.ok;
            if (typeof taskId !== "string" || kind !== "image_gen" || ok !== true) return;
            const url =
                typeof msg.data?.url === "string" && msg.data.url.trim()
                    ? msg.data.url.trim()
                    : "";
            if (!url) return;
            const filename =
                typeof msg.data?.filename === "string" && msg.data.filename.trim()
                    ? msg.data.filename.trim()
                    : "image.png";
            const mime =
                typeof msg.data?.mime === "string" && msg.data.mime.trim()
                    ? msg.data.mime.trim()
                    : "image/png";
            return setMessages((m) =>
                m.map((x) => {
                    if (x.id !== assistantId) return x;
                    const prev = x.blocks[taskId] || { block_id: taskId, kind: "image_gen", title: "Generated Image" };
                    if (prev.payload) return x;
                    return {
                        ...x,
                        blocks: {
                            ...x.blocks,
                            [taskId]: {
                                ...prev,
                                payload: {
                                    ok: true,
                                    kind: "image_gen",
                                    data: { url, filename, mime },
                                },
                            },
                        },
                        blockOrder: x.blockOrder.includes(taskId) ? x.blockOrder : [...x.blockOrder, taskId],
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
            stopReconcileWatch();
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
