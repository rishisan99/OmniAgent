export function apiBase(): string {
    const raw = process.env.NEXT_PUBLIC_API_BASE || "";
    return raw.replace(/\/+$/, "");
}

export function apiUrl(path: string): string {
    const base = apiBase();
    if (!path.startsWith("/")) return `${base}/${path}`;
    return `${base}${path}`;
}
