"use client";
import { useEffect, useState } from "react";

type ModelsResp = {
    providers: string[];
    models: Record<string, string[]>;
    default: { provider: string; model: string };
};

export function ModelPicker(props: {
    provider: string;
    model: string;
    onChange: (p: string, m: string) => void;
}) {
    const api = process.env.NEXT_PUBLIC_API_BASE!;
    const [data, setData] = useState<ModelsResp | null>(null);

    useEffect(() => {
        fetch(`${api}/api/models`)
            .then((r) => r.json())
            .then(setData);
    }, [api]);

    useEffect(() => {
        if (data) props.onChange(data.default.provider, data.default.model);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [data]);

    if (!data) return <div className="text-sm opacity-70">Loading models…</div>;

    const models = data.models[props.provider] || [];
    return (
        <div className="flex gap-2 items-center">
            <select
                className="material-select"
                value={props.provider}
                onChange={(e) =>
                    props.onChange(
                        e.target.value,
                        (data.models[e.target.value] || [])[0] || "",
                    )
                }
            >
                {data.providers.map((p) => (
                    <option key={p} value={p}>
                        {p}
                    </option>
                ))}
            </select>

            <select
                className="material-select"
                value={props.model}
                onChange={(e) => props.onChange(props.provider, e.target.value)}
            >
                {models.map((m) => (
                    <option key={m} value={m}>
                        {m}
                    </option>
                ))}
            </select>
        </div>
    );
}
