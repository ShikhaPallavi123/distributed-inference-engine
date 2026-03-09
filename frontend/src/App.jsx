import { useState, useRef } from "react";

const API = "http://localhost:5000";

export default function App() {
  const [mode, setMode]         = useState("single");   // single | batch | stream
  const [input, setInput]       = useState("");
  const [results, setResults]   = useState([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");
  const [stats, setStats]       = useState(null);
  const eventSourceRef          = useRef(null);

  const reset = () => { setResults([]); setError(""); setStats(null); };

  // ── Single inference ──
  async function runSingle() {
    if (!input.trim()) return;
    reset(); setLoading(true);
    try {
      const res  = await fetch(`${API}/api/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResults([data]);
      setStats({ latency_ms: data.latency_ms, workers: 1 });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  // ── Batch inference (MPI) ──
  async function runBatch() {
    const texts = input.split("\n").map(t => t.trim()).filter(Boolean);
    if (!texts.length) return;
    reset(); setLoading(true);
    try {
      const res  = await fetch(`${API}/api/infer/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResults(data.results);
      setStats({ latency_ms: data.total_latency_ms, workers: data.workers_used });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  // ── Streaming inference (SSE) ──
  function runStream() {
    const texts = input.split("\n").map(t => t.trim()).filter(Boolean);
    if (!texts.length) return;
    reset(); setLoading(true);

    if (eventSourceRef.current) eventSourceRef.current.close();

    fetch(`${API}/api/infer/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts })
    }).then(res => {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      function read() {
        reader.read().then(({ done, value }) => {
          if (done) { setLoading(false); return; }
          buffer += decoder.decode(value);
          const lines = buffer.split("\n");
          buffer = lines.pop();
          lines.forEach(line => {
            if (!line.startsWith("data: ")) return;
            try {
              const data = JSON.parse(line.slice(6));
              if (data.done) { setLoading(false); return; }
              setResults(prev => [...prev, data]);
            } catch {}
          });
          read();
        });
      }
      read();
    }).catch(e => { setError(e.message); setLoading(false); });
  }

  const handleRun = () => {
    if (mode === "single") runSingle();
    else if (mode === "batch") runBatch();
    else runStream();
  };

  const labelColor = (label) =>
    label === "positive" ? "#16a34a" : label === "negative" ? "#dc2626" : "#6b7280";

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", maxWidth: 800, margin: "40px auto", padding: "0 20px", color: "#111" }}>

      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: "1.8rem", fontWeight: 700, marginBottom: 6 }}>
          Distributed AI Inference Engine
        </h1>
        <p style={{ color: "#6b7280", fontSize: "0.95rem" }}>
          React → Flask API → MPI Workers → ML Model · Real-time streaming via SSE
        </p>
      </div>

      {/* Mode selector */}
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        {["single", "batch", "stream"].map(m => (
          <button key={m} onClick={() => { setMode(m); reset(); }}
            style={{
              padding: "8px 18px", borderRadius: 4, border: "1.5px solid",
              borderColor: mode === m ? "#1a3a5c" : "#d1d5db",
              background: mode === m ? "#1a3a5c" : "white",
              color: mode === m ? "white" : "#374151",
              fontWeight: 600, cursor: "pointer", fontSize: "0.85rem", textTransform: "capitalize"
            }}>
            {m === "single" ? "Single" : m === "batch" ? "Batch (MPI)" : "Stream (SSE)"}
          </button>
        ))}
      </div>

      {/* Input */}
      <textarea
        value={input}
        onChange={e => setInput(e.target.value)}
        placeholder={
          mode === "single"
            ? "Enter text to analyse..."
            : "Enter one text per line — each will be sent to a separate worker"
        }
        rows={mode === "single" ? 3 : 6}
        style={{
          width: "100%", padding: 12, borderRadius: 6,
          border: "1.5px solid #d1d5db", fontSize: "0.95rem",
          fontFamily: "inherit", resize: "vertical", marginBottom: 12
        }}
      />

      {/* Run button */}
      <button onClick={handleRun} disabled={loading || !input.trim()}
        style={{
          padding: "11px 28px", background: loading ? "#9ca3af" : "#1a3a5c",
          color: "white", border: "none", borderRadius: 4,
          fontWeight: 700, fontSize: "0.95rem", cursor: loading ? "not-allowed" : "pointer",
          marginBottom: 24
        }}>
        {loading ? "Running..." : "Run Inference"}
      </button>

      {/* Error */}
      {error && (
        <div style={{ background: "#fef2f2", border: "1px solid #fca5a5", borderRadius: 6, padding: 12, marginBottom: 16, color: "#dc2626" }}>
          {error}
        </div>
      )}

      {/* Stats bar */}
      {stats && (
        <div style={{ display: "flex", gap: 24, background: "#f0f4ff", borderRadius: 6, padding: "10px 16px", marginBottom: 16, fontSize: "0.85rem" }}>
          <span>Latency: <strong>{stats.latency_ms}ms</strong></span>
          <span>Workers used: <strong>{stats.workers}</strong></span>
          <span>Results: <strong>{results.length}</strong></span>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {results.map((r, i) => (
            <div key={i} style={{
              border: "1px solid #e5e7eb", borderRadius: 8, padding: "14px 18px",
              borderLeft: `4px solid ${labelColor(r.label)}`,
              background: "white", display: "grid", gridTemplateColumns: "1fr auto", gap: 12
            }}>
              <div>
                <div style={{ fontSize: "0.9rem", color: "#374151", marginBottom: 4 }}>{r.text}</div>
                {r.worker !== undefined && (
                  <div style={{ fontSize: "0.75rem", color: "#9ca3af" }}>Worker {r.worker}</div>
                )}
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontWeight: 700, color: labelColor(r.label), fontSize: "1rem" }}>
                  {r.label}
                </div>
                <div style={{ fontSize: "0.82rem", color: "#6b7280" }}>
                  {(r.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
