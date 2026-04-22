import { useEffect, useState } from "react";

export default function BackendHealthBanner() {
  const [reachable, setReachable] = useState(true);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        const body = await res.json().catch(() => ({}));
        if (!cancelled) setReachable(res.ok && body?.ok === true);
      } catch {
        if (!cancelled) setReachable(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (reachable || dismissed) return null;

  return (
    <div className="backend-health-banner" role="status">
      <strong>Backend not reachable.</strong>
      <span>
        Start it with <code>cd backend && uvicorn main:app --reload</code>.
      </span>
      <button type="button" onClick={() => setDismissed(true)}>
        Dismiss
      </button>
    </div>
  );
}
