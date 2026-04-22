import { useEffect, useState } from "react";

// Fetches `apiPath` (typically a splat binary from the backend) and
// returns a blob: URL that drei's <Splat> component can consume. Revokes
// the URL on unmount and on apiPath change to avoid leaking blobs.
//
// Returns `null` while loading, on error, or when apiPath is falsy.
export default function useBlobUrl(apiPath) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    let cancelled = false;
    let currentUrl = null;
    if (!apiPath) {
      setUrl(null);
      return;
    }
    (async () => {
      try {
        const res = await fetch(apiPath);
        if (!res.ok) throw new Error(`${res.status}`);
        const blob = await res.blob();
        if (cancelled) return;
        currentUrl = URL.createObjectURL(blob);
        setUrl(currentUrl);
      } catch {
        if (!cancelled) setUrl(null);
      }
    })();
    return () => {
      cancelled = true;
      if (currentUrl) {
        try {
          URL.revokeObjectURL(currentUrl);
        } catch {
          /* ignore */
        }
      }
    };
  }, [apiPath]);
  return url;
}
