// Small API helpers shared across pages. Keeps "fetch, check status,
// parse JSON, surface the backend's error message" consistent so we
// don't leak `Error: Failed to fetch` into the UI.

export async function parseApiError(response, fallback) {
  const fallbackMsg = fallback || `Request failed with status ${response.status}`;
  try {
    const body = await response.json();
    if (body?.detail) {
      return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
  } catch {
    // intentionally ignore JSON parse errors
  }

  try {
    const text = await response.text();
    return text || fallbackMsg;
  } catch {
    return fallbackMsg;
  }
}

// Fetches JSON from a URL. Returns `fallback` on any failure (network
// error, non-2xx, invalid JSON). Use when the page can keep working
// with a sensible default rather than blow up on a single bad response.
export async function fetchJson(url, fallback = null, init) {
  try {
    const res = await fetch(url, init);
    if (!res.ok) return fallback;
    return await res.json();
  } catch {
    return fallback;
  }
}
