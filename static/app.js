const $ = (id) => document.getElementById(id);

const textEl = $("mood-text");
const btn = $("analyze-btn");
const statusEl = $("status");
const results = $("results");
const moodLabel = $("mood-label");
const moodRationale = $("mood-rationale");
const confidenceBar = $("confidence-bar");
const confidenceVal = $("confidence-val");
const vaderList = $("vader-list");
const listSongs = $("list-songs");
const listMovies = $("list-movies");
const listSeries = $("list-series");

function setLoading(loading) {
  btn.disabled = loading;
  statusEl.textContent = loading ? "Analyzing…" : "";
}

function renderVader(v) {
  vaderList.innerHTML = "";
  const order = ["compound", "pos", "neg", "neu"];
  for (const k of order) {
    if (v[k] === undefined) continue;
    const li = document.createElement("li");
    li.textContent = `${k}: ${v[k]}`;
    vaderList.appendChild(li);
  }
}

function renderList(ul, items, type) {
  ul.innerHTML = "";
  for (const item of items) {
    const li = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = item.title;
    li.appendChild(title);
    const meta = document.createElement("div");
    meta.className = "item-meta";
    if (type === "song") {
      meta.textContent = item.artist ? `${item.artist} · ${item.note || ""}` : item.note || "";
    } else {
      const y = item.year ? ` (${item.year})` : "";
      meta.textContent = `${item.note || ""}${y}`.trim();
    }
    if (meta.textContent) li.appendChild(meta);
    ul.appendChild(li);
  }
}

async function analyze() {
  const text = textEl.value.trim();
  if (!text) {
    statusEl.textContent = "Add a short mood description first.";
    return;
  }
  setLoading(true);
  results.hidden = true;
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, per_category: 5 }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const data = await res.json();
    const a = data.analysis;
    const r = data.recommendations;

    moodLabel.textContent = a.label_display;
    moodRationale.textContent = a.rationale;
    const pct = Math.round(a.confidence * 100);
    confidenceBar.style.width = `${pct}%`;
    confidenceVal.textContent = `${pct}%`;

    renderVader(a.vader);
    renderList(listSongs, r.songs, "song");
    renderList(listMovies, r.movies, "movie");
    renderList(listSeries, r.series, "series");

    results.hidden = false;
  } catch (e) {
    statusEl.textContent = e.message || "Something went wrong.";
  } finally {
    setLoading(false);
  }
}

btn.addEventListener("click", analyze);
textEl.addEventListener("keydown", (ev) => {
  if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
    ev.preventDefault();
    analyze();
  }
});
