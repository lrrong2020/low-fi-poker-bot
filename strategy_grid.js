// All probabilities are in the selected public node's legal action order.
function renderRange({container, node, selected, policy, centers, onSelect, colors}) {
  if (!document.getElementById("range-grid-styles")) {
    const styles = document.createElement("style");
    styles.id = "range-grid-styles";
    styles.textContent = `
      .range-surface { container-type:inline-size; }
      .range-matrix,.range-buckets { display:grid; }
      .range-matrix { grid-template-columns:repeat(13,minmax(0,1fr)); gap:2px; }
      .range-buckets { grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px 11px; }
      .range-surface button {
        -webkit-appearance:none; appearance:none; box-sizing:border-box;
        font-family:inherit; color:#172a25; border:0; cursor:pointer;
        min-width:0; position:relative; overflow:hidden; text-align:center;
        font-variant-numeric:tabular-nums; touch-action:manipulation;
      }
      .range-matrix button {
        height:clamp(25px,5.45cqw,47px); border-radius:3px;
        padding:0 0 3px; font-size:clamp(9px,1.85cqw,14px); font-weight:600;
        letter-spacing:-.035em; line-height:1;
        background:linear-gradient(#fffc 0%,#fff9 100%),var(--mix);
      }
      .range-matrix button::after {
        content:""; position:absolute; left:0; right:0; bottom:0;
        height:clamp(4px,.85cqw,7px); background:var(--mix);
      }
      .range-surface button[aria-pressed=true]::before,.range-surface button:focus-visible::before {
        content:""; pointer-events:none; position:absolute; inset:0; z-index:1;
        border:2px solid #172a25; border-radius:inherit;
      }
      .range-surface button:focus-visible { outline:2px solid #0969a4; outline-offset:-4px; }
      @media (hover:hover) {
        .range-surface button:hover { filter:brightness(.96); }
      }
      .range-buckets button {
        display:grid; grid-template-columns:1fr auto; align-content:center; gap:5px 8px;
        padding:9px 11px; background:#f3f5ef; border-radius:4px;
        font-size:13px; min-height:44px; line-height:1.15;
      }
      .range-bucket-name { font-weight:600; text-align:left; }
      .range-centroid { font-size:12px; color:#43574b; }
      .range-bar { grid-column:1 / -1; height:7px; border-radius:1px; background:var(--mix); }
      @container (max-width:400px) {
        .range-matrix { gap:1px; }
        .range-matrix button { border-radius:2px; }
        .range-buckets { gap:5px; }
        .range-buckets button { padding:8px; }
      }
    `;
    document.head.append(styles);
  }

  container.classList.add("range-surface");
  const preflop = node.street === 0;
  const ranks = "23456789TJQKA";
  const actionNames = {
    fold: "Fold", call: node.toCall ? "Call" : "Check",
    half: "Half pot", pot: "Pot", allin: "All-in"
  };
  const grid = document.createElement("div");
  grid.className = preflop ? "range-matrix" : "range-buckets";
  grid.setAttribute("role", "group");
  grid.setAttribute("aria-label", preflop ? "Preflop hand classes" : "Postflop equity buckets");
  const entries = [];
  if (preflop) {
    for (let row = 12; row >= 0; row--) {
      for (let column = 12; column >= 0; column--) {
        const suffix = row === column ? "" : row > column ? "s" : "o";
        const label = ranks[Math.max(row, column)] + ranks[Math.min(row, column)] + suffix;
        entries.push({bucket:13 * row + column, label});
      }
    }
  } else {
    centers.forEach((center, bucket) => {
      entries.push({bucket, label:`Bucket ${bucket + 1}`, center});
    });
  }

  function select(button) {
    const wasFocused = document.activeElement === button;
    onSelect(Number(button.dataset.bucket));
    // The parent may rebuild the range to reflect the new selection.
    if (wasFocused) {
      container.querySelector(`button[data-bucket="${button.dataset.bucket}"]`)?.focus({preventScroll:true});
    }
  }

  entries.forEach(({bucket, label, center}, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.bucket = bucket;
    button.tabIndex = bucket === selected || (selected == null && index === 0) ? 0 : -1;
    button.setAttribute("aria-pressed", String(bucket === selected));
    const probabilities = policy(bucket);
    let mix = "repeating-linear-gradient(135deg,#e0e3dc 0 4px,#c9cfc7 4px 7px)";
    let summary = "No accumulated average strategy";
    if (probabilities) {
      const stops = [];
      const parts = [];
      let accumulated = 0;
      probabilities.forEach((p, i) => {
        const start = accumulated * 100;
        accumulated += p;
        const color = colors[node.actions[i]];
        if (p > 0) stops.push(`${color} ${start}%`, `${color} ${accumulated * 100}%`);
        parts.push(`${actionNames[node.actions[i]]} ${(p * 100).toFixed(1)}%`);
      });
      mix = `linear-gradient(to right,${stops.join(",")})`;
      summary = parts.join(", ");
    }
    button.style.setProperty("--mix", mix);
    let description = label;
    if (preflop) {
      button.textContent = label;
    } else {
      const name = document.createElement("span");
      name.className = "range-bucket-name";
      name.textContent = label;
      const equity = document.createElement("span");
      equity.className = "range-centroid";
      equity.textContent = `${(center * 100).toFixed(1)}%`;
      description += `, equity centroid ${equity.textContent}`;
      const bar = document.createElement("span");
      bar.className = "range-bar";
      button.append(name, equity, bar);
    }
    button.title = `${description} · ${summary}`;
    button.setAttribute("aria-label", `${description}. ${summary}`);
    grid.append(button);
  });
  grid.addEventListener("click", event => {
    const button = event.target.closest("button");
    if (button) select(button);
  });
  grid.addEventListener("keydown", event => {
    const button = event.target.closest("button");
    if (!button) return;
    const buttons = Array.from(grid.children);
    const current = buttons.indexOf(button);
    const width = preflop ? 13 : 2;
    const keys = {
      ArrowRight: current + 1, ArrowLeft: current - 1,
      ArrowDown: current + width, ArrowUp: current - width,
      Home: 0, End: buttons.length - 1
    };
    if (!Object.hasOwn(keys, event.key)) return;
    event.preventDefault();
    const next = buttons[Math.max(0, Math.min(buttons.length - 1, keys[event.key]))];
    next.focus();
    select(next);
  });
  container.replaceChildren(grid);
}
