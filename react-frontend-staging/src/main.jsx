import { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const metrics = [
  ["Calls received", "42", "+8 from yesterday", "good"],
  ["Checklist completion", "78%", "9 calls awaiting items", "watch"],
  ["Compliance pass rate", "83%", "+3 pp against last week", "good"],
  ["Outstanding items", "17", "4 high-risk calls", "risk"],
];

const reviewItems = [
  ["Risk disclosure was understood", "Fail", "Customer acknowledgement not found", "risk"],
  ["Suitability assessment completed", "Pass", "Two supporting evidence segments", "good"],
  ["Product fees were explained", "Unclear", "Evidence confidence is low", "watch"],
  ["Customer consent recorded", "Pass", "Explicit confirmation at 03:44", "good"],
];

function Badge({ tone, children }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

function Dashboard({ setPage, stats }) {
  const liveMetrics = stats ? [
    ["Synthetic calls loaded", String(stats.calls_received), "SageMaker evaluation dataset", "good"],
    ["Checklist completion", `${stats.checklist_completion}%`, "Labelled evaluation items", "watch"],
    ["Compliance pass rate", `${stats.pass_rate}%`, "Across synthetic checklist labels", "good"],
    ["Outstanding items", String(stats.outstanding_items), "Fail or unclear outcomes", "risk"],
  ] : metrics;
  return <>
    <section className="heading"><div><h1>Compliance review command</h1><p>A live view of automated call reviews, checklist outcomes, and work requiring expert attention.</p></div><div className="actions"><button>Export management report</button><button className="primary" onClick={() => setPage("reviews")}>Open review queue</button></div></section>
    <section className="hero card"><div><span className="eyebrow">Priority signal · 4 high-risk calls</span><h2>17 review items need compliance attention</h2><p>Most conversations are progressing automatically. Focus human judgement on unclear evidence, failed checklist items, and calls nearing their review deadline.</p><div className="actions"><button className="primary" onClick={() => setPage("reviews")}>Review priority calls</button><button onClick={() => setPage("reports")}>See performance results</button></div></div><div className="hero-summary"><small>Today’s completion rate</small><strong>78%</strong><Badge tone="watch">In progress</Badge><hr/><span>Calls processed <b>42</b></span><span>Checklists complete <b>33</b></span><span>Escalations open <b>4</b></span></div></section>
    <section className="metrics">{liveMetrics.map(([label, value, detail, tone]) => <article className="metric card" key={label}><div>{label}<i className={tone}/></div><strong>{value}</strong><small className={tone}>{detail}</small></article>)}</section>
    <section className="two-col"><article className="card padded"><div className="panel-title"><div><h2>Review progress by business area</h2><p>Call-level checklist completion for the current review window</p></div><Badge tone="info">Today</Badge></div><Progress label="Retail banking" value={91} note="11/12" tone="good"/><Progress label="Wealth advisory" value={67} note="8/12" tone="watch"/><Progress label="Cards & lending" value={80} note="12/15" tone="info"/><Progress label="Service desk" value={100} note="2/2" tone="good"/></article><article className="card padded"><div className="panel-title"><div><h2>Priority review queue</h2><p>AI flagged calls with material compliance risk</p></div><button onClick={() => setPage("reviews")}>View all</button></div><QueueItem id="CC-2481" text="Risk disclosure acknowledgement not found" tone="risk" onClick={() => setPage("reviews")}/><QueueItem id="CC-2477" text="Identity verification evidence is unclear" tone="watch" onClick={() => setPage("reviews")}/><QueueItem id="CC-2469" text="Potentially unsuitable product explanation" tone="risk" onClick={() => setPage("reviews")}/></article></section>
  </>;
}

function Progress({ label, value, note, tone }) {
  return <div className="progress-row"><span>{label}</span><div className="progress"><i className={tone} style={{ width: `${value}%` }}/></div><b>{value}% · {note}</b></div>;
}

function QueueItem({ id, text, tone, onClick }) {
  return <button className="queue-item" onClick={onClick}><i className={tone}/><span><b>{id}</b><small>{text}</small></span><Badge tone={tone}>{tone === "risk" ? "High risk" : "Review"}</Badge></button>;
}

function UploadZone({ onUpload }) {
  const input = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState("");
  const useFile = (file) => { if (file) { setFileName(file.name); onUpload(file); } };
  return <div className={`upload-zone ${dragging ? "dragging" : ""}`} role="button" tabIndex="0" onClick={() => input.current?.click()} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); useFile(e.dataTransfer.files[0]); }}><div className="upload-icon">↑</div><div><b>Upload a new call recording</b><small className={fileName ? "uploaded" : ""}>{fileName ? `${fileName} · Ready to analyse` : "Drag and drop MP3, WAV, or M4A here, or choose a file"}</small></div><button type="button" onClick={(e) => { e.stopPropagation(); input.current?.click(); }}>Choose audio</button><input ref={input} type="file" accept="audio/mpeg,audio/wav,audio/x-wav,audio/mp4,.mp3,.wav,.m4a" onChange={(e) => useFile(e.target.files[0])}/></div>;
}

function Reviews() {
  const [selected, setSelected] = useState(0);
  const [decision, setDecision] = useState("");
  const [uploaded, setUploaded] = useState(false);
  return <><section className="heading"><div><h1>Human review queue</h1><p>Validate AI-generated checklist answers with supporting transcript evidence.</p></div><div className="actions"><button>Filter: priority first</button><button className="primary">Approve current review</button></div></section><UploadZone onUpload={() => setUploaded(true)}/>{uploaded && <p className="upload-note">Audio recorded for secure ingestion. The analysis job will appear in this queue when processing is complete.</p>}<section className="review-layout"><article className="card padded"><div className="call-head"><div><Badge tone="risk">High risk</Badge><h2>CC-2481 · Risk disclosure review</h2><p>Wealth advisory · 09:47 today · Investment suitability v3.2</p></div><Badge tone="watch">Due in 1h 42m</Badge></div><div className="audio"><button className="play">▶</button><span>▁▃▅▂▆▃▂▅▆▃▁▅▂▆▃▅▂▁▅▆▃▂▅▆▃▁▅▂▆</span><small>04:38</small></div><div className="evidence"><b>02:18 · Advisor:</b> I have explained that the investment value can fluctuate and you may not recover the amount invested.<br/><br/><b>02:31 · Customer:</b> <mark>Could you repeat what I might lose?</mark></div><div className="rule-heading"><h2>AI checklist assessment</h2><p>Select an item to inspect its evidence and change the outcome if required.</p></div>{reviewItems.map(([title, outcome, detail, tone], index) => <button className={`result ${selected === index ? "selected" : ""}`} onClick={() => setSelected(index)} key={title}><i className={tone}>{outcome === "Pass" ? "✓" : outcome === "Fail" ? "!" : "?"}</i><span><b>{title}</b><small>AI outcome: {outcome} · {detail}</small></span><Badge tone={tone}>{outcome}</Badge></button>)}</article><aside className="card padded"><Badge tone={reviewItems[selected][3]}>Selected checklist item</Badge><h3>{reviewItems[selected][0]}</h3><p className="subtle">The assistant’s evidence and recommendation are shown below for reviewer validation.</p><div className="evidence"><b>02:18</b> “I have explained that the investment value can fluctuate…”<br/><br/><b>02:31</b> “Could you repeat what I might lose?”</div><span className="timestamp">Jump to 02:18</span><div className="rule-heading"><h3>Reviewer decision</h3><div className="actions"><button className={decision === "Pass" ? "primary" : ""} onClick={() => setDecision("Pass")}>Pass</button><button className={decision === "Fail" ? "danger" : "danger ghost"} onClick={() => setDecision("Fail")}>Confirm fail</button><button className={decision === "Unclear" ? "primary" : ""} onClick={() => setDecision("Unclear")}>Unclear</button></div><textarea placeholder="Add an optional review note…"/></div><p className="subtle">Your decision, note, and cited timestamp become part of the final compliance report.</p></aside></section></>;
}

function Checklists() { return <><section className="heading"><div><h1>Checklist library</h1><p>Structured questions used to analyse conversations consistently.</p></div><button className="primary">Create checklist</button></section><section className="info-grid"><Info title="Investment suitability" text="9 questions for wealth-advisory conversations." items={["Suitability assessment completed", "Risk disclosure understood", "Fees and charges explained"]}/><Info title="Retail service quality" text="7 questions for everyday-banking calls." items={["Identity verification completed", "Consent recorded", "Service request confirmed"]}/><Info title="Cards & lending" text="8 questions for credit and payment-product conversations." items={["Product terms explained", "Affordability information captured", "Customer acknowledgement recorded"]}/></section></> }
function Reports() { return <><section className="heading"><div><h1>Reports & performance</h1><p>Management evidence for operations, model evaluation, and compliance outcomes.</p></div><button className="primary">Generate review report</button></section><section className="info-grid"><Info title="Operational performance" text="Review turnaround, backlog, and checklist completion by business area." items={["42 calls processed today", "78% checklist completion", "17 outstanding items"]}/><Info title="Model evaluation" text="Document answer correctness, evidence relevance, and reviewer agreement." items={["91.6% reviewer agreement", "Evidence citations required", "Unclear when evidence is insufficient"]}/><Info title="Compliance outcomes" text="Track pass, fail, and unresolved outcomes for management reporting." items={["83% pass rate today", "5% fail / escalated", "12% pending human review"]}/></section></> }
function Info({ title, text, items }) { return <article className="card info-card"><div className="symbol">☑</div><h2>{title}</h2><p>{text}</p><ul>{items.map(item => <li key={item}>{item}</li>)}</ul></article> }

function App() {
  const [page, setPage] = useState("dashboard");
  const [stats, setStats] = useState(null);
  useEffect(() => {
    fetch(`${import.meta.env.VITE_API_BASE ?? "http://localhost:8000"}/api/dashboard`)
      .then((response) => response.ok ? response.json() : null)
      .then(setStats)
      .catch(() => setStats(null));
  }, []);
  const views = { dashboard: <Dashboard setPage={setPage} stats={stats}/>, reviews: <Reviews/>, checklists: <Checklists/>, reports: <Reports/> };
  const nav = [["dashboard", "▦", "Management dashboard"], ["reviews", "☷", "Review queue"], ["checklists", "☑", "Checklist library"], ["reports", "▤", "Reports & results"]];
  return <div className="app"><aside><div className="brand"><span>✓</span>ClearCall</div><div className="workspace"><small>Workspace</small><b>Customer communications</b><em>Compliance operations</em></div><nav>{nav.map(([id, icon, label]) => <button className={page === id ? "active" : ""} onClick={() => setPage(id)} key={id}><i>{icon}</i>{label}{id === "reviews" && <b>17</b>}</button>)}</nav><div className="privacy"><b>▣ Evidence-led review</b><p>AI answers are paired with transcript evidence and timestamps so officers can validate each decision.</p></div></aside><main><header><span>Compliance review / <b>{nav.find(x => x[0] === page)[2]}</b></span><div><Badge tone="pass">● Processing normally</Badge><i className="avatar">CO</i></div></header><div className="page">{views[page]}</div></main></div>;
}

createRoot(document.getElementById("root")).render(<App/>);
