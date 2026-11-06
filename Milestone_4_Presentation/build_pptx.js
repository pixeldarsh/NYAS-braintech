// Builds Nyas_Braintech_Milestone_4.pptx. Usage: node build_pptx.js [output.pptx] [slideNumberForPreview]
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR + "/scripts/apply_theme.js");

const OUT = process.argv[2] || path.join(__dirname, "Nyas_Braintech_Milestone_4.pptx");
const ONLY = process.argv[3] ? parseInt(process.argv[3], 10) : null;
const MEDIA = path.join(__dirname, "media");

const THEME = {
  name: "Nyas Braintech", headFontFace: "Arial", bodyFontFace: "Arial",
  colors: {
    dk1: "12233F", lt1: "F7F6F1", dk2: "44546A", lt2: "E9EFF7",
    accent1: "1F5FAD", accent2: "B84A1C", accent3: "FBE9E0", accent4: "C9D6E6",
    accent5: "F2915F", accent6: "1F3556", hlink: "1F5FAD", folHlink: "44546A",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Nyas Braintech Milestone 4";
pres.author = "Nyas Braintech";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const INK = C.text1, MUTED = C.text2, PAPER = C.background1, TINT = C.background2;
const BLUE = C.accent1, RUST = C.accent2, BLUSH = C.accent3, ICE = C.accent4, GLOW = C.accent5, PANEL = C.accent6;

const placeholders = (eyebrow, title, footer) => [
  { placeholder: { options: { name: "eyebrow", type: "body", x: 0.6, y: 0.42, w: 12.13, h: 0.36, fontSize: 14, bold: true, color: eyebrow, margin: 0 }, text: "" } },
  { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.78, w: 12.13, h: 1.12, fontSize: 32, bold: true, color: title, margin: 0, valign: "top", align: "left" }, text: "" } },
  { placeholder: { options: { name: "footer", type: "body", x: 0.6, y: 6.86, w: 11.3, h: 0.34, fontSize: 11, color: footer, margin: 0 }, text: "" } },
];
pres.defineSlideMaster({ title: "LIGHT", background: { color: PAPER }, objects: placeholders(RUST, INK, MUTED), slideNumber: { x: 12.1, y: 6.86, w: 0.63, h: 0.34, fontSize: 11, color: MUTED, align: "right" } });
pres.defineSlideMaster({ title: "DARK", background: { color: INK }, objects: placeholders(GLOW, PAPER, ICE), slideNumber: { x: 12.1, y: 6.86, w: 0.63, h: 0.34, fontSize: 11, color: ICE, align: "right" } });

pres.defineSlideMaster({ title: "COVER", background: { color: INK }, objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.9, y: 2.05, w: 11.5, h: 1.1, fontSize: 48, bold: true, color: PAPER, margin: 0, valign: "middle", align: "left" }, text: "" } }] });
pres.defineSlideMaster({ title: "STATEMENT", background: { color: INK }, objects: [
  { placeholder: { options: { name: "eyebrow", type: "body", x: 0.9, y: 1.35, w: 11.5, h: 0.42, fontSize: 18, bold: true, color: GLOW, margin: 0 }, text: "" } },
  { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 1.85, w: 11.5, h: 3.9, fontSize: 38, bold: true, color: PAPER, margin: 0, valign: "top", align: "left" }, text: "" } },
], slideNumber: { x: 12.1, y: 6.86, w: 0.63, h: 0.34, fontSize: 11, color: ICE, align: "right" } });

let sectionName = null;
let counter = 0;
const notes = (id) => {
  const s = fs.readFileSync(path.join(__dirname, "slide_source", id + ".html"), "utf8");
  const m = s.match(/<aside>([\s\S]*?)<\/aside>/);
  return m ? m[1].trim() : "";
};

function section(title) {
  sectionName = title;
  if (!ONLY) pres.addSection({ title });
}

function make(id, master, eyebrow, title, footer) {
  counter += 1;
  if (ONLY && ONLY !== counter) return null;
  const s = pres.addSlide(ONLY ? { masterName: master } : { masterName: master, sectionTitle: sectionName });
  if (eyebrow) s.addText(eyebrow, { placeholder: "eyebrow" });
  if (title) s.addText(title, { placeholder: "title" });
  if (footer) s.addText(footer, { placeholder: "footer" });
  s.addNotes(notes(id));
  return s;
}

const text = (s, t, o) => s.addText(t, { isTextBox: true, margin: 0, valign: "top", fontSize: 15, color: INK, ...o });

// A filled card with a heading and body inside one shape
function card(s, x, y, w, h, heading, body, o = {}) {
  const fill = o.fill || TINT, ink = o.ink || INK;
  const runs = [];
  if (o.label) runs.push({ text: o.label, options: { fontSize: 14, bold: true, color: o.labelColor || BLUE, breakLine: true, paraSpaceAfter: 4 } });
  if (heading) runs.push({ text: heading, options: { fontSize: o.headSize || 20, bold: true, color: ink, breakLine: !!body, paraSpaceAfter: 8 } });
  if (body) runs.push({ text: body, options: { fontSize: o.bodySize || 15, color: o.bodyInk || ink, bold: !!o.bodyBold } });
  s.addText(runs, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x, y, w, h, fill: { color: fill }, margin: 12, valign: o.valign || "top", align: "left", objectName: heading || "Card" });
}

const arrow = (s, x, y, w = 0.32, h = 0.24) => s.addShape(pres.shapes.RIGHT_ARROW, { x, y, w, h, fill: { color: MUTED }, line: { color: MUTED, width: 0 }, objectName: "Arrow" });

function bullets(s, items, o) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: i < items.length - 1, paraSpaceAfter: 6 } })), { isTextBox: true, margin: 0, valign: "top", fontSize: 14, color: INK, ...o });
}

const cell = (t, o = {}) => ({ text: t, options: { valign: "middle", fontSize: 14, color: INK, ...o } });

function dataTable(s, rows, opt) {
  const body = rows.map((r, ri) => r.map((c, ci) => {
    const hdr = ri === 0, hi = opt.highlight === ri;
    return cell(c, { bold: hdr || hi, color: hdr ? PAPER : INK, fill: hdr ? { color: INK } : hi ? { color: TINT } : { color: PAPER }, align: ci === 0 ? "left" : "center", fontSize: opt.fontSize || 14 });
  }));
  s.addTable(body, { x: opt.x, y: opt.y, w: opt.w, colW: opt.colW, rowH: opt.rowH || 0.42, border: { type: "solid", pt: 0.75, color: "C9CCD1" }, margin: 0.06, objectName: opt.name });
}

const axis = {
  catAxisLabelColor: "44546A", valAxisLabelColor: "44546A", catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", dataLabelFontSize: 13,
  dataLabelColor: "12233F", valGridLine: { color: "DEDDD6", size: 0.75 }, catGridLine: { style: "none" }, showValue: true,
};

// 1
section("The problem and who it affects");
let s = make("cover", "COVER", null, "When can we trust the tools?");
if (s) {
  text(s, "Junior Academy Innovation Challenge · Milestone 4", { x: 0.9, y: 0.75, w: 11.5, h: 0.4, fontSize: 16, bold: true, color: GLOW });
  text(s, "Testing the reliability of EEG sleep tracking and focused-ultrasound simulation for future sleep and anxiety research", { x: 0.9, y: 3.35, w: 10, h: 1.0, fontSize: 22, color: ICE });
  text(s, [{ text: "Nyas Braintech", options: { bold: true, fontSize: 16, color: PAPER, breakLine: true } }, { text: "[Student names] · Mentor: [Mentor name] · Team [number]", options: { fontSize: 14, color: ICE } }], { x: 0.9, y: 5.25, w: 11, h: 0.7 });
  text(s, "Software-only research. No ultrasound was given to any person or animal.", { x: 0.9, y: 6.5, w: 11, h: 0.32, fontSize: 12, color: ICE });
}

// 2
s = make("problem", "LIGHT", "Problem", "Promising brain tools can be wrong and still look convincing");
if (s) {
  const y = 2.05, h = 4.3, w = 3.85, g = 0.29;
  s.addText([
    { text: "The need", options: { fontSize: 22, bold: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Sleep problems and anxiety affect daily life, health and learning.", options: { fontSize: 19, breakLine: true, paraSpaceAfter: 14 } },
    { text: "[Local statistic and source: sleep or anxiety in our community]", options: { fontSize: 19, bold: true, color: RUST } },
  ], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: 0.6, y, w, h, fill: { color: TINT }, margin: 12, valign: "top", color: INK, objectName: "The need" });
  card(s, 0.6 + w + g, y, w, h, "The opportunity", "Researchers are studying low-intensity focused ultrasound (LIFU), a non-invasive way to target brain regions, and EEG can track sleep.", { bodySize: 19, headSize: 22 });
  card(s, 0.6 + 2 * (w + g), y, w, h, "The gap", "A sleep model can report high confidence when it is wrong. A simulation can give a precise number that depends on uncertain assumptions.", { fill: INK, ink: PAPER, bodySize: 19, headSize: 22 });
}

// 3
s = make("community", "LIGHT", "Who this affects", "Researchers benefit first; communities could benefit later");
if (s) {
  const y = 2.05, h = 2.75, w = 3.55, g = 0.74;
  [["Now", "Researchers", "See when a model or simulation should not be trusted."],
   ["Next", "Future studies", "Could be better designed before anyone is exposed to ultrasound."],
   ["Possibly, much later", "People and families", "Could gain from better evidence, if future research succeeds."]].forEach((c, i) => {
    const x = 0.6 + i * (w + g);
    card(s, x, y, w, h, c[1], c[2], { label: c[0], bodySize: 18, headSize: 22 });
    if (i < 2) arrow(s, x + w + 0.17, y + h / 2 - 0.14, 0.4, 0.28);
  });
  s.addText("[Our community: who we focused on, and what we heard from a researcher, clinician or community member]", { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: 0.6, y: 5.1, w: 12.13, h: 1.25, fill: { color: BLUSH }, margin: 12, fontSize: 16, bold: true, color: RUST, valign: "middle", objectName: "Community placeholder" });
}

// 4
section("Background and hypothesis");
s = make("eeg", "LIGHT", "Background: EEG and sleep", "EEG shows electrical patterns that change across sleep", "Sources: Kemp et al., 2000; Goldberger et al., 2000; AASM scoring manual (Berry et al.)");
if (s) {
  const stages = [["Wake", "Faster activity; alpha waves with eyes closed"], ["N1", "Brief, light transition into sleep; theta waves"], ["N2", "Sleep spindles and K-complexes"], ["N3", "Deep sleep; large, slow delta waves"], ["REM", "Dreaming stage; EEG can look like light sleep"]];
  const w = 2.27, g = 0.195;
  stages.forEach((c, i) => card(s, 0.6 + i * (w + g), 2.05, w, 2.2, c[0], c[1], { fill: c[0] === "N1" ? BLUSH : TINT, headSize: 24, bodySize: 15 }));
  text(s, [{ text: "Why it is hard", options: { bold: true, fontSize: 18, breakLine: true, paraSpaceAfter: 6 } }, { text: "Stages blend into each other, N1 is short and rare, and brain signals differ from person to person.", options: { fontSize: 15 } }], { x: 0.6, y: 4.6, w: 5.8, h: 1.6 });
  text(s, [{ text: "Our data", options: { bold: true, fontSize: 18, breakLine: true, paraSpaceAfter: 6 } }, { text: "Public Sleep-EDF recordings, with each 30-second epoch labelled by a human scorer.", options: { fontSize: 15 } }], { x: 6.93, y: 4.6, w: 5.8, h: 1.6 });
}

// 5
s = make("lifu", "LIGHT", "Background: focused ultrasound", "The skull makes it hard to know what reaches the target", "Sources: ITRUSST practical guide (arXiv 2407.07646); Angla et al. (arXiv 2207.12026); Aubry et al., 2022");
if (s) {
  const items = [["What LIFU is", "Low-intensity sound waves focused on a small brain region from outside the head. It is still being studied."],
    ["The challenge", "The skull reflects, absorbs and bends sound, and its properties vary. Published guidance describes skull absorption as considerably uncertain."],
    ["Why simulate", "Simulation lets us test “what if” questions with no exposure to anyone, but its answers must be checked."]];
  items.forEach((c, i) => text(s, [{ text: c[0], options: { bold: true, fontSize: 18, breakLine: true, paraSpaceAfter: 4 } }, { text: c[1], options: { fontSize: 15 } }], { x: 0.6, y: 2.05 + i * 1.5, w: 6.7, h: 1.4 }));
  s.addText([
    { text: "Our simulated setup", options: { fontSize: 15, bold: true, color: GLOW, breakLine: true, paraSpaceAfter: 10 } },
    { text: "500 kHz focused source through skin, a 3-layer skull and brain", options: { fontSize: 24, bold: true, color: PAPER, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Tissue values follow a published simulation benchmark, built here in 2D.", options: { fontSize: 14, color: ICE } },
  ], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: 7.7, y: 2.05, w: 5.03, h: 4.3, fill: { color: INK }, margin: 12, valign: "middle", objectName: "Simulated setup" });
}

// 6
s = make("hypothesis", "STATEMENT", "Hypothesis");
if (s) {
  s.addText("A framework that measures uncertainty will reveal when EEG sleep models and ultrasound simulations become unreliable, and which factors matter most, better than headline accuracy alone.", { placeholder: "title" });
}

// 7
section("How we tested it");
s = make("methods", "LIGHT", "Methods", "Two tracks, one reliability question", "Tools: Python, scikit-learn, k-Wave (Treeby and Cox, 2010). Software only.");
if (s) {
  const row = (y, label, color, fill, steps) => {
    text(s, label, { x: 0.6, y, w: 6, h: 0.32, fontSize: 15, bold: true, color });
    const w = 2.45, g = 0.777;
    steps.forEach((t, i) => {
      const x = 0.6 + i * (w + g);
      s.addText(t, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x, y: y + 0.36, w, h: 0.82, fill: { color: fill }, margin: 12, fontSize: 14, color: INK, valign: "middle", objectName: t });
      if (i < 3) arrow(s, x + w + 0.23, y + 0.65);
    });
  };
  row(1.55, "EEG track", BLUE, TINT, ["Sleep-EDF: 1 EEG channel", "10 frequency-band features", "Random forest vs simple baselines", "Confidence and noise tests"]);
  row(3.0, "Acoustic track", RUST, BLUSH, ["2D layered virtual head", "k-Wave beam simulation", "Checks against exact solutions", "Varying tissue properties"]);
  s.addText([{ text: "Fair testing: ", options: { bold: true } }, { text: "each person held out in turn, then a frozen model tested once on 3 new people" }], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: 0.6, y: 4.6, w: 5.92, h: 1.75, fill: { color: INK }, margin: 12, fontSize: 16, color: PAPER, valign: "middle", objectName: "Fair testing" });
  s.addText([{ text: "Joining the tracks: ", options: { bold: true } }, { text: "a software-only test of how often a hypothetical trigger would be mistimed" }], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: 6.81, y: 4.6, w: 5.92, h: 1.75, fill: { color: INK }, margin: 12, fontSize: 16, color: PAPER, valign: "middle", objectName: "Joining the tracks" });
}

// 8
section("What we found");
s = make("classifier", "LIGHT", "Results: blind test on 3 new people", "It held up, but accuracy alone misleads", "5 development and 3 blind participants, same dataset. Cross-validation: 76.5% accuracy, macro-F1 0.55.");
if (s) {
  // Row percentages from the blind-test confusion matrix (rows = true stage)
  const cm = [[5059, 555, 279, 24, 365], [37, 69, 42, 0, 41], [10, 14, 944, 21, 38], [9, 0, 122, 302, 0], [13, 33, 72, 0, 250]];
  const st = ["Wake", "N1", "N2", "N3", "REM"];
  const mix = (t) => {
    const a = [247, 251, 255], b = [8, 69, 148];
    return a.map((v, i) => Math.round(v + (b[i] - v) * t).toString(16).padStart(2, "0")).join("").toUpperCase();
  };
  const rows = [[cell("True \\ Predicted", { fontSize: 11, color: MUTED, bold: true, align: "center" }), ...st.map((n) => cell(n, { bold: true, align: "center", fontSize: 13, color: MUTED }))]];
  cm.forEach((r, i) => {
    const sum = r.reduce((a, b) => a + b, 0);
    rows.push([cell(st[i], { bold: true, align: "center", fontSize: 13, color: MUTED }), ...r.map((v) => {
      const f = v / sum;
      return cell(`${Math.round(f * 100)}%`, { align: "center", fontSize: 14, bold: f > 0.55, color: f > 0.55 ? "FFFFFF" : INK, fill: { color: mix(f) } });
    })]);
  });
  s.addTable(rows, { x: 0.6, y: 1.75, w: 5.7, colW: [1.2, 0.9, 0.9, 0.9, 0.9, 0.9], rowH: 0.7, border: { type: "solid", pt: 1, color: "F7F6F1" }, objectName: "Blind-test confusion matrix" });
  dataTable(s, [["Blind test", "Accuracy", "Macro-F1"], ["Our random forest", "79.8%", "0.61"], ["Logistic regression", "75.5%", "0.60"], ["Always guess Wake", "75.7%", "0.17"]], { x: 6.7, y: 1.75, w: 6.03, colW: [3.13, 1.45, 1.45], highlight: 1, name: "Blind test results" });
  s.addText([{ text: "Wake vs sleep is much easier: ", options: { bold: true } }, { text: "about 92% accuracy near sleep, using a cut-off taken from development data." }], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: 6.7, y: 3.75, w: 6.03, h: 1.25, fill: { color: TINT }, margin: 12, fontSize: 15, color: INK, valign: "middle", objectName: "Wake vs sleep" });
  text(s, [{ text: "N1 is the weak spot: ", options: { bold: true } }, { text: "only 37% of N1 epochs were identified." }], { x: 6.7, y: 5.3, w: 6.03, h: 0.7, fontSize: 15 });
}

// 9
s = make("confidence", "LIGHT", "Results: when confidence fails", "With noisier EEG, the model grew more confident and less capable", "Random forest, leave-one-person-out on 5 participants, mean of 3 noise seeds. White noise is a simplified stress test.");
if (s) {
  const rob = fs.readFileSync(path.join(__dirname, "..", "robustness_comparison.csv"), "utf8").trim().split("\n").map((l) => l.split(",")).filter((r) => r[0] === "RF (all)");
  const labels = rob.map((r) => r[1]);
  s.addChart(pres.charts.LINE, [
    { name: "Mean confidence", labels, values: rob.map((r) => +(+r[4]).toFixed(2)) },
    { name: "Macro-F1", labels, values: rob.map((r) => +(+r[3]).toFixed(2)) },
  ], { x: 0.6, y: 1.95, w: 7.6, h: 4.65, ...axis, chartColors: ["EB6834", "2A78D6"], lineSize: 3, lineDataSymbolSize: 9, dataLabelPosition: "t", dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "b", legendFontSize: 13, legendFontFace: "+mn-lt", legendColor: "12233F", showCatAxisTitle: true, catAxisTitle: "More noise added to the EEG →", catAxisTitleFontSize: 12, catAxisTitleColor: "44546A", catAxisTitleFontFace: "+mn-lt", altText: "Line chart. As noise increases from clean to 0 dB, mean confidence rises from 0.79 to 0.85 while macro-F1 falls from 0.56 to 0.21.", objectName: "Noise chart" });
  text(s, "0.86 → 0.66", { x: 8.6, y: 2.1, w: 4.13, h: 0.85, fontSize: 40, bold: true, color: RUST });
  text(s, "Its ability to flag its own errors (AUC), from clean EEG to 0 dB", { x: 8.6, y: 2.98, w: 4.13, h: 0.85, fontSize: 15 });
  s.addText([
    { text: "A confident answer is not always a good answer.", options: { fontSize: 16, bold: true, color: PAPER, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Check signal quality first.", options: { fontSize: 14, color: ICE } },
  ], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: 8.6, y: 4.2, w: 4.13, h: 2.15, fill: { color: INK }, margin: 12, valign: "middle", objectName: "Key message" });
}

// 10
s = make("beam", "LIGHT", "Results: what reaches the target", "About a third of the focal pressure remains", "2D simulation, uncalibrated, 500 kHz. Pressure is relative to the same source in water. Not a safety or dose value.");
if (s) {
  const w = 10.6, h = w * 320 / 1160, x = (13.33 - w) / 2, y = 2.05;
  text(s, "In water: focal pressure 1.00", { x, y: 1.62, w: w / 2, h: 0.36, fontSize: 15, bold: true });
  text(s, "Through skin, skull and brain: 0.34", { x: x + w / 2, y: 1.62, w: w / 2, h: 0.36, fontSize: 15, bold: true });
  const cover = "image/png;base64," + fs.readFileSync(path.join(MEDIA, "beam_still.png")).toString("base64");
  s.addMedia({ type: "video", path: path.join(MEDIA, "beam_animation.mp4"), x, y, w, h, cover, objectName: "Beam animation" });
  text(s, [{ text: "Treating the skull as one solid layer ", options: { bold: true } }, { text: "overestimates focal pressure by about 50% (0.51 vs 0.34)." }], { x: 0.6, y: 5.3, w: 5.9, h: 1.05, fontSize: 15 });
  text(s, [{ text: "Absorption matters: ", options: { bold: true } }, { text: "without it, the 3-layer model gives 0.54." }], { x: 6.83, y: 5.3, w: 5.9, h: 1.05, fontSize: 15 });
}

// 11
s = make("drivers", "LIGHT", "Results: uncertainty and checks", "Skull absorption and thickness matter most", "19 of 32 samples finished. Unfinished samples had more absorption, so the range uses a fit to all 32 inputs (raw: 0.25–0.58).");
if (s) {
  // Spearman rank correlations from lifu_mc.csv
  const names = ["Brain speed", "Bone density", "Bone speed", "Skull thickness", "Bone absorption"];
  s.addChart(pres.charts.BAR, [{ name: "Rank correlation with focal pressure", labels: names, values: [0.22, -0.28, -0.28, -0.57, -0.84] }], { x: 0.6, y: 1.7, w: 6.9, h: 4.9, barDir: "bar", ...axis, chartColors: ["2A78D6"], dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.00;-0.00", valAxisMinVal: -1, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.5, valAxisLabelFormatCode: "0.0", catAxisLabelPos: "low", showLegend: false, showTitle: true, title: "Rank correlation with focal pressure", titleFontSize: 14, titleColor: "12233F", titleFontFace: "+mn-lt", altText: "Bone absorption minus 0.84, skull thickness minus 0.57, bone speed minus 0.28, bone density minus 0.28, brain speed plus 0.22.", objectName: "Drivers chart" });
  text(s, "0.22–0.52", { x: 7.9, y: 1.75, w: 4.83, h: 0.8, fontSize: 40, bold: true, color: RUST });
  text(s, "Estimated range of focal pressure relative to water (5–95%), across assumed tissue ranges", { x: 7.9, y: 2.58, w: 4.83, h: 0.8, fontSize: 14 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.9, y: 3.6, w: 4.83, h: 2.85, rectRadius: 0.1, fill: { color: TINT }, objectName: "Checks background" });
  text(s, "How we checked our code", { x: 8.1, y: 3.78, w: 4.45, h: 0.4, fontSize: 18, bold: true });
  bullets(s, ["Layer transmission: within 0.5% of the exact solution", "2D focus: matched the analytic field", "Those checks caught a real bug in our code"], { x: 8.1, y: 4.25, w: 4.45, h: 2.05 });
}

// 12
s = make("gating", "LIGHT", "Results: bringing EEG and LIFU together in software", "Would a sleep-triggered system fire at the right time?", "Simulated timing only. N2 and N3 are examples; this work does not support targeting any sleep stage. Nothing was delivered to anyone.");
if (s) {
  dataTable(s, [["Hypothetical trigger rule (blind test)", "Precision", "Recall", "Fired in Wake"], ["N2, no filter", "0.65", "0.92", "19.1%"], ["N2, confidence ≥ 0.7 and 3 epochs in a row", "0.94", "0.55", "0.2%"], ["N3, no filter", "0.87", "0.70", "6.9%"]], { x: 0.6, y: 1.95, w: 12.13, colW: [6.43, 1.9, 1.7, 2.1], highlight: 2, rowH: 0.52, fontSize: 15, name: "Trigger timing" });
  [["Filters help, at a cost", "Better timing means missing more of the target stage."], ["People differ", "N3 precision was only 0.27 in our 5 development participants."], ["Noise removes triggers", "N2 recall fell from 0.82 to 0.22 at 5 dB (development data)."]].forEach((c, i) => {
    card(s, 0.6 + i * 4.14, 4.4, 3.85, 1.95, c[0], c[1], { headSize: 18, bodySize: 15 });
  });
}

// 13
section("Our solution, its limits and what comes next");
s = make("solution", "DARK", "Our solution and its impact", "A reliability-testing framework that shows when to trust these tools, before any human use", "It runs on existing technology today. Any clinical use would need far more validation, plus safety and ethics review.");
if (s) {
  const w = 3.85, g = 0.29;
  [["What is new", "It tests classifier confidence and acoustic uncertainty together, including how often timing would go wrong."],
   ["Positive impact", "Helps researchers find weak points earlier, with no exposure to ultrasound. It uses software and public data, so it can scale at low cost."],
   ["Risks we must manage", "Simulated numbers could be misread as safe doses, and small datasets may not represent everyone."]].forEach((c, i) => card(s, 0.6 + i * (w + g), 2.2, w, 4.15, c[0], c[1], { fill: PANEL, ink: PAPER, bodySize: 19, headSize: 22 }));
}

// 14
s = make("limits", "LIGHT", "Limits and next steps", "What we know, and what we still need to test");
if (s) {
  [[0.6, BLUSH, "Limits", ["8 people from one dataset", "2D head model, uncalibrated pressures", "Finest grid and 3D benchmark not finished", "Not experimentally or clinically validated"]],
   [6.81, TINT, "Next steps", ["Test about 20 more participants", "Add signal-quality screening and real artefacts", "Finish the published benchmark comparison", "Vary skull shape and add community input"]]].forEach(([x, fill, head, items]) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: 5.92, h: 2.95, rectRadius: 0.1, fill: { color: fill }, objectName: head + " background" });
    text(s, head, { x: x + 0.24, y: 1.92, w: 5.44, h: 0.42, fontSize: 20, bold: true });
    bullets(s, items, { x: x + 0.24, y: 2.42, w: 5.44, h: 2.15, fontSize: 17 });
  });
  s.addText("Before we ask whether a tool can help people, we need to know when it can’t be trusted.", { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x: 0.6, y: 4.98, w: 12.13, h: 1.5, fill: { color: INK }, margin: 12, fontSize: 24, bold: true, color: PAPER, valign: "middle", objectName: "Closing message" });
}

// 15
s = make("refs", "LIGHT", null, "References");
if (s) {
  const refs = [
    "Aubry J-F, et al. Benchmark problems for transcranial ultrasound simulation: intercomparison of compressional wave models. J Acoust Soc Am. 2022;152(2):1003–1019.",
    "Kemp B, et al. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE Trans Biomed Eng. 2000;47(9):1185–1194.",
    "Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation. 2000;101(23):e215–e220.",
    "Treeby BE, Cox BT. k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields. J Biomed Opt. 2010;15(2):021314.",
    "ITRUSST Consortium. A practical guide to transcranial ultrasonic stimulation. arXiv:2407.07646.",
    "Angla C, et al. Transcranial ultrasound simulations: a review. arXiv:2207.12026.",
    "Berry RB, et al. The AASM Manual for the Scoring of Sleep and Associated Events. American Academy of Sleep Medicine.",
    "[Our local sleep or anxiety statistics source]",
  ];
  s.addText(refs.map((t, i) => ({ text: t, options: { bullet: { type: "number", indent: 24 }, breakLine: i < refs.length - 1, paraSpaceAfter: 7 } })), { isTextBox: true, x: 0.6, y: 1.55, w: 12.13, h: 5.1, margin: 0, fontSize: 14, color: INK, valign: "top" });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
