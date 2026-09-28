const pptxgen = require("pptxgenjs");

const NAVY = "1E2761";
const BLACK = "000000";
const WHITE = "FFFFFF";
const GRAY = "444444";
const LIGHT_TINT = "EEF1F8";
const FONT = "Arial";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5 in
const PAGE_W = 13.33;
const MARGIN = 0.7;
const CONTENT_W = PAGE_W - MARGIN * 2;

pres.defineSlideMaster({
  title: "PLAIN",
  background: { color: WHITE },
});

function addHeader(slide, title, kicker) {
  if (kicker) {
    slide.addText(kicker.toUpperCase(), {
      x: MARGIN, y: 0.35, w: CONTENT_W, h: 0.35,
      fontFace: FONT, fontSize: 14, color: NAVY, bold: true, charSpacing: 2,
      margin: 0,
    });
  }
  slide.addText(title, {
    x: MARGIN, y: kicker ? 0.68 : 0.5, w: CONTENT_W, h: 0.8,
    fontFace: FONT, fontSize: 34, color: NAVY, bold: true,
    margin: 0,
  });
}

function numberedCircle(slide, x, y, num, size = 0.55) {
  slide.addShape("ellipse", {
    x, y, w: size, h: size,
    fill: { color: NAVY }, line: { type: "none" },
  });
  slide.addText(String(num), {
    x, y, w: size, h: size,
    fontFace: FONT, fontSize: 22, color: WHITE, bold: true,
    align: "center", valign: "middle", margin: 0,
  });
}

// ---------- Slide 1: Title ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  slide.addText("Singularity-Aware Physics-Informed\nNeural Networks for Real-Time\nInverse Kinematics", {
    x: MARGIN, y: 1.9, w: CONTENT_W, h: 2.4,
    fontFace: FONT, fontSize: 38, color: NAVY, bold: true,
    align: "center", valign: "middle", margin: 0, lineSpacing: 46,
  });
  slide.addText("Safe, Real-Time Motion Control for Robot Arms Using AI", {
    x: MARGIN, y: 4.25, w: CONTENT_W, h: 0.6,
    fontFace: FONT, fontSize: 22, color: GRAY, italic: true,
    align: "center", margin: 0,
  });
  slide.addShape("line", {
    x: PAGE_W / 2 - 1.2, y: 5.15, w: 2.4, h: 0,
    line: { color: NAVY, width: 1.5 },
  });
  slide.addText("First Project Review", {
    x: MARGIN, y: 5.35, w: CONTENT_W, h: 0.45,
    fontFace: FONT, fontSize: 20, color: BLACK, bold: true,
    align: "center", margin: 0,
  });
  slide.addText("Name: __________________     Roll No: __________     Guide: __________________", {
    x: MARGIN, y: 5.85, w: CONTENT_W, h: 0.45,
    fontFace: FONT, fontSize: 16, color: GRAY,
    align: "center", margin: 0,
  });
}

// ---------- Slide 2: Introduction ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "Introduction", "Section 1");
  const pts = [
    "Robot arms are used in factories, hospitals, warehouses, and space missions.",
    "To move the arm's hand to a target point, we must find the correct angle for every joint.",
    "This calculation is called Inverse Kinematics (IK).",
    "Modern arms have 7 joints — more than the minimum needed — which makes this calculation harder.",
  ];
  slide.addText(
    pts.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < pts.length - 1, paraSpaceAfter: 18 } })),
    { x: MARGIN, y: 1.75, w: CONTENT_W, h: 3.6, fontFace: FONT, fontSize: 24, color: BLACK, margin: 0, lineSpacing: 32 }
  );

  // simple flow visual: Target -> Joint Angles
  const by = 5.35, bw = 3.6, bh = 1.0;
  slide.addShape("roundRect", { x: MARGIN, y: by, w: bw, h: bh, rectRadius: 0.08, fill: { color: LIGHT_TINT }, line: { color: NAVY, width: 1 } });
  slide.addText("Target Point\n(a location in space)", { x: MARGIN, y: by, w: bw, h: bh, fontFace: FONT, fontSize: 18, color: NAVY, bold: true, align: "center", valign: "middle", margin: 0 });

  slide.addShape("rightArrow", { x: MARGIN + bw + 0.15, y: by + bh / 2 - 0.2, w: 0.9, h: 0.4, fill: { color: NAVY }, line: { type: "none" } });

  const bx2 = MARGIN + bw + 0.15 + 0.9 + 0.15;
  slide.addShape("roundRect", { x: bx2, y: by, w: bw, h: bh, rectRadius: 0.08, fill: { color: LIGHT_TINT }, line: { color: NAVY, width: 1 } });
  slide.addText("Joint Angles\n(7 angles, one per joint)", { x: bx2, y: by, w: bw, h: bh, fontFace: FONT, fontSize: 18, color: NAVY, bold: true, align: "center", valign: "middle", margin: 0 });

  slide.addText("Inverse Kinematics = solving this direction", { x: bx2 + bw + 0.3, y: by + 0.25, w: 3.0, h: 0.5, fontFace: FONT, fontSize: 14, italic: true, color: GRAY, margin: 0 });
}

// ---------- Slide 3: Key Terms ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "Key Terms, Explained Simply", "Section 1");
  const terms = [
    ["Forward Kinematics", "Know the joint angles → calculate where the hand ends up. (The easy direction.)"],
    ["Inverse Kinematics", "Know where we want the hand → find the joint angles. (The hard direction.)"],
    ["Redundant Arm", "An arm with extra joints — many different angle combinations reach the same point."],
    ["Singularity", "A risky arm position where the controller can suddenly demand impossibly fast joint movement."],
  ];
  let y = 1.65;
  const rowH = 1.3;
  terms.forEach((t, i) => {
    numberedCircle(slide, MARGIN, y + 0.05, i + 1, 0.5);
    slide.addText(t[0], { x: MARGIN + 0.75, y: y - 0.02, w: CONTENT_W - 0.75, h: 0.4, fontFace: FONT, fontSize: 22, color: NAVY, bold: true, margin: 0 });
    slide.addText(t[1], { x: MARGIN + 0.75, y: y + 0.42, w: CONTENT_W - 0.75, h: 0.8, fontFace: FONT, fontSize: 16, color: BLACK, margin: 0, lineSpacing: 21 });
    y += rowH;
  });
}

// ---------- Slide 4: Motivation ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "Motivation", "Section 2");
  const pts = [
    "Classical methods can freeze, or move dangerously fast, near singularities.",
    "This risks damaging expensive robots or harming people working nearby.",
    "Extra joints mean many solutions exist — plain AI models often blend them into nonsense.",
    "A fast, self-checking AI solver could let robots react safely in real time.",
  ];
  slide.addText(
    pts.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < pts.length - 1, paraSpaceAfter: 20 } })),
    { x: MARGIN, y: 1.8, w: CONTENT_W, h: 4.8, fontFace: FONT, fontSize: 24, color: BLACK, margin: 0, lineSpacing: 32 }
  );
}

// ---------- Slide 5: Literature Review ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "Literature Review", "Section 3");

  const headerRow = ["Author(s)", "Title", "Journal", "Outcomes", "Limitations"].map(t => ({
    text: t, options: { bold: true, color: WHITE, fill: { color: NAVY }, fontSize: 12, align: "center", valign: "middle" }
  }));

  const rows = [
    ["Aydogmus & Boztas (2024) [1]", "Singularity-free IK for humanoid arm via Bayesian-optimized DNN", "Measurement, vol. 229",
      "High accuracy (MAE 0.02); 25ms predictions", "Needs ~10B training samples; no explicit singularity-avoidance loss"],
    ["Deng et al. (2024) [2]", "Physics-informed ML model for inverse dynamics in robotic manipulators", "Applied Soft Computing, vol. 163",
      "Physics-informed network improves torque prediction on a 7-DOF arm", "Solves inverse dynamics (torque), not kinematics; no singularity handling"],
    ["Calzada-Garcia et al. (2025) [3]", "Review: IK, control & planning for manipulators via DNNs", "Algorithms, vol. 18",
      "Surveys 5 years of DNN-based IK methods", "Review only — no new singularity-safe solver proposed"],
    ["Shi et al. (2021) [4]", "Kinematics and singularity analysis of a 7-DOF redundant manipulator", "Sensors, vol. 21",
      "Closed-form singularity conditions for a 7-DOF arm", "Purely analytical; no learning-based real-time solver"],
    ["Hong, Li & Huang (2024) [5]", "RL-enhanced pseudo-inverse for self-collision avoidance of redundant robots", "Frontiers in Neurorobotics, vol. 18",
      "99.72% success avoiding self-collision on a 7-DOF Franka Panda", "Targets self-collision, not singularity / manipulability safety"],
  ];

  const tableRows = [headerRow, ...rows.map(r => r.map(t => ({ text: t, options: { fontSize: 11, color: BLACK, valign: "middle" } })))];

  slide.addTable(tableRows, {
    x: MARGIN, y: 1.65, w: CONTENT_W, h: 5.0,
    colW: [1.9, 3.1, 2.0, 2.85, 2.28],
    border: { type: "solid", color: "D9D9D9", pt: 0.75 },
    autoPage: false,
    valign: "middle",
    margin: [4, 6, 4, 6],
  });
}

// ---------- Slide 6: Gaps in Literature ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "Gaps in Literature", "Section 4");
  const pts = [
    "Most learning-based IK methods have no explicit rule to avoid singularities.",
    "Physics-informed learning has been used for torque prediction, not singularity-safe kinematics.",
    "Classical singularity analysis is accurate but not learning-based or adaptive.",
    "No reviewed work combines all three — training, joint limits, and singularity safety — in one model.",
  ];
  slide.addText(
    pts.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < pts.length - 1, paraSpaceAfter: 20 } })),
    { x: MARGIN, y: 1.8, w: CONTENT_W, h: 4.8, fontFace: FONT, fontSize: 24, color: BLACK, margin: 0, lineSpacing: 32 }
  );
}

// ---------- Slide 7: Objectives ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "Objectives", "Section 5");
  const objs = [
    ["Build the Simulated Arm", "Create a computer model of a 7-joint robot arm that can be learned from."],
    ["Design the AI Model", "Build a neural network that predicts safe joint movements."],
    ["Teach Three Safety Rules", "Reach the target, respect joint limits, and avoid risky positions."],
    ["Compare With Tradition", "Benchmark the AI model against a classical numerical solver."],
    ["Test and Fix", "Find and fix real problems uncovered during training."],
  ];
  let y = 1.65;
  const rowH = 1.02;
  objs.forEach((o, i) => {
    numberedCircle(slide, MARGIN, y, i + 1, 0.55);
    slide.addText(o[0], { x: MARGIN + 0.8, y: y - 0.05, w: CONTENT_W - 0.8, h: 0.4, fontFace: FONT, fontSize: 21, color: NAVY, bold: true, margin: 0 });
    slide.addText(o[1], { x: MARGIN + 0.8, y: y + 0.35, w: CONTENT_W - 0.8, h: 0.5, fontFace: FONT, fontSize: 16, color: BLACK, margin: 0 });
    y += rowH;
  });
}

// ---------- Slide 8: Our Approach ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "Our Approach", "Section 6");

  const steps = ["Target\nPoint", "AI Model\n(uses current pose)", "Predicted\nJoint Angles", "Safety\nCheck", "Safe Arm\nMovement"];
  const n = steps.length;
  const arrowW = 0.5;
  const boxW = (CONTENT_W - arrowW * (n - 1)) / n;
  const boxH = 1.3;
  const y = 2.5;
  let x = MARGIN;
  steps.forEach((s, i) => {
    slide.addShape("roundRect", { x, y, w: boxW, h: boxH, rectRadius: 0.08, fill: { color: i === 3 ? NAVY : LIGHT_TINT }, line: { color: NAVY, width: 1 } });
    slide.addText(s, { x, y, w: boxW, h: boxH, fontFace: FONT, fontSize: 15, bold: true, color: i === 3 ? WHITE : NAVY, align: "center", valign: "middle", margin: 0 });
    x += boxW;
    if (i < n - 1) {
      slide.addShape("rightArrow", { x: x + 0.02, y: y + boxH / 2 - 0.15, w: arrowW - 0.04, h: 0.3, fill: { color: NAVY }, line: { type: "none" } });
      x += arrowW;
    }
  });

  slide.addText(
    "The AI looks at where the arm is right now, then predicts a small, safe movement toward the target — instead of guessing a brand-new pose from scratch.",
    { x: MARGIN, y: y + boxH + 0.6, w: CONTENT_W, h: 1.2, fontFace: FONT, fontSize: 20, color: BLACK, align: "center", margin: 0, lineSpacing: 28 }
  );
}

// ---------- Slide 9: How We Keep It Safe ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "How We Keep It Safe", "Section 6");
  const rules = [
    ["Reach the Target", "The predicted hand position should end up close to the goal point."],
    ["Respect Joint Limits", "No joint should twist past its safe mechanical range."],
    ["Avoid Risky Positions", "The AI is discouraged from moving into “locked”, high-risk configurations."],
  ];
  let y = 1.9;
  const rowH = 1.55;
  rules.forEach((r, i) => {
    numberedCircle(slide, MARGIN, y, i + 1, 0.6);
    slide.addText(r[0], { x: MARGIN + 0.9, y: y - 0.05, w: CONTENT_W - 0.9, h: 0.45, fontFace: FONT, fontSize: 24, color: NAVY, bold: true, margin: 0 });
    slide.addText(r[1], { x: MARGIN + 0.9, y: y + 0.45, w: CONTENT_W - 0.9, h: 0.65, fontFace: FONT, fontSize: 18, color: BLACK, margin: 0, lineSpacing: 24 });
    y += rowH;
  });
  slide.addText("The AI is trained on all three rules at the same time — no separate steps.", {
    x: MARGIN, y: y + 0.05, w: CONTENT_W, h: 0.5, fontFace: FONT, fontSize: 16, italic: true, color: GRAY, margin: 0,
  });
}

// ---------- Slide 10: What We Built So Far ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "What We Built So Far", "Section 7");
  const items = [
    "A simulated 7-joint robot arm (Franka Panda) that a computer can move and measure",
    "A traditional (classical) solver, used as a fair comparison baseline",
    "The AI model itself — the physics-informed neural network",
    "A training system that generates its own practice examples — no manual dataset needed",
    "A testing and comparison system to measure accuracy and safety",
  ];
  slide.addText(
    items.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < items.length - 1, paraSpaceAfter: 20 } })),
    { x: MARGIN, y: 1.85, w: CONTENT_W, h: 4.6, fontFace: FONT, fontSize: 24, color: BLACK, margin: 0, lineSpacing: 32 }
  );
}

// ---------- Slide 11: Testing & Debugging ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "Testing & Debugging", "Section 7");
  slide.addText("We didn’t just build it — we tested it and found real problems.", {
    x: MARGIN, y: 1.65, w: CONTENT_W, h: 0.5, fontFace: FONT, fontSize: 20, italic: true, color: GRAY, margin: 0,
  });
  const problems = [
    ["Problem 1", "A math error in how arm sensitivity was calculated — caught by cross-checking against an independent method, then fixed."],
    ["Problem 2", "Some training examples asked the AI to do something physically impossible for it — found by analyzing results, then fixed."],
    ["Problem 3", "The “avoid risky positions” rule was barely active because its safety threshold was set too strictly — found and fixed."],
  ];
  let y = 2.35;
  const rowH = 1.5;
  problems.forEach((p, i) => {
    numberedCircle(slide, MARGIN, y, i + 1, 0.55);
    slide.addText(p[0], { x: MARGIN + 0.85, y: y - 0.05, w: CONTENT_W - 0.85, h: 0.4, fontFace: FONT, fontSize: 20, color: NAVY, bold: true, margin: 0 });
    slide.addText(p[1], { x: MARGIN + 0.85, y: y + 0.38, w: CONTENT_W - 0.85, h: 0.9, fontFace: FONT, fontSize: 16, color: BLACK, margin: 0, lineSpacing: 21 });
    y += rowH;
  });
}

// ---------- Slide 12: Preliminary Results - Training Progress ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "Preliminary Results: Training Progress", "Section 8");

  const steps = [1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000];
  const errs = [145.44, 98.97, 72.41, 59.68, 55.64, 49.32, 44.50, 42.96, 40.06, 40.10, 40.20];

  slide.addChart("line", [{ name: "Position error (mm)", labels: steps.map(String), values: errs }], {
    x: MARGIN, y: 1.7, w: CONTENT_W, h: 4.3,
    showTitle: true, title: "How Far the Predicted Hand Is From the Target, Over Training", titleFontFace: FONT, titleFontSize: 16, titleColor: NAVY,
    showLegend: false,
    chartColors: [NAVY],
    lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 6,
    catAxisTitle: "Training step", catAxisLabelColor: GRAY, catAxisLabelFontSize: 11, catAxisTitleFontSize: 13,
    valAxisTitle: "Error (mm) — lower is better", valAxisLabelColor: GRAY, valAxisLabelFontSize: 11, valAxisTitleFontSize: 13,
    valGridLine: { color: "E5E5E5", size: 1 },
    catGridLine: { style: "none" },
  });

  slide.addText("Error dropped steadily and kept improving as training continued (10,000 steps shown).", {
    x: MARGIN, y: 6.15, w: CONTENT_W, h: 0.5, fontFace: FONT, fontSize: 16, italic: true, color: GRAY, align: "center", margin: 0,
  });
}

// ---------- Slide 13: Preliminary Results - AI vs Traditional ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "Preliminary Results: AI vs Traditional Method", "Section 8");

  const halfW = (CONTENT_W - 0.5) / 2;

  slide.addChart("bar", [{ name: "Distance to target (mm)", labels: ["Our AI Model", "Classical Method"], values: [32.69, 0.47] }], {
    x: MARGIN, y: 1.75, w: halfW, h: 4.0,
    showTitle: true, title: "Distance to Target (mm)\nlower is better", titleFontFace: FONT, titleFontSize: 14, titleColor: NAVY,
    showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: BLACK,
    chartColors: [NAVY],
    catAxisLabelColor: GRAY, catAxisLabelFontSize: 13,
    valAxisLabelColor: GRAY, valAxisLabelFontSize: 11,
    valGridLine: { color: "E5E5E5", size: 1 }, catGridLine: { style: "none" },
  });

  slide.addChart("bar", [{ name: "Safety margin", labels: ["Our AI Model", "Classical Method"], values: [0.0153, 0.0105] }], {
    x: MARGIN + halfW + 0.5, y: 1.75, w: halfW, h: 4.0,
    showTitle: true, title: "Safety Margin Near Risky Positions\nhigher is safer", titleFontFace: FONT, titleFontSize: 14, titleColor: NAVY,
    showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: BLACK, dataLabelFormatCode: "0.0000",
    chartColors: ["4C7A3A"],
    catAxisLabelColor: GRAY, catAxisLabelFontSize: 13,
    valAxisLabelColor: GRAY, valAxisLabelFontSize: 11,
    valGridLine: { color: "E5E5E5", size: 1 }, catGridLine: { style: "none" },
  });

  slide.addText("Our model is already safer near risky positions — accuracy is still catching up with more training.", {
    x: MARGIN, y: 5.95, w: CONTENT_W, h: 0.6, fontFace: FONT, fontSize: 17, italic: true, color: GRAY, align: "center", margin: 0,
  });
}

// ---------- Slide 14: Limitations & Next Steps ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "Current Limitations & Next Steps", "Section 9");
  const pts = [
    "Accuracy still trails the traditional method — needs more training and a bigger network.",
    "Not yet connected to a 3D physics simulator for visual demonstration.",
    "Next: longer training runs, a bigger network, and simulator integration for a live demo.",
  ];
  slide.addText(
    pts.map((t, i) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: i < pts.length - 1, paraSpaceAfter: 20 } })),
    { x: MARGIN, y: 1.8, w: CONTENT_W, h: 4.3, fontFace: FONT, fontSize: 24, color: BLACK, margin: 0, lineSpacing: 32 }
  );
}

// ---------- Slide 15: References ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  addHeader(slide, "References", "IEEE Format");
  const refs = [
    "[1]  O. Aydogmus and G. Boztas, “Implementation of singularity-free inverse kinematics for humanoid robotic arm using Bayesian optimized deep neural network,” Measurement, vol. 229, p. 114471, 2024.",
    "[2]  W. Deng, F. Ardiani, K. T. P. Nguyen, M. Benoussaad, and K. Medjaher, “Physics informed machine learning model for inverse dynamics in robotic manipulators,” Applied Soft Computing, vol. 163, p. 111877, 2024.",
    "[3]  A. Calzada-Garcia, J. G. Victores, F. J. Naranjo-Campos, and C. Balaguer, “A review on inverse kinematics, control and planning for robotic manipulators with and without obstacles via deep neural networks,” Algorithms, vol. 18, no. 1, p. 23, 2025.",
    "[4]  X. Shi, Y. Guo, X. Chen, Z. Chen, and Z. Yang, “Kinematics and singularity analysis of a 7-DOF redundant manipulator,” Sensors, vol. 21, no. 21, 2021.",
    "[5]  T. Hong, W. Li, and K. Huang, “A reinforcement learning enhanced pseudo-inverse approach to self-collision avoidance of redundant robots,” Frontiers in Neurorobotics, vol. 18, 2024.",
    "[6]  Y. Nakamura and H. Hanafusa, “Inverse kinematic solutions with singularity robustness for robot manipulator control,” Journal of Dynamic Systems, Measurement, and Control, vol. 108, no. 3, pp. 163–171, 1986.",
  ];
  slide.addText(
    refs.map((t, i) => ({ text: t, options: { breakLine: i < refs.length - 1, paraSpaceAfter: 16 } })),
    { x: MARGIN, y: 1.7, w: CONTENT_W, h: 5.2, fontFace: FONT, fontSize: 14, color: BLACK, margin: 0, lineSpacing: 18 }
  );
}

// ---------- Slide 16: Thank You ----------
{
  const slide = pres.addSlide({ masterName: "PLAIN" });
  slide.addText("Thank You", {
    x: MARGIN, y: 2.9, w: CONTENT_W, h: 1.0,
    fontFace: FONT, fontSize: 44, color: NAVY, bold: true, align: "center", margin: 0,
  });
  slide.addText("Questions?", {
    x: MARGIN, y: 3.9, w: CONTENT_W, h: 0.7,
    fontFace: FONT, fontSize: 24, color: GRAY, align: "center", margin: 0,
  });
}

pres.writeFile({ fileName: "PINN_IK_Mid_Review.pptx" }).then(() => {
  console.log("wrote PINN_IK_Mid_Review.pptx");
});
