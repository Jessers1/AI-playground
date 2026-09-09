// Build a .docx report from a report markdown file plus a transcript markdown file (appended as Appendix A)
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, BorderStyle, LevelFormat, PageBreak, Footer, PageNumber,
} = require("docx");

const [reportMd, transcriptMd, OUT] = process.argv.slice(2);
if (!OUT) { console.error("usage: node build_report_docx.js report.md transcript.md out.docx [title]"); process.exit(1); }
const TITLE = process.argv[5] || "Report";
const report = fs.readFileSync(reportMd, "utf8");
const transcript = fs.readFileSync(transcriptMd, "utf8");

const FONT = "Calibri";
const PAGE_W = 12240, MARGIN = 1440, CONTENT_W = PAGE_W - 2 * MARGIN;

// ---- inline markdown -> TextRuns (**bold**, *italic*, `code`) ----
function inline(text, base = {}) {
  const runs = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) runs.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) runs.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    else if (tok.startsWith("`")) runs.push(new TextRun({ text: tok.slice(1, -1), font: "Consolas", ...base }));
    else runs.push(new TextRun({ text: tok.slice(1, -1), italics: true, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), ...base }));
  return runs;
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "E8EDF3", color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: inline(text, { bold: !!opts.header, size: 19 }), spacing: { after: 0 } })],
  });
}

function table(rows) {
  const ncol = rows[0].length;
  // give first column a bit less, distribute rest
  let widths;
  if (ncol === 2) widths = [Math.round(CONTENT_W * 0.28), CONTENT_W - Math.round(CONTENT_W * 0.28)];
  else if (ncol === 3) widths = [Math.round(CONTENT_W * 0.3), Math.round(CONTENT_W * 0.5), CONTENT_W - Math.round(CONTENT_W * 0.3) - Math.round(CONTENT_W * 0.5)];
  else if (ncol === 4) widths = [Math.round(CONTENT_W * 0.16), Math.round(CONTENT_W * 0.3), Math.round(CONTENT_W * 0.36), 0].map((w, i, a) => i === 3 ? CONTENT_W - a[0] - a[1] - a[2] : w);
  else widths = Array(ncol).fill(Math.floor(CONTENT_W / ncol));
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => cell(c, { width: widths[j], header: i === 0 })),
    })),
  });
}

// ---- block markdown -> docx elements ----
function mdToBlocks(md, { headingOffset = 0, quoteStyle = false } = {}) {
  const lines = md.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    let line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith("---")) { i++; continue; }
    let h = line.match(/^(#{1,4})\s+(.*)/);
    if (h) {
      const lvl = Math.min(h[1].length + headingOffset, 4);
      const map = { 1: HeadingLevel.HEADING_1, 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3, 4: HeadingLevel.HEADING_4 };
      out.push(new Paragraph({ heading: map[lvl], children: inline(h[2].replace(/\*\*/g, "")) }));
      i++; continue;
    }
    if (line.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        const cells = lines[i].slice(1, lines[i].endsWith("|") ? -1 : undefined).split("|").map(s => s.trim());
        if (!cells.every(c => /^-+$/.test(c))) rows.push(cells);
        i++;
      }
      out.push(table(rows));
      out.push(new Paragraph({ spacing: { after: 60 } }));
      continue;
    }
    if (line.startsWith("> ")) {
      while (i < lines.length && lines[i].startsWith("> ")) {
        out.push(new Paragraph({
          children: inline(lines[i].slice(2).replace(/\s+$/, "")),
          indent: { left: 540 },
          border: { left: { style: BorderStyle.SINGLE, size: 12, color: "9AA5B1", space: 8 } },
          spacing: { after: 60 },
        }));
        i++;
      }
      out.push(new Paragraph({ spacing: { after: 40 } }));
      continue;
    }
    if (/^- /.test(line)) {
      out.push(new Paragraph({ children: inline(line.slice(2).replace(/\s+$/, "")), numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 } }));
      i++; continue;
    }
    let n = line.match(/^(\d+)\.\s+(.*)/);
    if (n) {
      out.push(new Paragraph({ children: inline(n[2].replace(/\s+$/, "")), numbering: { reference: "numbers", level: 0 }, spacing: { after: 60 } }));
      i++; continue;
    }
    // transcript line: **[mm:ss] SPEAKER:** text
    const t = line.match(/^\*\*\[(\d\d:\d\d)\]\s*([^*]+?):\*\*\s*(.*)$/);
    if (t && quoteStyle) {
      const runs = [
        new TextRun({ text: `[${t[1]}] `, font: "Consolas", size: 18, color: "6B7280" }),
        new TextRun({ text: t[2] + ": ", bold: true, size: 20 }),
        ...inline(t[3].replace(/\s+$/, ""), { size: 20 }),
      ];
      out.push(new Paragraph({ children: runs, spacing: { after: 50 }, indent: { left: 1000, hanging: 1000 } }));
      i++; continue;
    }
    // paragraph (possibly with trailing double-space line breaks -> separate paragraphs)
    out.push(new Paragraph({ children: inline(line.replace(/\s+$/, "")), spacing: { after: 120 } }));
    i++;
  }
  return out;
}

// Report body: strip the trailing "Appendix A" stub from the md (we append the real transcript)
const bodyMd = report.split(/\n---\n\n# Appendix A/)[0];
const children = [];

// Title block
children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun(TITLE)] }));
children.push(new Paragraph({ children: [new TextRun({ text: "CONFIDENTIAL — DRAFT FOR REVIEW", bold: true, color: "9B1C1C", size: 20 })], spacing: { after: 200 } }));
// skip the original H1
const afterTitle = bodyMd.replace(/^# .*\n/, "");
children.push(...mdToBlocks(afterTitle));

// Appendix A
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Appendix A — Full transcript")] }));
const tBody = transcript.replace(/^# .*\n/, "");
children.push(...mdToBlocks(tBody, { headingOffset: 1, quoteStyle: true }));

const doc = new Document({
  creator: "Transcription pipeline",
  title: TITLE,
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 40, bold: true, color: "1F2937" }, paragraph: { spacing: { after: 120 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, color: "1F2937" }, paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, color: "1F2937" }, paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 23, bold: true, color: "374151" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
      { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, italics: true, color: "374151" }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 3 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 15840 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "Confidential — page ", size: 18, color: "6B7280" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "6B7280" }),
      ] })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length); });
