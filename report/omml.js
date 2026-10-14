// Minimal LaTeX-subset -> OMML (Word equation) converter built on docx's BuilderElement.
// Supported: plain text, {groups}, x_{..}, x^{..}, \frac{}{}, \sqrt{}, \sum_{}^{}{body}, \prod,
// \hat{x}, \bar{x}, \tilde{x}, \text{upright}, \op{upright-italic-free op}, \lim{op}{below},
// delimiters \p{..} ( ), \b{..} [ ], \c{..} { }, \a{..} ⟨ ⟩, \n{..} ‖ ‖, \abs{..} | |, \floor{..}.
const { BuilderElement, Math: OMath } = require("docx");

const E = (name, children = [], attributes) => new BuilderElement({ name, children, attributes });
const val = (v) => ({ val: { key: "m:val", value: v } });

function run(text, upright = false) {
  const kids = [];
  if (upright) kids.push(E("m:rPr", [E("m:sty", [], val("p"))]));
  kids.push(new BuilderElement({ name: "m:t", attributes: { space: { key: "xml:space", value: "preserve" } }, children: [text] }));
  return E("m:r", kids);
}
const base = (k) => E("m:e", k);
const delim = (open, close, kids) =>
  E("m:d", [E("m:dPr", [E("m:begChr", [], val(open)), E("m:endChr", [], val(close))]), base(kids)]);
const acc = (chr, kids) => E("m:acc", [E("m:accPr", [E("m:chr", [], val(chr))]), base(kids)]);
const nary = (chr, sub, sup, body) => E("m:nary", [
  E("m:naryPr", [E("m:chr", [], val(chr)), E("m:limLoc", [], val("undOvr")),
    ...(sup ? [] : [E("m:supHide", [], val("1"))]), ...(sub ? [] : [E("m:subHide", [], val("1"))])]),
  E("m:sub", sub || []), E("m:sup", sup || []), base(body)]);

const DELIMS = { p: ["(", ")"], b: ["[", "]"], c: ["{", "}"], a: ["⟨", "⟩"], n: ["‖", "‖"], abs: ["|", "|"], floor: ["⌊", "⌋"] };

function parse(src) {
  let i = 0;
  function group() { // expects '{'
    if (src[i] !== "{") throw new Error(`expected { at ${i} in ${src}`);
    i++;
    const out = seq("}");
    i++; // skip }
    return out;
  }
  function atom() {
    const ch = src[i];
    if (ch === "{") return group();
    if (ch === "\\") {
      i++;
      let name = "";
      while (i < src.length && /[a-zA-Z]/.test(src[i])) name += src[i++];
      if (name === "") { const c = src[i++]; return [run(c)]; } // escaped char e.g. \{ \_
      if (name === "frac") { const n = group(), d = group(); return [E("m:f", [E("m:num", n), E("m:den", d)])]; }
      if (name === "sqrt") { const k = group(); return [E("m:rad", [E("m:radPr", [E("m:degHide", [], val("1"))]), E("m:deg"), base(k)])]; }
      if (name === "hat") return [acc("̂", group())];
      if (name === "bar") return [acc("̅", group())];
      if (name === "tilde") return [acc("̃", group())];
      if (name === "text") return [run(rawGroup(), true)];
      if (name === "sum" || name === "prod") {
        let sub = null, sup = null;
        for (let t = 0; t < 2; t++) {
          if (src[i] === "_") { i++; sub = scriptArg(); } else if (src[i] === "^") { i++; sup = scriptArg(); }
        }
        skipSpace();
        const body = group();
        return [nary(name === "sum" ? "∑" : "∏", sub, sup, body)];
      }
      if (name === "lim") { // \lim{op}{below}
        const op = rawGroup(); const below = group();
        return [E("m:limLow", [base([run(op, true)]), E("m:lim", below)])];
      }
      if (DELIMS[name]) { const [o, c] = DELIMS[name]; return [delim(o, c, group())]; }
      throw new Error(`unknown command \\${name}`);
    }
    // plain text: one character (scripts attach to it); runs are merged afterwards
    i++;
    const r = run(ch); r.plain = ch;
    return [r];
  }
  function rawGroup() {
    if (src[i] !== "{") throw new Error(`expected { for raw group in ${src}`);
    let depth = 0, s = "";
    i++;
    while (!(src[i] === "}" && depth === 0)) { if (src[i] === "{") depth++; if (src[i] === "}") depth--; s += src[i++]; }
    i++;
    return s;
  }
  function scriptArg() { return src[i] === "{" ? group() : atom(); }
  function skipSpace() { while (src[i] === " ") i++; }
  function seq(stop) {
    const out = [];
    let buf = "";
    const flush = () => { if (buf) out.push(run(buf)); buf = ""; };
    while (i < src.length && src[i] !== stop) {
      let a = atom();
      let sub = null, sup = null;
      while (src[i] === "_" || src[i] === "^") {
        const t = src[i++];
        if (t === "_") sub = scriptArg(); else sup = scriptArg();
      }
      if (sub && sup) a = [E("m:sSubSup", [E("m:sSubSupPr"), base(a), E("m:sub", sub), E("m:sup", sup)])];
      else if (sub) a = [E("m:sSub", [E("m:sSubPr"), base(a), E("m:sub", sub)])];
      else if (sup) a = [E("m:sSup", [E("m:sSupPr"), base(a), E("m:sup", sup)])];
      if (a.length === 1 && a[0].plain !== undefined) { buf += a[0].plain; continue; }
      flush();
      out.push(...a);
    }
    flush();
    return out;
  }
  return seq(null);
}

const math = (src) => new OMath({ children: parse(src) });
module.exports = { math, parse };
