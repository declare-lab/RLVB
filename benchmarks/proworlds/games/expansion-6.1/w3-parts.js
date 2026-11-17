// the shared form of the PART games. A member = one value of each binary part (the rules text names every
// member's parts in words). The board is one constant background tile; each part shows its value's tile at one or more fixed slots. Long uses the
// SAME tiles with wider public gaps between the slots, and the shift moves by whole grid steps (stride sites), so a gap site is
// background in every world. Every site that can vary shows a part tile in some worlds, so one look at it can tell about the member (C5).
import { defineTile, tile, tileNames } from './tiles.js';
import { w3Tile } from './w3-tiles.js';

export function mirrored(name) {                       // the left-right mirror of a defined tile (for example the left mouth corner)
  const key = name + '~m'; if (tileNames().includes(key)) return key;
  const t = tile(name), m = new Uint8Array(256);
  for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) m[y * 16 + x] = t[y * 16 + 15 - x];
  return defineTile(key, m);
}

// parts: [{ one: tileName, zero: tileName, at: { normal: [[x, y, mirror?], ...], long: [[x, y, mirror?], ...] } }]
export function partsGame({ prefix, members, parts, background, boards, ...text }) {
  const bg = w3Tile(prefix, background);
  const P = parts.map(p => ({ ...p, one: w3Tile(prefix, p.one), zero: w3Tile(prefix, p.zero) }));
  const names = Object.keys(members);
  for (const [m, bits] of Object.entries(members)) if (bits.length !== P.length || !/^[01]+$/.test(bits)) throw Error(`${prefix}: member ${m} bits ${bits}`);
  return {
    ...text, members: names,
    boards: Object.fromEntries(Object.entries(boards).map(([k, b]) => [k, { cols: b.cols, rows: b.rows, budget: b.budget }])),
    nuisances(board) {                                  // shiftX / shiftY (default: shift) in whole stride steps
      const b = boards[board], sx = b.shiftX ?? b.shift, sy = b.shiftY ?? b.shift, out = [];
      for (let dy = -sy; dy <= sy; dy++) for (let dx = -sx; dx <= sx; dx++) out.push({ dx: dx * (b.stride ?? 1), dy: dy * (b.stride ?? 1) }); return out;
    },
    layout(member, nu, board) {
      const b = boards[board], sites = Array(b.cols * b.rows).fill(bg), bits = members[member];
      for (let k = 0; k < P.length; k++) {
        const t = bits[k] === '1' ? P[k].one : P[k].zero;
        const L = boards.long.scale ?? 3;                   // long default: the normal slots x scale (rounded), gaps between them
        const at = P[k].at[board] ?? P[k].at.normal.map(([x, y, m]) => [Math.round(x * L) + 1, Math.round(y * L) + 1, m]);
        for (const [x, y, mir] of at) {
          const name = mir ? mirrored(t) : t;
          const X = x + nu.dx, Y = y + nu.dy;
          if (X < 0 || Y < 0 || X >= b.cols || Y >= b.rows) return null;
          sites[Y * b.cols + X] = name;
        }
      }
      return sites;
    },
  };
}
