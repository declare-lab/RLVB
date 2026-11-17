// the shared form of the PART games (= the w3-parts.js partsGame, with the tiles drawn by
// paint in the game file instead of the W3 table). A member = one value of each binary part (the rules text names every
// member's parts in words). The board is one constant background tile; each part shows its value's tile at fixed slots that
// move together by the hidden shift. Long uses the SAME tiles with wider public gaps (slots x scale), shifting by whole strides.

// parts: [{ one: tileName, zero: tileName, at: { normal: [[x, y], ...] } }]; boards: { normal / long: { cols, rows, budget, shift(X/Y), stride, scale } }
export function partsGame({ members, parts, background, boards, ...text }) {
  for (const [m, bits] of Object.entries(members)) if (bits.length !== parts.length || !/^[01]+$/.test(bits)) throw Error(`member ${m} bits ${bits}`);
  return {
    ...text, members: Object.keys(members),
    boards: Object.fromEntries(Object.entries(boards).map(([k, b]) => [k, { cols: b.cols, rows: b.rows, budget: b.budget }])),
    nuisances(board) {
      const b = boards[board], sx = b.shiftX ?? b.shift, sy = b.shiftY ?? b.shift, out = [];
      for (let dy = -sy; dy <= sy; dy++) for (let dx = -sx; dx <= sx; dx++) out.push({ dx: dx * (b.stride ?? 1), dy: dy * (b.stride ?? 1) }); return out;
    },
    layout(member, nu, board) {
      const b = boards[board], sites = Array(b.cols * b.rows).fill(background), bits = members[member], L = boards.long.scale ?? 3;
      for (let k = 0; k < parts.length; k++) {
        const t = bits[k] === '1' ? parts[k].one : parts[k].zero;
        const at = parts[k].at[board] ?? parts[k].at.normal.map(([x, y]) => [Math.round(x * L) + 1, Math.round(y * L) + 1]);
        for (const [x, y] of at) {
          const X = x + nu.dx, Y = y + nu.dy;
          if (X < 0 || Y < 0 || X >= b.cols || Y >= b.rows) return null;
          sites[Y * b.cols + X] = t;
        }
      }
      return sites;
    },
  };
}
// "name = part value, part value, ..." for every member (= w3-text.js partList).
export const partList = (members, words) => Object.entries(members).map(([m, b]) => `${m} = ${[...b].map((v, k) => words[k][v === '1' ? 0 : 1]).join(', ')}`).join('; ');
