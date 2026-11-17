// the shared form of the CLUE-CHAIN games (footprints, scent trail), after the wind vanes. Four levels
// of clue places form a binary tree: 1 place at level 1, 2 at level 2, 4 at level 3, 8 at level 4. Every clue place holds a clue tile. The real
// chain starts at the level-1 clue: its tile (one of two directions) is the member's first bit and points to the real level-2 place, and so on;
// the level-4 clue points to the end of the route, where the member's icon stands. Clues off the real chain are decoys: each of the 11 decoys
// points its own random way (11 nuisance bits, as the wind vanes' decoys). So each real clue is one bit, and where to look next depends on what it showed.
import { w3Tile } from './w3-tiles.js';

// spec.boards[b]: { cols, rows, budget, nodes: [[p], [p0, p1], [p00..p11], [p000..p111]], leaves: {code: [x, y]} } with p = [x, y];
// nodes[L][path] = the clue place reached by the bits `path` (as a binary number, first bit high); spec.dirs[L] = [tile for bit 1, tile for bit 0].
export function chainGame({ prefix, background, dirs, members, icons, boards, ...text }) {
  const D = dirs.map(([a, b]) => [w3Tile(prefix, a), w3Tile(prefix, b)]);
  const I = Object.fromEntries(Object.keys(members).map(m => [m, w3Tile(prefix, icons[m])]));
  const bg = background.includes(':') ? background : w3Tile(prefix, background);
  return {
    ...text, members: Object.keys(members),
    boards: Object.fromEntries(Object.entries(boards).map(([k, b]) => [k, { cols: b.cols, rows: b.rows, budget: b.budget }])),
    nuisances: () => Array.from({ length: 2048 }, (_, n) => n),
    layout(member, nu, board) {
      const b = boards[board], code = members[member], sites = Array(b.cols * b.rows).fill(bg);
      const put = ([x, y], t) => { if (x < 0 || y < 0 || x >= b.cols || y >= b.rows) throw Error(`${prefix}/${board}: off board`); sites[y * b.cols + x] = t; };
      for (const [m, c] of Object.entries(members)) put(b.leaves[c], I[m]);           // the ten icons at their route ends (public scenery)
      let k = 0;                                           // the k-th decoy place (in place order) takes nuisance bit k
      for (let L = 0; L < 4; L++) {
        const realPath = L === 0 ? 0 : parseInt(code.slice(0, L), 2);
        b.nodes[L].forEach((p, path) => put(p, D[L][(path === realPath ? code[L] === '1' : ((nu >> k++) & 1) === 1) ? 0 : 1]));
      }
      return sites;
    },
  };
}
// The 16 places of a centred binary layout from the four level offsets.
export function treeNodes(root, steps) {
  const nodes = [[root]];
  for (let L = 1; L < 4; L++) nodes.push(nodes[L - 1].flatMap(([x, y]) => [[x + steps[L - 1][0][0], y + steps[L - 1][0][1]], [x + steps[L - 1][1][0], y + steps[L - 1][1][1]]]));
  return nodes;
}
export function leavesOf(nodes, step, codes) {
  return Object.fromEntries(codes.map(c => { const [x, y] = nodes[3][parseInt(c.slice(0, 3), 2)]; const s = step[c[3] === '1' ? 0 : 1]; return [c, [x + s[0], y + s[1]]]; }));
}
