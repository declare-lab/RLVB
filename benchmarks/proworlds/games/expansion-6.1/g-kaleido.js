// Kaleidoscope (assemble): one large kaleidoscope pattern fills the board. Its name follows from three parts: the centre, the
// spokes and the rim (the rules name the parts of every pattern).
import { partsGame } from './w3-parts.js';
import { partList } from './w3-text.js';
const MEMBERS = { snowflake: '111', starburst: '011', wheel: '110', cross: '010', diamond: '101', web: '001', petal: '100', ring: '000' };
const WORDS = [['a solid centre dot', 'a hollow centre circle'], ['straight spokes', 'curved spokes'], ['a rim of points', 'a rim of round bumps']];
export const KALEIDO = partsGame({
  prefix: 'kaleidoscope', background: 'scene_plain', name: 'Kaleidoscope', kind: 'symmetry', picture: true, members: MEMBERS,
  intro: { normal: 'One large kaleidoscope pattern fills the board: its centre in the middle, one spoke piece up and to the right of the centre, and one rim piece at the lower left edge. Every other site is plain. The pattern may sit one site further left or right.',
           long: 'One very large kaleidoscope pattern fills the board: its centre in the middle, one spoke piece far up and to the right, and one rim piece far down at the lower left edge. Every other site is plain. The pattern may sit two sites further left or right.' },
  ask: 'Which kaleidoscope pattern is it?',
  setText: `one of eight patterns: ${partList(MEMBERS, WORDS)}.`,
  parts: [
    { one: 'centre_dot', zero: 'centre_hollow', at: { normal: [[4, 4]] } },
    { one: 'spokes_straight', zero: 'spokes_curved', at: { normal: [[7, 1]] } },
    { one: 'rim_points', zero: 'rim_round', at: { normal: [[1, 7]] } },
  ],
  boards: { normal: { cols: 9, rows: 9, budget: 35, shiftX: 1, shiftY: 0, stride: 1 }, long: { cols: 24, rows: 24, budget: 65, shiftX: 1, shiftY: 0, stride: 2, scale: 2.5 } },
});
