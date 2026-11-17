// Face (assemble): one large face fills the board. Its emotion follows from four parts that sit in their places on the face:
// the brows (raised or lowered), the eyes (open or closed), the mouth corners (up or down) and the mouth (open or closed). The rules text names
// the parts of every emotion. The eight emotions use eight of the sixteen part sets, so that every informative look halves what is left.
import { partsGame } from './w3-parts.js';

const MEMBERS = {               // brows raised? eyes open? corners up? mouth open?
  happy: '0110', sad: '1100', angry: '0101', surprised: '1111', sleepy: '0000', scared: '1001', calm: '1010', laughing: '0011',
};
const WORD = [['raised brows', 'lowered brows'], ['open eyes', 'closed eyes'], ['mouth corners up', 'mouth corners down'], ['an open mouth', 'a closed mouth']];
const says = Object.entries(MEMBERS).map(([m, b]) => `${m} = ${[...b].map((v, k) => WORD[k][v === '1' ? 0 : 1]).join(', ')}`).join('; ');
export const FACE = partsGame({
  prefix: 'face', background: 'scene_cheek',
  name: 'Face', kind: 'relative', picture: true, members: MEMBERS,
  intro: {
    normal: 'One large face fills the board. Two brows sit at the top, two eyes below them, the two mouth corners lower down and the middle of the mouth at the bottom; every other site shows plain skin. The whole face may sit up to one site higher, lower, left or right.',
    long: 'One large face fills the board, drawn wide: two brows at the top, two eyes below them, the two mouth corners lower down and the middle of the mouth at the bottom, with wide plain skin between the parts. The whole face may sit three sites higher, lower, left or right.',
  },
  ask: 'Which emotion does the face show?',
  setText: `one of eight emotions, each with its own brows, eyes and mouth: ${says}.`,
  parts: [
    { one: 'brows_raised', zero: 'brows_lowered', at: { normal: [[2, 1], [7, 1, true]], long: [[6, 3], [24, 3, true]] } },
    { one: 'eyes_open', zero: 'eyes_closed', at: { normal: [[2, 4], [7, 4]], long: [[6, 12], [24, 12]] } },
    { one: 'corners_up', zero: 'corners_down', at: { normal: [[2, 7, true], [7, 7]], long: [[6, 24, true], [24, 24]] } },
    { one: 'mouth_open', zero: 'mouth_closed', at: { normal: [[4, 8]], long: [[15, 27]] } },
  ],
  boards: { normal: { cols: 10, rows: 10, budget: 35, shift: 1, stride: 1 }, long: { cols: 31, rows: 31, budget: 65, shift: 1, stride: 3 } },
});
