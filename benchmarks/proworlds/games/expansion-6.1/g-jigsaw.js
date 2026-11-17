// Jigsaw scene (assemble): the pieces of one landscape picture lie on the board in their places. The place name follows from four
// parts of the picture: water, trees, buildings and snow (the rules name the parts of every place).
import { partsGame } from './w3-parts.js';
import { partList } from './w3-text.js';
const MEMBERS = { harbour: '1010', island: '1100', 'lakeside cabin': '1110', 'frozen lake': '1001', 'fishing village': '1011', 'snowy woods': '0101', 'ski resort': '0111', city: '0010', jungle: '0100', snowfield: '0001' };
const WORDS = [['water', 'no water'], ['trees', 'no trees'], ['buildings', 'no buildings'], ['snow', 'no snow']];
export const JIGSAW = partsGame({
  prefix: 'jigsaw', background: 'building_none', name: 'Jigsaw scene', kind: 'assemble', picture: true, members: MEMBERS,
  intro: { normal: 'The pieces of one landscape picture lie on the board in their places: the ground piece at the bottom middle, the plant piece at the middle left, the building piece in the middle and the sky piece at the top right. Every other site is a blank piece. The picture may sit one site further left or right. A building piece with no building looks like a blank piece. The ground piece shows water or plain land, the plant piece shows trees or only grass, and the sky piece shows falling snow or the sun.',
           long: 'The pieces of one large landscape picture lie on the board in their places, far apart: the ground piece at the bottom middle, the plant piece at the middle left, the building piece in the middle and the sky piece at the top right. Every other site is a blank piece. The picture may sit two sites further left or right. A building piece with no building looks like a blank piece. The ground piece shows water or plain land, the plant piece shows trees or only grass, and the sky piece shows falling snow or the sun.' },
  ask: 'Which place does the picture show?',
  setText: `one of ten places: ${partList(MEMBERS, WORDS)}.`,
  parts: [
    { one: 'ground_water', zero: 'ground_land', at: { normal: [[3, 6]], long: [[10, 16]] } },
    { one: 'plants_trees', zero: 'plants_none', at: { normal: [[1, 3]], long: [[3, 10]] } },
    { one: 'building_house', zero: 'building_none', at: { normal: [[4, 2]], long: [[12, 8]] } },
    { one: 'weather_snow', zero: 'weather_sun', at: { normal: [[6, 0]], long: [[19, 1]] } },
  ],
  boards: { normal: { cols: 8, rows: 8, budget: 35, shiftX: 1, shiftY: 0, stride: 1 }, long: { cols: 24, rows: 22, budget: 65, shiftX: 1, shiftY: 0, stride: 2, scale: 2.2 } },
});
