// the rules-text helper of the PART games: "name = part value, part value, ..." for every member.
export function partList(members, words) {
  return Object.entries(members).map(([m, b]) => `${m} = ${[...b].map((v, k) => words[k][v === '1' ? 0 : 1]).join(', ')}`).join('; ');
}
