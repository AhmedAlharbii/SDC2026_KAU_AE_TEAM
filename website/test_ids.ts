import fs from 'fs';

const files = [
  'src/data/part1.md',
  'src/data/part2.md',
  'src/data/part3.md',
  'src/data/part4.md'
];

let paperMarkdown = "";
files.forEach(f => {
  if (fs.existsSync(f)) {
    paperMarkdown += fs.readFileSync(f, 'utf8') + '\n\n';
  }
});

const regex = /^##\s+(.+)$/gm;
let match;
while ((match = regex.exec(paperMarkdown)) !== null) {
  if (match[1].trim() === "TABLE OF CONTENTS") continue;
  const originalTitle = match[1].trim();
  const title = originalTitle.replace(/^\d+\.\s*/, '');
  const id = originalTitle
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/[\s_-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  console.log(`originalTitle: ${originalTitle} | id: ${id}`);
}
