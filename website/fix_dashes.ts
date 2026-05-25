import fs from 'fs';
import path from 'path';

const filesToProcess = [
  'src/data/part1.md',
  'src/data/part2.md',
  'src/data/part3.md',
  'src/data/part4.md',
  'src/components/Problem.tsx'
];

for (const file of filesToProcess) {
  const filePath = path.resolve(file);
  if (fs.existsSync(filePath)) {
    let content = fs.readFileSync(filePath, 'utf-8');
    content = content.replaceAll('–', '-'); // En dash to hyphen
    fs.writeFileSync(filePath, content, 'utf-8');
    console.log(`Replaced en dashes in ${file}`);
  }
}
