const fs = require("fs");
let text = fs.readFileSync("paper_text.txt", "utf8");
text = text.replace(/<img[^>]*>/g, "\n[IMAGE]\n");
text = text.replace(/<[^>]+>/g, "\n");
text = text.replace(/\n\s*\n/g, "\n");
fs.writeFileSync("paper_clean.txt", text);
