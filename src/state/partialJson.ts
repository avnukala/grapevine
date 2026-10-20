// Parses JSON that a model is still writing. It closes the open string, arrays, and objects, and drops a
// trailing piece that cannot parse yet, such as a key with no value.
export function parsePartialJson(text: string): unknown {
  for (let end = text.length; end > 0; ) {
    const parsed = tryClosed(text.slice(0, end));
    if (parsed !== undefined) return parsed;
    const cut = Math.max(text.lastIndexOf(",", end - 1), text.lastIndexOf("{", end - 1), text.lastIndexOf("[", end - 1));
    if (cut < 0) return undefined;
    end = text[cut] === "," ? cut : cut + 1;
  }
  return undefined;
}

function tryClosed(text: string): unknown {
  const closers: string[] = [];
  let inString = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inString) {
      if (c === "\\") i++;
      else if (c === '"') inString = false;
    } else if (c === '"') inString = true;
    else if (c === "{") closers.push("}");
    else if (c === "[") closers.push("]");
    else if (c === "}" || c === "]") closers.pop();
  }
  const body = (inString ? text + '"' : text).replace(/[,:]\s*$/, "");
  try {
    return JSON.parse(body + closers.reverse().join(""));
  } catch {
    return undefined;
  }
}
