// A small set language over experimental conditions.
//
// The question an analysis asks is a parameter, not three fixed conditions with fixed meanings: a curriculum names its conditions, and an analysis says which cells it cares about as an expression over them:
//
//   both and not sight and not sound        the cross-modal conjunction
//   sight and not both                       cells the pair suppresses
//   attended and not unattended              an attention experiment
//   a or b or c                              anything any condition drives
//
// Each name is the set of cells in the top fraction of the response to that condition, per item, so the expression is evaluated once per item and yields the cells that item's measure is computed over.
//
// Deliberately not `new Function`, for the reasons src/expr.js gives: a scene file is data, and data that can run arbitrary JavaScript when opened is not data.
// The grammar is names, and, or, not, parentheses, and nothing else, so an expression cannot reach the page or the clock.

const isName = c => /[A-Za-z0-9_.\/-]/.test(c);

function tokenize(src){
  const out = [];
  let i = 0;
  while(i < src.length){
    const c = src[i];
    if(/\s/.test(c)){ i++; continue; }
    if(c === '(' || c === ')'){ out.push({ t:c, at:i }); i++; continue; }
    // & | ! are accepted as synonyms, since people write both
    if(c === '&'){ out.push({ t:'and', at:i }); i++; if(src[i] === '&') i++; continue; }
    if(c === '|'){ out.push({ t:'or', at:i }); i++; if(src[i] === '|') i++; continue; }
    if(c === '!'){ out.push({ t:'not', at:i }); i++; continue; }
    if(isName(c)){
      let j = i;
      while(j < src.length && isName(src[j])) j++;
      const w = src.slice(i, j);
      const lw = w.toLowerCase();
      out.push(lw === 'and' || lw === 'or' || lw === 'not'
        ? { t:lw, at:i } : { t:'name', v:w, at:i });
      i = j; continue;
    }
    throw new Error('unexpected character "' + c + '" at ' + i);
  }
  return out;
}

// precedence: not, then and, then or
export function parse(src){
  const ts = tokenize(src);
  let k = 0;
  const peek = () => ts[k];
  const eat = t => { if(!ts[k] || ts[k].t !== t)
    throw new Error('expected ' + t + (ts[k] ? ' but found ' + (ts[k].v || ts[k].t) : ' but the expression ended'));
    return ts[k++]; };
  const primary = () => {
    const tk = peek();
    if(!tk) throw new Error('the expression ended early');
    if(tk.t === 'not'){ k++; return { op:'not', a:primary() }; }
    if(tk.t === '('){ k++; const e = orExpr(); eat(')'); return e; }
    if(tk.t === 'name'){ k++; return { op:'name', v:tk.v }; }
    throw new Error('unexpected "' + (tk.v || tk.t) + '"');
  };
  const andExpr = () => {
    let a = primary();
    while(peek() && peek().t === 'and'){ k++; a = { op:'and', a, b:primary() }; }
    return a;
  };
  const orExpr = () => {
    let a = andExpr();
    while(peek() && peek().t === 'or'){ k++; a = { op:'or', a, b:andExpr() }; }
    return a;
  };
  if(!ts.length) throw new Error('the expression is empty');
  const e = orExpr();
  if(k < ts.length) throw new Error('unexpected "' + (ts[k].v || ts[k].t) + '" after the expression');
  return e;
}

// Every name an expression mentions, so a caller can check them against the conditions a curriculum actually declares and say which one is wrong rather than failing at evaluation time with an empty set.
export function namesUsed(ast, into = new Set()){
  if(!ast) return into;
  if(ast.op === 'name') into.add(ast.v);
  namesUsed(ast.a, into); namesUsed(ast.b, into);
  return into;
}

// sets: name -> Uint8Array membership over the same cell indexing.
// Returns a Uint8Array of the same length.
export function evaluate(ast, sets, n){
  const out = new Uint8Array(n);
  const get = name => {
    const s = sets[name];
    if(!s) throw new Error('no condition named "' + name + '"');
    return s;
  };
  const walk = (node, dst) => {
    if(node.op === 'name'){ dst.set(get(node.v).subarray(0, n)); return dst; }
    if(node.op === 'not'){
      const a = walk(node.a, new Uint8Array(n));
      for(let i=0;i<n;i++) dst[i] = a[i] ? 0 : 1;
      return dst;
    }
    const a = walk(node.a, new Uint8Array(n));
    const b = walk(node.b, new Uint8Array(n));
    if(node.op === 'and') for(let i=0;i<n;i++) dst[i] = (a[i] && b[i]) ? 1 : 0;
    else for(let i=0;i<n;i++) dst[i] = (a[i] || b[i]) ? 1 : 0;
    return dst;
  };
  return walk(ast, out);
}

// Parse once, report the fault in a form a person editing a field can act on, and hand back something that can be evaluated per item.
export function compileSet(src, known){
  let ast;
  try { ast = parse(src); }
  catch(e){ return { error:e.message }; }
  if(known && known.length){
    const bad = [...namesUsed(ast)].filter(n => !known.includes(n));
    if(bad.length) return { error:'unknown condition ' + bad.map(b => '"' + b + '"').join(', ') +
      '. This curriculum declares: ' + known.join(', ') };
  }
  return { ast, names:[...namesUsed(ast)] };
}
