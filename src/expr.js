// A small expression language for the operations node.
//
// Deliberately not `new Function`.
// Two reasons, both load bearing:
//
//   Graphs are meant to be shared. A scene file is data, and a scene file
//   that can run arbitrary JavaScript the moment somebody opens it is not
//   data, it is a program. Nothing here can reach the page.
//
//   Determinism is a project rule: the same graph and seeds must always
//   compute the same network. This language has no clock, no Math.random and
//   no way to reach either, so an expression is a pure function of the
//   point it is given. `rand` is seeded from the point's stable identity,
//   so it is stable across rewiring and across machines.
//
// The grammar is the familiar arithmetic subset: numbers, names, the usual binary and unary operators with C precedence, a ternary, calls from a fixed table, and semicolon or newline separated assignments.
// Anything outside that is a parse error naming the offending token, which is the behavior a person editing a formula wants.

const FUNCS = {
  sin:[1,Math.sin], cos:[1,Math.cos], tan:[1,Math.tan],
  asin:[1,Math.asin], acos:[1,Math.acos], atan:[1,Math.atan],
  atan2:[2,Math.atan2], sqrt:[1,Math.sqrt], abs:[1,Math.abs],
  floor:[1,Math.floor], ceil:[1,Math.ceil], round:[1,Math.round],
  exp:[1,Math.exp], log:[1,x => Math.log(Math.max(1e-30, x))],
  sign:[1,Math.sign], pow:[2,Math.pow],
  min:[2,Math.min], max:[2,Math.max],
  clamp:[3,(v,lo,hi) => v < lo ? lo : v > hi ? hi : v],
  lerp:[3,(a,b,t) => a + (b-a)*t],
  step:[2,(edge,v) => v < edge ? 0 : 1],
  smoothstep:[3,(e0,e1,v) => {
    const t = Math.max(0, Math.min(1, (v-e0)/((e1-e0) || 1e-30)));
    return t*t*(3-2*t); }],
  // fill: noise and rand are bound per computation, since they need the seed
  noise:[3,null], rand:[0,null],
};

const KEYWORDS = new Set(['PI','E']);

function lex(src){
  const out = [];
  const isNum = c => c >= '0' && c <= '9';
  const isId0 = c => (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c === '_';
  const isId = c => isId0(c) || isNum(c);
  let i = 0;
  while(i < src.length){
    const c = src[i];
    if(c === '#' || (c === '/' && src[i+1] === '/')){
      while(i < src.length && src[i] !== '\n') i++;
      continue;
    }
    if(c === ' ' || c === '\t' || c === '\r' || c === '\n'){ i++; continue; }
    if(isNum(c) || (c === '.' && isNum(src[i+1]))){
      let j = i;
      while(j < src.length && (isNum(src[j]) || src[j] === '.')) j++;
      if(src[j] === 'e' || src[j] === 'E'){
        j++;
        if(src[j] === '+' || src[j] === '-') j++;
        while(j < src.length && isNum(src[j])) j++;
      }
      const text = src.slice(i, j);
      const v = Number(text);
      if(!isFinite(v)) throw new Error('not a number: ' + text);
      out.push({ t:'num', v }); i = j; continue;
    }
    if(isId0(c)){
      let j = i;
      while(j < src.length && isId(src[j])) j++;
      out.push({ t:'name', v:src.slice(i, j) }); i = j; continue;
    }
    const three = src.substr(i, 2);
    if(['==','!=','<=','>=','&&','||','**'].includes(three)){
      out.push({ t:'op', v:three }); i += 2; continue;
    }
    if('+-*/%<>!?:();,='.includes(c)){ out.push({ t:'op', v:c }); i++; continue; }
    throw new Error('unexpected character ' + JSON.stringify(c));
  }
  return out;
}

// precedence climbing; higher binds tighter
const BIN = {
  '||':1, '&&':2,
  '==':3, '!=':3, '<':4, '<=':4, '>':4, '>=':4,
  '+':5, '-':5, '*':6, '/':6, '%':6, '**':7,
};

function parse(src, readable, writable){
  const tok = lex(src);
  let p = 0;
  const peek = () => tok[p];
  const eat = (t, v) => {
    const k = tok[p];
    if(!k || k.t !== t || (v !== undefined && k.v !== v))
      throw new Error('expected ' + (v || t) + ' but found ' +
        (k ? (k.v !== undefined ? k.v : k.t) : 'end of expression'));
    p++; return k;
  };
  const isOp = v => tok[p] && tok[p].t === 'op' && tok[p].v === v;

  function primary(){
    const k = peek();
    if(!k) throw new Error('expression ended early');
    if(k.t === 'num'){ p++; return { k:'num', v:k.v }; }
    if(k.t === 'op' && (k.v === '-' || k.v === '!')){
      p++; return { k:'un', op:k.v, a:unary() };
    }
    if(k.t === 'op' && k.v === '+'){ p++; return unary(); }
    if(k.t === 'op' && k.v === '('){
      p++; const e = expr(0); eat('op', ')'); return e;
    }
    if(k.t === 'name'){
      p++;
      if(isOp('(')){
        const f = FUNCS[k.v];
        if(!f) throw new Error('no such function: ' + k.v);
        p++;
        const args = [];
        if(!isOp(')')){
          args.push(expr(0));
          while(isOp(',')){ p++; args.push(expr(0)); }
        }
        eat('op', ')');
        if(args.length !== f[0])
          throw new Error(k.v + ' takes ' + f[0] + ' argument' +
            (f[0] === 1 ? '' : 's') + ', got ' + args.length);
        return { k:'call', name:k.v, args };
      }
      if(KEYWORDS.has(k.v)) return { k:'const', v:k.v === 'PI' ? Math.PI : Math.E };
      if(!readable.has(k.v))
        throw new Error('unknown name: ' + k.v);
      return { k:'var', name:k.v };
    }
    throw new Error('unexpected ' + (k.v !== undefined ? k.v : k.t));
  }
  function unary(){ return primary(); }
  function expr(minp){
    let lhs = unary();
    for(;;){
      const k = peek();
      if(!k || k.t !== 'op') break;
      if(k.v === '?'){
        if(minp > 0) break;
        p++;
        const a = expr(0); eat('op', ':'); const b = expr(0);
        lhs = { k:'tern', c:lhs, a, b };
        continue;
      }
      const pr = BIN[k.v];
      if(pr === undefined || pr < minp) break;
      p++;
      // ** is right associative, everything else left
      const rhs = expr(k.v === '**' ? pr : pr + 1);
      lhs = { k:'bin', op:k.v, a:lhs, b:rhs };
    }
    return lhs;
  }

  // Statements are separated by a semicolon or simply by starting a new assignment, which is what makes one-per-line work without the lexer having to keep newlines and without a formula being forbidden from wrapping across lines.
  // A statement starts at `name =`, and `==` is a single token, so a comparison can never be mistaken for one.
  const startsStmt = () => tok[p] && tok[p].t === 'name' &&
    tok[p+1] && tok[p+1].t === 'op' && tok[p+1].v === '=';
  const stmts = [];
  while(p < tok.length){
    if(isOp(';')){ p++; continue; }
    const name = eat('name').v;
    if(!writable.has(name)){
      if(readable.has(name))
        throw new Error(name + ' can be read but not assigned (writable: ' +
          [...writable].join(', ') + ')');
      throw new Error('cannot assign to ' + name + ' (writable: ' +
        [...writable].join(', ') + ')');
    }
    eat('op', '=');
    const value = expr(0);
    stmts.push({ name, value });
    if(p < tok.length && !isOp(';') && !startsStmt())
      throw new Error('expected the end of the assignment to ' + name +
        ' but found ' + (tok[p].v !== undefined ? tok[p].v : tok[p].t));
  }
  if(!stmts.length) throw new Error('no assignments; write something like  y = y + 10');
  return stmts;
}

// Compiles the AST to a closure over an environment object.
// The environment is a plain object of numbers, refilled per point by the caller, so no allocation happens inside the loop.
function compile(node, env, fns){
  switch(node.k){
    case 'num': case 'const': { const v = node.v; return () => v; }
    case 'var': { const nm = node.name; return () => env[nm]; }
    case 'un': {
      const a = compile(node.a, env, fns);
      return node.op === '-' ? () => -a() : () => (a() ? 0 : 1);
    }
    case 'bin': {
      const a = compile(node.a, env, fns), b = compile(node.b, env, fns);
      switch(node.op){
        case '+': return () => a() + b();
        case '-': return () => a() - b();
        case '*': return () => a() * b();
        case '/': return () => { const d = b(); return d === 0 ? 0 : a()/d; };
        case '%': return () => { const d = b(); return d === 0 ? 0 : a()%d; };
        case '**': return () => Math.pow(a(), b());
        case '<': return () => a() < b() ? 1 : 0;
        case '<=': return () => a() <= b() ? 1 : 0;
        case '>': return () => a() > b() ? 1 : 0;
        case '>=': return () => a() >= b() ? 1 : 0;
        case '==': return () => a() === b() ? 1 : 0;
        case '!=': return () => a() !== b() ? 1 : 0;
        case '&&': return () => (a() && b()) ? 1 : 0;
        case '||': return () => (a() || b()) ? 1 : 0;
      }
      throw new Error('unhandled operator ' + node.op);
    }
    case 'tern': {
      const c = compile(node.c, env, fns), a = compile(node.a, env, fns),
            b = compile(node.b, env, fns);
      return () => c() ? a() : b();
    }
    case 'call': {
      const args = node.args.map(x => compile(x, env, fns));
      const f = fns[node.name] || FUNCS[node.name][1];
      switch(args.length){
        case 0: return () => f();
        case 1: return () => f(args[0]());
        case 2: return () => f(args[0](), args[1]());
        case 3: return () => f(args[0](), args[1](), args[2]());
      }
      throw new Error('too many arguments to ' + node.name);
    }
  }
  throw new Error('unhandled node ' + node.k);
}

// Public entry.
// `readable` and `writable` are Sets of names; `fns` supplies the bindings for noise and rand, which need the computation's seed.
// Returns
// { env, run }: fill env with the point's values, call run(), read the
// writable names back out.
// Throws on a bad expression, with a message that names what went wrong.
export function buildExpr(src, readable, writable, fns){
  const stmts = parse(src, readable, writable);
  const env = {};
  for(const nm of readable) env[nm] = 0;
  const compiled = stmts.map(s => ({ name:s.name, run:compile(s.value, env, fns) }));
  return { env, assigned:[...new Set(compiled.map(s => s.name))],
    run(){ for(let i=0;i<compiled.length;i++) env[compiled[i].name] = compiled[i].run(); } };
}

const EXPR_FUNCS = Object.keys(FUNCS);
