// Cloudflare Worker in front of every static asset request (run_worker_first in wrangler.toml routes all requests through here).
//
// SITE_OPEN, a plain variable in wrangler.toml, decides whether the site is public.
// "1": every request is served.
// Anything else: HTTP Basic Auth with the SITE_PASSWORD secret set in the Cloudflare dashboard (project > Settings > Variables and Secrets); visitors enter any username and the shared password in the browser prompt, and the site refuses to serve if the secret is unset rather than opening by accident.
// Flipping SITE_OPEN in wrangler.toml and pushing is how the gate goes on or off, with no dashboard step.
export default {
  async fetch(request, env){
    if(String(env.SITE_OPEN) === '1') return env.ASSETS.fetch(request);
    const expected = env.SITE_PASSWORD;
    if(!expected)
      return new Response('SITE_PASSWORD is not configured; refusing to serve.', { status: 503 });
    const auth = request.headers.get('Authorization') || '';
    if(auth.startsWith('Basic ')){
      try {
        const given = atob(auth.slice(6)).split(':').slice(1).join(':');
        if(given === expected) return env.ASSETS.fetch(request);
      } catch(e){}
    }
    return new Response('Password required.', {
      status: 401,
      headers: { 'WWW-Authenticate': 'Basic realm="linen"' },
    });
  },
};
