// Every scene that can be opened by name: the Tab menu's, then any a checkout adds in localscenes.js.
// The address bar (?scenario=), the trainer, the headless tools and the tests resolve names here.
import { SCENARIOS } from './scenarios.js';
import { LOCAL_SCENES, localQuery } from './localscenes.js';

export const ALL_SCENARIOS = [...SCENARIOS, ...LOCAL_SCENES];

// A training run's address-bar parameters for the added scenes, applied before the scene is built; returns lines for the run log.
export function applyScenarioQuery(Q){ return localQuery(Q) || []; }
