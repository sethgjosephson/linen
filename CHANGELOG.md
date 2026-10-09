# Changelog

One line per user-visible change, per release, newest first. The list
starts at the public release.

A release tag names the app. The files the app writes carry their own
numbers, checked on read and migrated where a migration exists, and a
release lists the number of each that it reads and writes:

| file | number | where it is checked |
|---|---|---|
| scene (`scenes/*.json`) | format 13 | `src/migrate.js`, every step from an older format runs on load |
| project (`project.json`) | format 1 | `src/project.js` |
| brain (`brains/*.npb`) | NPB1, NPB2, NPB3, version 1 | `src/brain.js` |
| scenario layout (`src/layouts/*.json`) | format 3 | `src/layoutfile.js` |
| recording (`recordings/*.json`) | linen-recording-1 | `src/recording.js` |
| trial matrix (`runs/*/trials.npt`) | np-trials-1 | `src/train.js`, `tools/np_trials.py` |
| headless run folder (`run.json`, `rates.csv`, `spikes.txt`, `cells.csv`, `settings.json`) | as written by `tools/linen.mjs` | `tools/linen_read.py` |
| engine init message | protocol 2 | `src/protocol.js`, refused on mismatch by every engine |

## Unreleased

The first public release.
