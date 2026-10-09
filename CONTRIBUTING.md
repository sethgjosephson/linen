# Contributing

Contributions are welcome. One process requirement comes first, because
it cannot be applied retroactively.

## Contributor License Agreement

Before a pull request can be merged, contributors must sign the
[Contributor License Agreement](CLA.md).

**Why this exists.** linen is released under the AGPL, and
the copyright holder also reserves the ability to offer separate
commercial licenses and to relicense the project in future, for example
if a funder, journal, or institutional partner requires a specific
license. Doing either requires holding the rights to every line in the
project. If a contribution is merged without a CLA, the project can no
longer be relicensed without tracking down that contributor and getting
their permission, and one unreachable contributor can freeze the
project's licensing permanently.

A Developer Certificate of Origin is not sufficient for this. A DCO
certifies that a contributor had the right to submit their code. It does
not grant the project the right to license that code under different
terms later. The CLA does.

The CLA does **not** take your copyright away. You keep it. You grant a
broad license alongside it.

A check on every pull request looks for the signature (the sentence in
the CLA's signing section, posted as a comment). One signature covers
every later pull request from the same account.

## Third-party code

Code from elsewhere may come in only under a permissive license (MIT,
BSD, ISC, Apache-2.0), with its license header kept and its origin named
in a comment. Code under a copyleft license (GPL, LGPL, AGPL, CeCILL and
the like) cannot be merged, however small the piece, because a derivative
of it could not be relicensed. Equations and parameter values from a
paper or another simulator are fine and are cited where they are used;
the implementation is written here.

## Making a change

1. Open an issue first for anything beyond a small fix, so the design
   can be discussed before the work.
2. Read `EXTENDING.md` for how a node, a mechanism or a measure is added
   and what is frozen, and `ENGINE.md` for the simulation engine
   contract. Changes to the engine protocol require updating `ENGINE.md`,
   all three engines, and the validation battery in the same commit.
3. Run the validation battery. Statistical changes to network behavior
   need to be justified against it, not merely explained.
4. No build step and no runtime dependencies. The project is served as
   static files and vendors Three.js. Please keep it that way.

## Scientific claims

Anything that changes simulation behavior needs a source. The
convention throughout the project is that published models are cited
where they are implemented, and that limitations are documented rather
than smoothed over. See the sources section of `README.md` for the
existing citations and the tone expected.
