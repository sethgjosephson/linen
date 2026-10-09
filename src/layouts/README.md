Saved scenario layouts, one file per scenario (the scene menu writes them).
A file is a scenario's graph as someone arranged it: positions, groups,
every wire, the nodes added and removed since the scenario was opened.
It is keyed by the order the scenario builds its nodes and refused when
the builder's nodes have changed, so a changed scene falls back to the
computed layout until its file is saved again. Settings on the scenario's
own nodes come from the code, not the file.
