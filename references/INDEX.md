# references: saved papers

Local copies of papers the project cites or reads for background. The
PDFs sit in this folder and are not committed (`references/*.pdf` is in
`.gitignore`): some are under licenses that allow reading but not
redistribution. This index is committed, so the folder can be rebuilt
from the links. README "Sources & inspirations" and the docs references
page list what the code cites; this file lists what is on disk.

## Cortical columns and the thousand brains theory

The Hawkins papers are theory ("Hypothesis and Theory" articles and
preprints), not measurements. Harris and Shepherd 2015 is the empirical
review to check them against. Haueis 2016 is the history of the column
concept, including the view that the column is not a functional unit.

| File | Citation | Source | License |
|---|---|---|---|
| `hawkins-2016-why-neurons-have-thousands-of-synapses.pdf` | Hawkins, J., Ahmad, S. (2016). Why neurons have thousands of synapses, a theory of sequence memory in neocortex. Frontiers in Neural Circuits 10:23. doi:10.3389/fncir.2016.00023 | https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2016.00023/full | CC BY |
| `hawkins-2017-columns-learn-structure-of-world.pdf` | Hawkins, J., Ahmad, S., Cui, Y. (2017). A theory of how columns in the neocortex enable learning the structure of the world. Frontiers in Neural Circuits 11:81. doi:10.3389/fncir.2017.00081 | https://pmc.ncbi.nlm.nih.gov/articles/PMC5661005/ | CC BY |
| `hawkins-2019-framework-grid-cells-neocortex.pdf` | Hawkins, J., Lewis, M., Klukas, M., Purdy, S., Ahmad, S. (2019). A framework for intelligence and cortical function based on grid cells in the neocortex. Frontiers in Neural Circuits 12:121. doi:10.3389/fncir.2018.00121 | https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00121/full | CC BY |
| `clay-2024-thousand-brains-project.pdf` | Clay, V., Leadholm, N., Hawkins, J. (2024). The Thousand Brains Project: a new paradigm for sensorimotor intelligence. arXiv:2412.18354 | https://arxiv.org/abs/2412.18354 | arXiv preprint |
| `leadholm-2025-thousand-brains-systems.pdf` | Leadholm, N., Clay, V., Knudstrup, S., Lee, H., Hawkins, J. (2025). Thousand-brains systems: sensorimotor intelligence for rapid, robust learning and inference. arXiv:2507.04494 | https://arxiv.org/abs/2507.04494 | arXiv preprint |
| `hawkins-2025-hierarchy-or-heterarchy.pdf` | Hawkins, J., Leadholm, N., Clay, V. (2025). Hierarchy or heterarchy? A theory of long-range connections for the sensorimotor brain. arXiv:2507.05888. The saved copy is the revision of 2026-08-20, retitled "The Thousand Brains Theory 2.0: an extension for the long-range connections of the neocortical heterarchy". | https://arxiv.org/abs/2507.05888 | arXiv preprint |
| `haueis-2016-life-of-the-cortical-column.pdf` | Haueis, P. (2016). The life of the cortical column: opening the domain of functional architecture of the cortex (1955 to 1981). History and Philosophy of the Life Sciences 38:2. doi:10.1007/s40656-016-0103-4 | https://pmc.ncbi.nlm.nih.gov/articles/PMC4914527/ | CC BY |

Not saved:

- Harris, K. D., Shepherd, G. M. G. (2015). The neocortical circuit:
  themes and variations. Nature Neuroscience 18(2), 170-181.
  doi:10.1038/nn.3917. The author manuscript is free to read at
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4889215/ but is not in the
  open access subset, so it cannot be fetched by script. Save it from a
  browser as `harris-2015-neocortical-circuit-themes-variations.pdf`.
- Hawkins, J. (2021). A Thousand Brains: A New Theory of Intelligence.
  Basic Books. A book, foreword by Richard Dawkins (he is not a
  co-author). No free copy exists. Its scientific content is in the
  2016, 2017 and 2019 papers above.

Two corrections to the reading list these came from: the item listed as
an undated Numenta whitepaper by Hawkins is the 2017 Frontiers paper by
Hawkins, Ahmad and Cui; and the 2019 framework paper has five authors
and is volume 12 (2018 volume year, published January 2019).

## Dendritic compartments and plasticity

| File | Citation | Source | License |
|---|---|---|---|
| `wright-2025-plasticity-rules-dendritic-compartments.pdf` | Wright, W. J., Hedrick, N. G., Komiyama, T. (2025). Distinct synaptic plasticity rules operate across dendritic compartments in vivo during learning. Science 388(6744), 322-328. doi:10.1126/science.ads4706 | https://www.science.org/doi/10.1126/science.ads4706 (author manuscript PMC12906676) | publisher copy, saved by Seth from the Science site |

## What the project takes from these

What each paper says, and where the mouse cortical column scene stands
against it. Nothing here is a build decision.

**Harris and Shepherd 2015.** The shared circuit across areas and
species: thalamic input to layer 4, layer 4 to layer 2/3, layer 2/3 to
layer 5; three classes of excitatory cell by projection target
(intratelencephalic, pyramidal tract, corticothalamic); interneurons in
three groups (Pvalb, Sst, Htr3a). The column scene has the layer order
and the Potjans and Diesmann 2014 pair table, a fast spiking and low
threshold spiking split for the interneurons, and bursting layer 5
pyramids. It does not separate the three excitatory projection classes,
and it has no Htr3a (VIP) group, a known limitation.

**Hawkins and Ahmad 2016.** A pyramidal cell as a pattern detector with
three zones: proximal input fires the cell, and a match on a basal or
apical dendritic segment depolarizes it without firing it, a prediction.
Fast local inhibition then lets the predicted cells fire first and
silence the rest. The engines run point neurons, so there is no
segment and no predicted state. The feedback inhibition pool is the
closest existing piece (winner selection by fast inhibition).

**Hawkins, Ahmad and Cui 2017.** A column as a two layer circuit: an
input layer (layer 4) that combines a sensory feature with a location
signal, and an output layer (layer 2/3) whose activity stays fixed for
one object while the sensor moves; long-range lateral connections
between output layers let columns settle on the same object. They name
layers 6 and 5 as a second instance of the same input and output pair.
The column scene has the layer 4 to layer 2/3 pathway at the published
probability. It has one column, no location signal, and no movement of
the sensor.

**Hawkins et al. 2019.** Proposes grid-cell-like cells in layer 6 of
every column (location of the sensed feature in the frame of the
object) and displacement cells in layer 5 (thick tufted cells). A
hypothesis (the 2017 paper says the location signal was deduced, not
observed). Nothing in the project models it.

**Hawkins, Leadholm and Clay 2025.** Roles for the long-range
connections: the thalamus converts orientation and movement from the
frame of the sensor to the frame of the object; feedforward, feedback
(layer 6a to layer 1 of the lower region) and cortico-thalamo-cortical
paths let columns learn objects made of other objects. The project has
a thalamocortical loop scene and a visual pathway scene; neither has
two cortical regions.

**Clay et al. 2024; Leadholm et al. 2025.** The software version
(Monty): learning modules in place of columns, a common message format
between them, voting across modules, Hebbian-like associative learning.
Not spiking and not a biophysical model. Useful as a statement of what
a column is supposed to compute, which is what a spiking column would
be tested against.

**Haueis 2016.** How the column went from Mountcastle's vertical
electrode tracks (1955) to Hubel and Wiesel's Nobel Prize (1981), and
that its status as an elementary unit is still disputed. A caution on
the word: the scene is a column by geometry (a 400 um wide block
through all layers with minicolumn strands), not a claim about a
functional unit.

**Wright, Hedrick and Komiyama 2025.** In layer 2/3 pyramidal cells of
mouse motor cortex during learning, apical synapses strengthen when
they are active together with neighbors within about 10 um, whether
or not the cell fires; basal synapses strengthen when their activity
coincides with the cell's action potentials, and blocking the cell's
spiking reduces basal potentiation and leaves apical plasticity
intact. Every plasticity rule in the project is a function of pre and
postsynaptic spike times, which is the basal rule. The apical rule
needs synapse position on a dendrite and a local coactivity term,
which a point neuron does not have.

The 2016 theory paper, the 2025 Science paper and the fly T4 result
(direction selectivity does not appear with point neurons) all point at the same missing piece, a dendritic compartment.
