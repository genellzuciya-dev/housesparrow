# Eight Hours With a Sparrow

### What "doing machine learning" actually looks like when you're not doing machine learning

---

## Skills applied

| Skill | What I did with it |
|---|---|
| **AI Modeling** | |
| Sampling bias correction | Replaced randomly placed absence points with target-group background sampling, drawn from other species' records, so the model measured habitat rather than where birdwatchers go. |
| Feature interpretation | Separated feature importance, which variable a model relies on, from effect direction, whether the animal prefers more or less of it. |
| Evaluation judgment | Read a drop in accuracy from 0.73 to 0.63 correctly — as evidence the corrected model was solving the harder, right problem. |
| **AI Engineering** | |
| Reproducible experiments | Built the bias correction as a configuration flag, so both versions run from identical code and can be compared directly. |
| Pipeline staging | Separated slow satellite processing from fast rendering, so changing a layout takes seconds instead of re-running twelve months of imagery queries. |
| Automated diagnostics | Wrote reports that print effect direction and seasonal breakdowns on every run, so the model's claims get checked without anyone remembering to. |

**Stack** — Python · scikit-learn · Google Earth Engine · Sentinel-2 · Landsat 8 · NLCD · GBIF and eBird APIs · Folium · Git / GitHub Pages

---

I spent a day building a model that predicts where house sparrows are likely to be in Albuquerque, month by month, across a full calendar year. The output is two interactive maps: one that animates January through December, one that lets you toggle seasons side by side.

Here's the honest accounting of that day. Roughly ten lines of it were machine learning. The rest was learning to ask satellites questions.

That gap is the whole story, and I think it's the most useful thing I can hand to anyone else standing where I was standing this morning.

---

## The setup: a lot of birds, badly distributed

I started where you'd expect — with data that already existed. GBIF (the Global Biodiversity Information Facility) and eBird between them hold millions of bird observations contributed by researchers and hobbyists over decades. For house sparrows in the Albuquerque basin, that's thousands of dated, geolocated sightings going back years.

The naive move is to pull all of it and train on the pile. I did that first. It produced a map. The map was useless, and it took me a while to articulate why: **a sightings dataset is not a distribution.** It's a record of where people looked and happened to write it down. Every point in that dataset is the intersection of a bird being somewhere *and* a human being there with a phone.

That's not a footnote. It's the central methodological problem of the entire field, and I'd walked straight into it on version one.

---

## Baseline first, weaknesses out loud

I built the simplest thing that would run end to end: pull occurrences, generate pseudo-absence points, attach some environmental values to each, fit a RandomForest, score a grid, draw a map.

The environmental values in that first version were **synthetic** — mathematical surfaces I generated to stand in for real terrain while I got the plumbing working. Reasonable scaffolding. Except when I rendered the output, the suitability surface came out as a perfect radial ripple, clean concentric rings spreading from the city center.

Sparrows do not live in concentric rings.

I want to sit on this moment, because it's the one I'd most want a hiring manager or a collaborator to see. The model wasn't broken. It fit beautifully. It had faithfully learned the structure of my placeholder math and reported it back to me as ecology, with confidence. The only thing standing between that artifact and a published map was **knowing what a sparrow distribution should not look like.**

Domain knowledge isn't a nice complement to modeling. It's the error-detection layer. Nothing in the training loop was ever going to flag that.

---

## From placeholder to planet

Killing the synthetic surfaces meant going to actual satellites, and this is where the day really went.

Through Google Earth Engine, I built eight environmental features from live imagery:

| Feature | Source | What it senses |
|---|---|---|
| NDVI | Sentinel-2 (B8−B4) | Vegetation greenness |
| NDWI | Sentinel-2 (B3−B8) | Water and moisture |
| NDBI | Sentinel-2 (B11−B8) | Built-up surface, spectrally |
| NDRE | Sentinel-2 (B8−B5) | Vegetation health via red edge |
| SAVI | Sentinel-2 | Vegetation, corrected for bare soil |
| Land surface temp | Landsat 8 (ST_B10) | Actual ground temperature, °C |
| Impervious surface | NLCD 2021 | Pavement and rooftop fraction |
| Built density | NLCD 2021 | Development intensity |

Each is a different lens on the same landscape. Two of them see vegetation and disagree about it. Two see the built environment through completely different physics — one spectral, one from a classified land-cover product.

**This is the part that actually took the day.** Not architecture selection, not hyperparameters. Learning which band combination answers which question, why SAVI exists when you already have NDVI, how to composite a month of Sentinel passes into one usable image, and how to calibrate a Landsat thermal band into degrees.

If you came up through optics — I spent years at Imagineering building 3D LUT color pipelines, characterizing how light behaves through a projection chain — this is deeply familiar territory wearing unfamiliar clothes. Spectral indices are just band math on a sensor with a known response curve. I already had that muscle. I'd never once thought to point it at a satellite.

---

## Making time a first-class variable

The first working version gave me one map. One static prediction for "where sparrows are."

But sparrows don't experience an average year, and the drivers don't hold still either. Albuquerque's NDVI in June and NDVI in January are describing two different cities. Averaging across that doesn't produce a general model — it produces a model of a season that never happens.

So I restructured around time. Twelve separate GBIF queries, one per calendar month, so no month gets swamped by its neighbors. Sightings routed to their true calendar month. **Twelve models**, each trained on satellite imagery matched to its own month. Then the whole thing stamped with real dates and pushed through a time-dimension layer so you can press play and watch the year run.

The animation isn't decoration. It's the finding. Suitability doesn't just intensify and fade — it *relocates*, and you can watch it move.

---

## What the model actually said

The first version of this section was wrong, and how it got fixed is the most useful thing in this write-up.

Feature importance ranks land surface temperature first in every season. But importance only reports that a variable *matters* — it says nothing about direction. To get direction you compare each driver at sparrow locations against your background points, and that means the choice of background points decides the answer.

Originally those were scattered at random across the bounding box. That sounds neutral. It isn't: random points land on unsurveyed desert mesa, so the model can score well by learning **where birders go** rather than where sparrows live. Replacing them with target-group background — drawn from other bird species' actual records, places somebody demonstrably stood and looked — changed the result:

| Driver | Random background | Target-group background |
|---|---|---|
| Built density | +0.61 | +0.47 |
| Impervious surface | +0.41 | +0.29 |
| Land surface temp | **−0.32** | **+0.35** |
| NDVI greenness | +0.39 | +0.09 |
| SAVI | +0.38 | +0.16 |
| NDBI built-up | −0.54 | −0.16 |

Temperature reversed sign. The sparrow values never moved — same birds, same places — but background temperature dropped from 30.4 °C to 28.3 °C, because unsurveyed desert is hot and surveyed places aren't. Relative to bare mesa, sparrows looked cool-seeking. Relative to where people actually go, they're in the warm spots. And the vegetation preference, which had looked strong, mostly evaporated: birders go to green places.

**The first answer was largely a map of birdwatchers.**

### The real result: two behaviours, not one

Splitting direction by season shows something an annual average hides completely:

| | winter | spring | summer | fall |
|---|---|---|---|---|
| Land surface temp | +0.36 | +0.42 | +0.28 | +0.35 |
| NDVI greenness | **+0.41** | **−0.19** | +0.08 | +0.06 |
| NDRE veg health | +0.42 | −0.13 | +0.14 | +0.14 |
| SAVI | +0.46 | −0.11 | +0.13 | +0.15 |

The thermal association is nearly flat across the year. In a city where July ground temperatures are punishing, these birds never move to the cool blocks.

Vegetation does the opposite. Sparrows crowd into green in winter and ignore it by spring. That reads as scarcity: in a high-desert winter the only green is irrigated or evergreen, so it's rare and worth competing for; by spring the whole basin greens up, green stops distinguishing anywhere from anywhere, and the signal dissolves. (Those four indices come from overlapping bands and are heavily correlated — one signal confirmed four ways, not four independent findings.)

Summer flattens everything else. Impervious lands at exactly 0.00, built density falls to +0.18 from +0.62 in spring, and summer AUC is the year's weakest at 0.630. For three months the sparrows are nearly indistinguishable from every other bird in town — except that they're still in the warmer places.

Seasonal AUC ran 0.63–0.81 and fell in summer and fall when target-group background replaced random. That drop is the correct direction: the easy wins from empty desert disappeared and the model had to do real work.

## Points of expansion

Everything above is a first day. Here's what would make it real, roughly in order of how much it would change the answer:

**1. Explain the summer collapse.** Summer AUC is 0.630 and nearly every driver flattens toward zero. Either the birds genuinely disperse after breeding, or three months of peak birding saturate the map so thoroughly that sparrow records and target-group records describe the same places. Those are very different findings and the current data can't separate them.

**2. Use eBird complete checklists.** The stronger version: zero-fill complete checklists into genuine detection/non-detection data, carrying duration, distance traveled, observer count, and time of day as effort covariates. Train with them, hold them fixed at prediction time. That converts the whole thing from presence-only inference to something much closer to a real survey.

**3. Spatially blocked cross-validation.** Random splits on clustered citizen-science points leak badly — neighboring records land in both folds and every feature looks brilliant. Spatial blocking gives an honest generalization estimate.

**4. Permutation importance on held-out data.** Replaces the biased impurity measure, and lets me report confidence bands across bootstrap draws instead of single point estimates per season.

**5. Fixed suitability thresholds.** If the prime/suitable/marginal bins are computed per-month, "prime" means something different in December than in June, and the animation would imply a stability that isn't there. Global thresholds, or publish the cutoffs.

**6. The cloud confound.** Winter composites are built from fewer usable Sentinel scenes than summer ones. Summer gets both more birds *and* cleaner imagery — two biases pointing the same direction. Worth quantifying before trusting any seasonal contrast.

**7. A second species.** House sparrow is a generalist commensal. Running the identical pipeline on a specialist — something piñon-juniper dependent — and comparing which spectral lens each one loads onto would turn a single map into a comparative finding.

**8. The design question.** This is where it goes next for me. If what a synanthropic bird tracks in a desert city is the *irrigated, planted, shaded* fraction of the built environment, then a suitability surface is also a map of **where the city is successfully buffering its own heat** — and its inverse is a map of where it isn't. Those are the blocks where people suffer in July. That's a biomimicry brief hiding inside a species distribution model, and it's the version of this work I actually want to build.

---

## What I'd tell someone starting tomorrow

**Your first model will be fluent and wrong, and it will not tell you.** The synthetic-ripple map was internally consistent and confidently rendered. Only knowing the organism caught it.

**Budget your time honestly.** If you think you're spending the day on machine learning, you're going to spend it on coordinate systems, API authentication, band math, and date handling instead. That's not the tax on the real work. That *is* the real work.

**Bring the expertise you already have.** Spectral index math is band math is color science. Prior domain fluency is the thing that lets you smell a wrong answer, and it transfers further than it looks like it will.

**Say what's broken.** The version of this writeup that lists no weaknesses is less trustworthy than the one you just read, and considerably less useful to anyone building on it.

---

*Code and both interactive maps in this repo. Built in a day; wrong in at least eight documented ways, listed above.*
