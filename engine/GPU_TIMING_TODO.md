# Mesure GPU : points ouverts

Points relevés par la revue de la branche `gpu-timing-bench`, laissés en
l'état faute de vrai GPU pour les vérifier : tout a été testé sur lavapipe
(Vulkan logiciel, sous Xvfb), dont les chiffres ne veulent rien dire.

## À faire en premier, sur une machine avec GPU

Depuis `engine/` :

```text
cargo run --release --bin bench -- --save a
cargo run --release --bin bench -- --save b
cargo run --release --bin bench -- --compare a b
cargo run --release --bin hello -- --no-vsync
```

- Si `--compare` affiche `note: run a has GPU timings for only N of 600
  measured frames`, les points 1 et 2 ci-dessous se produisent réellement.
- Vérifier aussi que deux runs du même code donnent des écarts faibles, que
  redimensionner la fenêtre pendant un run affiche « changed size » sans écrire
  de fichier, et que le titre de `hello` montre `gpu X.XX ms`.

## 1. Plusieurs résultats GPU dans une frame : seul le dernier est gardé

`crates/etib/src/app.rs:216` — `poll_timings().pop()` ne garde que le résultat
le plus récent de chaque relecture (ce que demandait la spec). Quand le GPU
termine deux frames entre deux relectures, la plus ancienne est perdue. Le
benchmark ne reçoit alors jamais ses 600 mesures, attend les 120 frames de
grâce, et enregistre un échantillon partiel qui sous-représente justement les
frames irrégulières.

**Piste :** transmettre à l'app tous les résultats arrivés, par exemple
`FrameStats::gpu_arrived: &'a [GpuTimings]`, en plus (ou à la place) du plus
récent.

## 2. Seulement 4 emplacements de mesure

`crates/etib-core/src/timing.rs:25` — `TIMING_SLOTS = 4`. Un emplacement n'est
libéré que lorsqu'une relecture, au début d'une frame suivante, voit son
résultat prêt. Sans vsync et limité par le GPU, le CPU prend plusieurs frames
d'avance : chaque emplacement reste occupé 3 à 5 frames, `Timer::lease`
renvoie `None` (journalisé seulement en debug) et ces frames ne sont pas
mesurées.

**Piste :** passer à 8 (coût négligeable), et journaliser au moins une fois en
`warn` quand une frame n'a pas pu être mesurée.

## 3. `FrameStats::gpu` recopié à chaque frame

`crates/etib/src/app.rs:273` — `self.last_gpu.clone()` copie le `Vec` des passes
et une `String` par label à chaque frame, même pour les apps qui ne lisent pas
`stats.gpu` : jusqu'à ~33 allocations par frame, qui ajoutent du bruit au temps
CPU que mesure le benchmark.

**Piste :** `pub gpu: Option<&'a GpuTimings>` (`FrameStats` a déjà la durée de
vie `'a`). C'est un changement d'API publique : plus simple avant la fusion
qu'après. Sites à adapter : `examples/hello/main.rs`, `examples/bench/main.rs`,
et le test `frame_stats_carry_the_latest_gpu_timings`.

## 4. Relecture bloquée si un résultat n'arrive jamais

`crates/etib-core/src/timing.rs:179` — `Timer::collect` s'arrête au premier
résultat en attente. Si son `map_async` ne se termine jamais (perte du
périphérique, soumission en erreur), tous les suivants restent bloqués, les 4
emplacements finissent occupés, plus rien n'est mesuré et `FrameStats::gpu`
reste figé, sans message au-dessus du niveau debug.

**Piste :** abandonner un résultat en attente depuis trop de frames (libérer
son emplacement) et le signaler en `warn`.
