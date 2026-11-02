# Mesure GPU : points ouverts

Points relevés par la revue de la branche `gpu-timing-bench`, laissés en
l'état faute de vrai GPU pour les vérifier : tout a été testé sur lavapipe
(Vulkan logiciel, sous Xvfb), dont les chiffres ne veulent rien dire.

## Corrigé (vérifié sur Apple M5 Pro, Metal)

Deux runs du même code donnent maintenant 600 mesures GPU sur 600, à moins de
0,5 % près sur `scene` et sur le temps CPU médian.

1. Tous les résultats arrivés sont transmis : `FrameStats::gpu_arrived`, que
   `bench` lit en entier.
2. 8 emplacements de mesure au lieu de 4, et un `warn` (une fois) quand ils
   sont tous occupés.
3. `FrameStats::gpu` est un `Option<&GpuTimings>` : plus de copie par frame.
4. Un résultat en attente ne bloque plus les suivants ; il est signalé en
   `warn` au bout de 120 relectures et garde son emplacement.

Les runs faits avant ces corrections perdaient la moitié des frames, surtout
celles où le GPU était en retard : leurs chiffres sont trop optimistes (p95 de
`cull` 0,36 ms au lieu de 0,64) et ne se comparent pas aux nouveaux.

## `post` non mesurable sur Metal

Cause trouvée : sur Metal, une passe qui dessine dans l'image de la fenêtre
n'a pas de timestamp de fin valable au moment de la relecture. La valeur lue
est celle d'une frame précédente, le plus souvent antérieure au début.
Dessinée dans une texture hors fenêtre, la même passe `post` se mesure
normalement : environ 0,24 ms par frame sur M5 Pro, soit à peu près 13 % du
GPU total.

Ce qui est fait :
- `PassTime::ms` vaut `None` quand la fin tombe avant le début, au lieu de 0 ;
  `total_ms` ne compte que les passes mesurées (il s'arrête donc à la fin de
  `scene`, sans `post`).
- `--compare` affiche `—` pour une passe sans temps sur au moins la moitié des
  frames, avec une note : une fin périmée qui tombe par hasard après le début
  ne se distingue pas d'une vraie mesure (1 frame sur 600 dans un run).
- Les runs enregistrés avant gardent leur faux `0.00` : les refaire.

Reste ouvert : le signaler à wgpu ; et, si un jour il faut mesurer `post`
sur Metal, dessiner dans une texture puis la copier dans la fenêtre (une
copie plein écran par frame, à réserver au bench).
