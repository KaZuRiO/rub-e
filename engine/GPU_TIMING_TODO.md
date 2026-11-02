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

## Ouvert : `post` mesuré à 0 sur Metal

Le timestamp de fin de la passe `post` arrive plus tôt que son début (de 3 à
15 ms) : la soustraction le ramène à 0, et `GPU total`, qui s'arrête alors à
la fin de `scene`, ne compte pas `post`. La valeur lue ressemble à la fin de
`post` d'environ deux frames plus tôt, comme si la fin n'était pas encore
écrite quand on relit l'emplacement.

Écarté : l'ordre dans le command buffer (relecture dans un command buffer
séparé, même résultat) et le fait que `post` soit la dernière passe (une passe
ajoutée après, même résultat). Piste restante : `post` est la seule passe qui
dessine dans l'image de la fenêtre.
