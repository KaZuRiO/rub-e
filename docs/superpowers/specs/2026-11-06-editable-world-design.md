# Monde modifiable par chunks — conception

Date : 2026-11-06. Branche : `editable-world`, partie de `bench-scenarios`, qui n'est pas encore fusionnée dans `main`.

## Intention

Le moteur vise tous les genres de jeux, et le décor doit pouvoir changer pendant la partie, selon ce que chaque jeu veut faire. Aujourd'hui, les cubes statiques sont figés à la création de la `Scene`.

Ce projet est le socle des optimisations statiques à venir : retrait des cubes enfouis, puis meshes de chunks. Il n'apporte aucun gain de vitesse par lui-même. Il rend le décor modifiable et donne à ces optimisations un point unique où se brancher : la reconstruction d'un chunk.

Le décor change de deux façons, et les deux sont demandées :
- **bloc par bloc**, le joueur doit voir son geste tout de suite ;
- **par grandes zones**, comme une explosion, et l'image ne doit pas se figer.

## Objectifs

1. Un jeu peut poser, remplacer et retirer un bloc, et modifier une zone entière, pendant la partie.
2. Seuls les chunks touchés sont reconstruits et renvoyés au GPU.
3. Un chunk en cours de reconstruction reste affiché dans son ancienne version : jamais de trou transitoire.
4. Le benchmark mesure les deux façons de modifier le décor.
5. Les scènes qui ne modifient rien ne coûtent pas plus cher qu'aujourd'hui.

## Hors périmètre

- Retrait des cubes enfouis et meshes de chunks (projets suivants).
- Reconstruction sur un thread à part : elle viendra avec les meshes de chunks, quand reconstruire coûtera cher. Ici, reconstruire revient à recopier les instances d'un chunk.
- Sauvegarde et chargement du monde.
- Mise à jour de la physique et des collisions : `etib-scene` ne connaît pas la physique, c'est au jeu de la tenir à jour.
- Modification ou ajout de cubes hors grille à l'unité, en dehors du retrait par zone décrit plus bas.
- Agrandissement automatique des buffers GPU au-delà de la capacité déclarée.

## À ne pas faire

- Une explosion qui fige l'image, même si elle finit par s'afficher correctement.
- Un trou visible dans le décor pendant une reconstruction.
- Un coût par image qui dépend de la taille du monde quand rien ne change, au-delà de ce qu'il est aujourd'hui.
- Une reconstruction qui dépend du temps mesuré : le benchmark doit rester rejouable à l'identique.

## Contraintes

- Rust édition 2024, aucune nouvelle dépendance.
- Tests d'abord. Les tests qui demandent un GPU suivent le modèle de `etib_core::test_util::device` ; ils sont sautés sans GPU, sauf avec `ETIB_REQUIRE_GPU=1`.
- Tous les éléments publics sont documentés, comme dans le reste du dépôt.
- Machine de référence pour les mesures : MacBook Pro M5 Pro, sur secteur. Les timestamps GPU sur Metal restent à confirmer par le premier run.

## Le modèle

### Chunks

Les chunks passent de tuiles 32×32 de hauteur libre à des **cubes de 32×32×32**. Avec une hauteur libre, une tour très haute coûterait cher à reconstruire pour un seul bloc changé. La passe `chunk cull` et son `ChunkRaw` gardent leur rôle. Les tests `cubes_are_grouped_by_tile` et suivants s'adaptent au nouveau découpage.

### Cases et cubes hors grille

Un **bloc** occupe une case repérée par un `IVec3`. Un `ModelCube` dont la position est entière est un bloc de cette case.

Un cube dont la position n'est pas entière reste accepté à la création, comme dans `scattered` :
- il est rangé dans le chunk qui contient son centre ;
- il n'occupe aucune case : `set_block` et `remove_block` ne le voient pas ;
- `edit_region` lui applique la réponse de la case obtenue en arrondissant son centre avec `Vec3::round` : `Remove` le retire, `Keep` et `Set` le laissent en place, à côté du bloc éventuel. Une boule retire donc ceux de la boule, pas ceux des coins de la boîte, et un cube dont la case arrondie est hors de la boîte n'est pas touché.

### Doublons

Deux blocs dans la même case donnés à la création : le dernier gagne. `static_count` compte alors un seul cube, et le benchmark le montre par `kept_cubes` inférieur à `generated_cubes`.

## L'API de `Scene`

`Scene::new` reçoit un paramètre de plus, sur le modèle de `dynamic_capacity` :

```rust
pub fn new(
    device: &Device,
    static_cubes: &[ModelCube],
    static_capacity: usize, // cubes statiques au maximum, au moins static_cubes.len()
    dynamic_capacity: usize,
    width: u32,
    height: u32,
) -> Self

/// Pose ou remplace un bloc. Visible à la prochaine image.
pub fn set_block(&mut self, cell: IVec3, color: Vec3) -> Result<()>
/// Retire un bloc. Sans effet sur une case vide. Visible à la prochaine image.
pub fn remove_block(&mut self, cell: IVec3)
/// La couleur du bloc de cette case. Reflète toujours la dernière modification.
pub fn block(&self, cell: IVec3) -> Option<Vec3>
/// Applique `edit` à chaque case de la boîte, bornes comprises, en une seule opération.
pub fn edit_region(&mut self, min: IVec3, max: IVec3, edit: impl FnMut(IVec3) -> BlockEdit) -> Result<()>
/// Le budget de reconstruction des zones, en cubes par image.
pub fn set_rebuild_budget(&mut self, cubes: u32)
```

```rust
pub enum BlockEdit { Keep, Set(Vec3), Remove }
```

- **Capacité :** `set_block` et `edit_region` renvoient `Error::Capacity` si le total de cubes dépasserait `static_capacity`, sans rien modifier. `edit_region` est tout ou rien. Tant que le total reste dans la capacité, une modification ne doit jamais échouer, même si le buffer GPU est fragmenté.
- **Ce que compte la capacité :** uniquement le total côté CPU. Les anciennes versions encore affichées n'y entrent pas : voir « Plein à ras bord » plus bas.
- **Vérité côté CPU :** `block` et `static_count` reflètent la modification dès la fin de l'appel, même si le GPU affiche encore l'ancienne version du chunk. Le jeu lit donc toujours l'état réel.
- **Scène sans statiques :** avec une capacité supérieure à zéro, une scène créée vide peut recevoir des blocs. Avec une capacité nulle, `set_block` renvoie `Error::Capacity`.
- **Exemples :** chaque appel à `Scene::new` passe `cubes.len()` en capacité, et leur comportement ne change pas.

## Deux vitesses de reconstruction

Une modification marque le chunk à reconstruire, puis `Scene::render` fait le travail, avant les passes.

- **Priorité immédiate (`set_block`, `remove_block`) :** le chunk est reconstruit et envoyé dans l'image en cours, sans limite. Un jeu qui veut modifier beaucoup de blocs utilise `edit_region`. Ce choix est documenté sur `set_block`.
- **Par budget (`edit_region`) :** les chunks marqués sont reconstruits, les plus proches de la caméra d'abord, jusqu'à épuiser le budget de l'image. Un chunk commencé est toujours fini, et s'il dépasse à lui seul le budget, il passe quand même, un seul par image.
- **Déduplication :** un chunk marqué plusieurs fois n'est reconstruit qu'une fois. S'il passe en priorité immédiate, il quitte la file du budget.
- **Budget :** exprimé en cubes, et non en millisecondes, pour que le benchmark reste rejouable. Valeur de départ : 50 000 cubes par image, soit environ 4 Mo envoyés. Elle sera réglée avec les mesures sur Mac.

### Plein à ras bord

Les anciennes versions des chunks en attente occupent encore le buffer GPU. Le total CPU peut donc tenir dans la capacité alors que le GPU n'a plus de place, par exemple si une zone lointaine est retirée et une zone proche ajoutée. Quand un chunk n'a pas la place d'être écrit :

- **Par budget :** il attend, et les chunks en attente qui rétrécissent passent devant lui, quelle que soit leur distance.
- **Priorité immédiate :** les chunks en attente qui rétrécissent sont appliqués dans la même image, au-delà du budget, jusqu'à ce qu'il y ait la place. Une modification de bloc n'échoue donc jamais et reste visible tout de suite. Le prix est une image plus lente, seulement dans une scène pleine à ras bord.

Le budget n'est dépassé que dans ce cas. Un jeu qui veut l'éviter déclare une capacité un peu plus grande que son monde.

### Décalage entre chunks

Pendant une grande modification, deux chunks voisins peuvent être décalés de quelques images. C'est un compromis accepté.

Un chunk se met à jour en une fois : ses instances, sa ligne `ChunkRaw` et ses identifiants de chunk partent ensemble avant la passe `cull`. Une image ne voit jamais un chunk à moitié écrit.

La reconstruction passe par une seule fonction qui prend un chunk et produit ses instances. C'est là que se brancheront le retrait des cubes enfouis puis les meshes. Son résultat est déterministe : les mêmes blocs donnent les mêmes instances, dans le même ordre.

## Côté GPU

Aujourd'hui, `instances` et `chunk_ids` sont remplis une fois, tassés, et `visible` a exactement leur taille.

- `instances`, `chunk_ids` et `visible` sont créés pour `static_capacity` cubes.
- Chaque chunk possède une région de `instances` et peut en changer quand sa taille change. Une région plus grande que nécessaire évite de déplacer le chunk à chaque ajout.
- Les emplacements libres ne doivent jamais être dessinés. Solution que je suggère, sans toucher à `cull.wgsl` : réserver une ligne de chunk « vide », à `count == 0`, que `chunk_cull.wgsl` déclare toujours invisible, et faire pointer vers elle les emplacements libres.
- La passe `cull` tourne sur les emplacements utilisés jusqu'au dernier occupé, pas sur toute la capacité. Une scène sans modification tasse ses chunks au chargement et garde donc le coût de `cull` actuel.
- La table des chunks grandit à mesure que de nouveaux chunks apparaissent, dans une capacité fixée à la création, assez grande pour que `static_capacity` cubes ne l'épuisent jamais. Le constructeur décide comment la dimensionner sans gaspiller, en sachant que le pire cas est un cube par chunk.
- `StaticCubes::reset` et le dessin indirect ne changent pas. `Scene` crée `statics` dès que la capacité est non nulle.

La mémoire GPU suit `static_capacity`, à 80 octets par cube dans `instances` et autant dans `visible`, soit 32 Mo pour 200 000 cubes de capacité.

## Benchmark

Deux scénarios s'ajoutent à `examples/bench`, à ajouter aussi dans `Scenario` et dans `--scenario` :

- **`edit-blocks` :** le monde de `terrain`, avec 4 blocs de surface basculés à chaque image, chacun dans un chunk différent, près de la caméra. Basculer veut dire retirer un bloc présent ou reposer un bloc absent : le monde change vraiment d'une image à l'autre. Cela représente un jeu avec quelques joueurs ou personnages qui cassent et posent des blocs, soit environ 40 000 cubes renvoyés par image selon une estimation.
- **`edit-area` :** le monde de `terrain`, avec une boule de rayon 32 retirée par `edit_region` toutes les 120 images, puis reposée 60 images plus tard, devant la caméra. Ce rayon doit toucher assez de chunks pour étaler le travail sur plusieurs images au budget de départ. S'il ne le fait pas, le constructeur l'augmente.

Les deux sont des fonctions pures du numéro d'image, comme le reste de `scenario.rs`. La capacité statique est celle du monde plus 10 %, pour que le cas « plein à ras bord » ne fausse pas les mesures.

`Summary` gagne `max_ms`, en `Option<f32>` avec `#[serde(default)]` : un ancien fichier le relit comme non mesuré, pas comme 0. `--compare` l'affiche. Une explosion se voit dans l'image la plus lente, que la médiane et le p95 peuvent cacher.

`SCENARIO_VERSION` ne change pas : le contenu des quatre scénarios existants reste le même.

## Critères de réussite

Mesurés sur le Mac, par rapport aux références de `bench-scenarios`, sur `terrain`, `scattered`, `moving` et `few-changes` :

1. Médiane et p95 de `frame CPU` et `GPU total` à moins de 5 % de la référence. Le passage aux chunks 32³ peut déplacer légèrement `cull` : au-delà de 5 %, on regarde pourquoi avant de conclure.
2. `edit-blocks` : p95 de `frame CPU` à moins de 1 ms au-dessus de celui de `terrain`.
3. `edit-area` : image la plus lente à moins de 2 ms au-dessus de la plus lente de `terrain` joué avec le nouveau code, puisque les anciens fichiers n'ont pas `max_ms`. Toute la zone est affichée en moins de 10 images. Ce chiffre et le précédent sont provisoires.

Et, en tests automatiques :

4. Après toute suite de modifications, le contenu GPU d'un chunk reconstruit correspond à ce que donnerait une scène neuve créée avec les mêmes blocs.
5. Un chunk en attente de budget garde son ancien contenu GPU, et `block` renvoie déjà le nouveau.
6. Une modification qui dépasserait `static_capacity` renvoie `Error::Capacity` et ne change rien.
7. Remplir la capacité, tout retirer et tout reposer dans un autre ordre ne fait jamais échouer une modification, malgré la fragmentation.
8. Un bloc modifié sur la frontière de deux chunks ne reconstruit que le chunk qui le contient.

9. Dans une scène pleine à ras bord, retirer une zone lointaine, ajouter une zone proche de même taille puis poser un bloc ne renvoie aucune erreur, et le bloc est affiché à l'image suivante.

## Définition de « terminé »

- Le plan se termine quand les tests automatiques passent et que la relecture est faite : les critères 4 à 9 sont vérifiés ici.
- Les critères 1 à 3 se mesurent sur le Mac. La branche n'est fusionnée qu'après ces mesures.
- Les références sont à produire sur `bench-scenarios`, avant de lancer le nouveau code. Elles n'existent pas encore.
- Si un seuil provisoire est manqué, on ajuste le budget s'il suffit. Sinon, on décide ensemble entre revoir le seuil et ouvrir une optimisation à part, comme l'envoi d'une partie seulement d'un chunk, qui n'entre pas dans ce projet.

## Laissé au constructeur

- Le stockage des blocs dans un chunk : creux ou dense, tant qu'il reste déterministe.
- L'allocateur de régions et la façon de garantir l'absence d'échec par fragmentation, compaction comprise, tant qu'elle ne fige pas l'image.
- La marge d'une région et la façon de marquer les chunks.
- L'emplacement du code, dans `culling.rs` ou un nouveau module.
- Les positions exactes d'`edit-blocks` et le centre de la boule d'`edit-area`, dans les limites ci-dessus.

## Choix que j'ai faits, à vérifier

- Chunks 32×32×32 au lieu de 32×32 de hauteur libre.
- Budget en cubes par image, et non en millisecondes comme annoncé en discussion, pour garder un benchmark rejouable.
- Les chunks les plus proches de la caméra d'abord.
- Priorité immédiate sans limite pour `set_block`.
- Capacité statique déclarée à la création, comme pour les dynamiques, sans agrandissement automatique.
- Pas de modification à l'unité des cubes hors grille.
- Seuils de réussite provisoires : 5 %, 1 ms, 2 ms et 10 images.
- Quand le GPU est plein d'anciennes versions : les chunks qui rétrécissent passent devant, et le budget peut être dépassé pour une modification de bloc.
- Cubes hors grille dans `edit_region` : ils suivent la réponse de leur case arrondie.
- `edit-blocks` à 4 blocs dans 4 chunks plutôt que 16, et `edit-area` à rayon 32 plutôt que 12.
- Le projet s'arrête aux tests et à la relecture, les mesures Mac conditionnent la fusion, et aucune optimisation supplémentaire n'y entre.
