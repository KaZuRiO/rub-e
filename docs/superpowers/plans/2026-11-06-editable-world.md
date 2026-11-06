# Monde modifiable par chunks — plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** permettre aux jeux de modifier le décor statique bloc par bloc et par zones, sans figer l'image, et mesurer les deux cas.

**Architecture :** `World` est la vérité côté CPU : des chunks 32³ qui connaissent leurs blocs et ce qui reste à reconstruire. `plan_rebuilds` choisit, à chaque image, les chunks à appliquer. `StaticCubes` les écrit dans un buffer GPU à capacité fixe, où les emplacements d'un chunk peuvent être dispersés, ce qui supprime tout problème de fragmentation.

**Tech Stack :** Rust édition 2024, wgpu via `etib-core`, WGSL, `glam`, `serde`, `clap`.

**Spec :** `docs/superpowers/specs/2026-11-06-editable-world-design.md`

## Global Constraints

- Aucune nouvelle dépendance.
- Tests d'abord. Tests GPU sur le modèle `let Some(device) = device() else { return };`.
- Tous les éléments publics sont documentés. Commentaires et messages de commit en anglais.
- Budget de départ : `50_000` cubes par image.
- Mêmes blocs, mêmes instances, dans le même ordre : pas de `HashMap` dont l'ordre atteint le résultat.
- Scène sans modification : aucun travail de reconstruction par image.
- `SCENARIO_VERSION` reste à 1.
- Tâche finale exclue, les commandes partent de `engine/`. `<scene-test>` signifie `ETIB_REQUIRE_GPU=1 cargo test --offline -p etib-scene`, et `<bench-test>` signifie `cargo test --offline -p etib-examples --bin bench`.
- Dans l'environnement cloud, chaque commande passe dans
  `nix-shell -p alsa-lib udev xorg.libX11 xorg.libXcursor xorg.libXi xorg.libXrandr libxkbcommon vulkan-loader mesa.drivers pkg-config`,
  avec `CARGO_TARGET_DIR` hors de `/home`, `LD_LIBRARY_PATH` vers ces bibliothèques et `VK_ICD_FILENAMES` vers `lvp_icd.x86_64.json` de `mesa.drivers`.

## Review Focus

1. **`set_rebuild_budget(0)` :** une zone modifiée doit quand même finir par s'afficher, à un chunk par image. Test `a_budget_of_zero_still_makes_progress` (tâche 4).
2. **Coordonnées négatives :** la case `(-1, 0, 0)` est dans le chunk `[-1, 0, 0]`, pas `[0, 0, 0]`. Test `negative_cells_fall_in_negative_chunks` (tâche 2).
3. **Boîte inversée dans `edit_region`, `min > max` :** ne fait rien, renvoie `Ok`, n'appelle pas `edit`. Test `an_inverted_box_changes_nothing` (tâche 2).
4. **Remplacer un bloc à pleine capacité :** réussit, car le total ne change pas. Test `replacing_a_block_at_capacity_succeeds` (tâche 2).
5. **Tout retirer puis reposer :** une scène vidée se dessine sans erreur, et on peut y remettre des blocs. Test `a_scene_emptied_and_refilled_matches_a_fresh_one` (tâche 6).

## Fichiers

| Fichier | Rôle |
|---|---|
| `crates/etib-scene/src/world.rs` (nouveau) | Blocs par chunk, capacité, chunks à reconstruire. Aucun GPU |
| `crates/etib-scene/src/slots.rs` (nouveau) | Emplacements libres du buffer d'instances. Aucun GPU |
| `crates/etib-scene/src/rebuild.rs` (nouveau) | Quels chunks appliquer à cette image. Aucun GPU |
| `crates/etib-scene/src/culling.rs` | Buffers à capacité fixe, application d'un chunk |
| `crates/etib-scene/src/scene.rs` | API publique, reconstruction dans `render` |
| `crates/etib-scene/src/shaders/chunk_cull.wgsl` | Ligne de chunk vide toujours invisible |
| `examples/bench/*` | `max_ms`, `rebuild_frames`, `edit-blocks`, `edit-area` |
| `ARCHITECTURE.md` | Lignes 667 et 932, plus les nouveaux fichiers |

---

### Task 1 : Chunks 32×32×32

**Files :** Modify `crates/etib-scene/src/culling.rs`

**Interfaces :**
- Produces : `pub(crate) type ChunkKey = [i32; 3];` et `pub(crate) fn chunk_of(position: Vec3) -> ChunkKey`, qui remplace `tile_of`.

- [ ] **Step 1 : Renommer `cubes_are_grouped_by_tile` en `cubes_are_grouped_by_chunk` et ajouter**

```rust
#[test]
fn a_tall_column_spans_several_chunks() {
    let layout = build_chunks(&[cube(0.0, 0.0, 0.0), cube(0.0, 33.0, 0.0), cube(0.0, -1.0, 0.0)]);
    assert_eq!(layout.chunks.len(), 3);
}
```

- [ ] **Step 2 : Vérifier l'échec.** Run `<scene-test> a_tall_column`. Expected : FAIL, 1 chunk au lieu de 3.
- [ ] **Step 3 : Implémenter `chunk_of`** avec `floor(position / CHUNK_SIZE)` sur les trois axes. `build_chunks` trie toujours ses clés. Mettre à jour le commentaire de tête du fichier.
- [ ] **Step 4 : Vérifier.** Run `<scene-test>`. Expected : tout passe.
- [ ] **Step 5 : Commit** `feat(etib-scene): cut chunks in three dimensions`.

---

### Task 2 : Le monde côté CPU

**Files :** Create `crates/etib-scene/src/world.rs` (avec `mod world;` dans `lib.rs`). `BlockEdit` est exporté dans `lib.rs`.

**Interfaces :**
- Consumes : `ChunkKey`, `chunk_of`, `crate::Error::Capacity { needed, capacity }`.
- Produces :

```rust
pub enum BlockEdit { Keep, Set(Vec3), Remove }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Priority { Budget, Immediate }

impl World {
    pub fn new(cubes: &[ModelCube], capacity: usize) -> Self;
    pub fn capacity(&self) -> usize;
    pub fn cube_count(&self) -> usize;
    pub fn set_block(&mut self, cell: IVec3, color: Vec3) -> Result<()>;
    pub fn remove_block(&mut self, cell: IVec3);
    pub fn block(&self, cell: IVec3) -> Option<Vec3>;
    pub fn edit_region(&mut self, min: IVec3, max: IVec3,
                       edit: impl FnMut(IVec3) -> BlockEdit) -> Result<()>;
    /// Toutes les clés de chunks non vides, dans l'ordre.
    pub fn chunk_keys(&self) -> impl Iterator<Item = ChunkKey> + '_;
    /// LA fonction de reconstruction : les instances du chunk. Vide s'il n'existe plus.
    pub fn instances(&self, key: ChunkKey) -> Vec<CubeRaw>;
    pub fn chunk_len(&self, key: ChunkKey) -> u32;
    pub fn dirty(&self) -> impl Iterator<Item = (ChunkKey, Priority)> + '_;
    pub fn clean(&mut self, key: ChunkKey);
}
```

Stockage : `BTreeMap<ChunkKey, Chunk>`, et dans `Chunk` un `BTreeMap<[i32; 3], Vec3>` de blocs et un `Vec<ModelCube>` de cubes hors grille. Un cube est un bloc si `position == position.round()`. Un chunk vide quitte la map. `instances` donne les blocs dans l'ordre de leur clé, puis les cubes hors grille dans leur ordre d'arrivée. `new` ne marque rien à reconstruire.

- [ ] **Step 1 : Écrire les tests**

```rust
fn world(capacity: usize) -> World { World::new(&[], capacity) }
const RED: Vec3 = Vec3::new(1.0, 0.0, 0.0);
const BLUE: Vec3 = Vec3::new(0.0, 0.0, 1.0);

#[test] fn a_set_block_is_read_back_at_once() {
    let mut w = world(4);
    w.set_block(IVec3::new(1, 2, 3), RED).unwrap();
    assert_eq!(w.block(IVec3::new(1, 2, 3)), Some(RED));
    assert_eq!(w.cube_count(), 1);
}
#[test] fn removing_an_empty_cell_marks_nothing() {
    let mut w = world(4);
    w.remove_block(IVec3::ZERO);
    assert_eq!(w.dirty().count(), 0);
}
#[test] fn the_last_of_two_blocks_in_a_cell_wins() {
    let c = |color| ModelCube { position: Vec3::ZERO, color };
    let w = World::new(&[c(RED), c(BLUE)], 2);
    assert_eq!(w.block(IVec3::ZERO), Some(BLUE));
    assert_eq!(w.cube_count(), 1);
}
#[test] fn going_past_capacity_fails_and_changes_nothing() {
    let mut w = world(1);
    w.set_block(IVec3::ZERO, RED).unwrap();
    w.clean([0, 0, 0]);
    assert!(matches!(w.set_block(IVec3::X, RED),
        Err(Error::Capacity { needed: 2, capacity: 1 })));
    assert_eq!(w.block(IVec3::X), None);
    assert_eq!(w.dirty().count(), 0);
}
#[test] fn replacing_a_block_at_capacity_succeeds() {
    let mut w = world(1);
    w.set_block(IVec3::ZERO, RED).unwrap();
    w.set_block(IVec3::ZERO, BLUE).unwrap();
    assert_eq!(w.block(IVec3::ZERO), Some(BLUE));
}
#[test] fn negative_cells_fall_in_negative_chunks() {
    let mut w = world(1);
    w.set_block(IVec3::new(-1, 0, 0), RED).unwrap();
    assert_eq!(w.chunk_keys().collect::<Vec<_>>(), [[-1, 0, 0]]);
}
#[test] fn a_block_on_a_border_rebuilds_only_its_own_chunk() {
    let mut w = world(4);
    w.set_block(IVec3::new(32, 0, 0), RED).unwrap();
    w.clean([1, 0, 0]);
    w.set_block(IVec3::new(31, 0, 0), RED).unwrap();
    assert_eq!(w.dirty().collect::<Vec<_>>(), [([0, 0, 0], Priority::Immediate)]);
}
#[test] fn a_region_marks_each_chunk_once_for_the_budget() {
    let mut w = world(100);
    w.edit_region(IVec3::ZERO, IVec3::new(40, 0, 0), |_| BlockEdit::Set(RED)).unwrap();
    assert_eq!(w.dirty().collect::<Vec<_>>(),
        [([0, 0, 0], Priority::Budget), ([1, 0, 0], Priority::Budget)]);
}
#[test] fn a_block_makes_a_waiting_chunk_immediate() {
    let mut w = world(4);
    w.edit_region(IVec3::ZERO, IVec3::ZERO, |_| BlockEdit::Set(RED)).unwrap();
    w.set_block(IVec3::X, BLUE).unwrap();
    assert_eq!(w.dirty().collect::<Vec<_>>(), [([0, 0, 0], Priority::Immediate)]);
}
#[test] fn a_region_past_capacity_changes_nothing() {
    let mut w = world(1);
    let result = w.edit_region(IVec3::ZERO, IVec3::new(1, 0, 0), |_| BlockEdit::Set(RED));
    assert!(matches!(result, Err(Error::Capacity { needed: 2, capacity: 1 })));
    assert_eq!(w.cube_count(), 0);
    assert_eq!(w.dirty().count(), 0);
}
#[test] fn a_region_counts_what_it_removes_before_checking_capacity() {
    let mut w = world(1);
    w.set_block(IVec3::ZERO, RED).unwrap();
    w.edit_region(IVec3::ZERO, IVec3::X, |c| {
        if c == IVec3::ZERO { BlockEdit::Remove } else { BlockEdit::Set(BLUE) }
    }).unwrap();
    assert_eq!(w.block(IVec3::X), Some(BLUE));
    assert_eq!(w.cube_count(), 1);
}
#[test] fn an_inverted_box_changes_nothing() {
    let mut w = world(4);
    let mut calls = 0;
    w.edit_region(IVec3::ONE, IVec3::ZERO, |_| { calls += 1; BlockEdit::Set(RED) }).unwrap();
    assert_eq!(calls, 0);
}
#[test] fn off_grid_cubes_follow_the_answer_for_their_rounded_cell() {
    let off = |x| ModelCube { position: Vec3::new(x, 0.25, 0.0), color: RED };
    let mut w = World::new(&[off(0.25), off(3.25)], 4);
    w.edit_region(IVec3::ZERO, IVec3::new(3, 0, 0), |c| {
        if c.x == 0 { BlockEdit::Remove } else { BlockEdit::Keep }
    }).unwrap();
    assert_eq!(w.cube_count(), 1);
    let kept = w.instances([0, 0, 0]);
    assert_eq!(kept[0].model[3][0], 3.25);
}
#[test] fn the_same_blocks_give_the_same_instances() {
    let cells = [IVec3::new(5, 1, 2), IVec3::new(1, 0, 0), IVec3::new(9, 9, 9)];
    let build = |order: &[IVec3]| {
        let mut w = world(4);
        for &c in order { w.set_block(c, RED).unwrap(); }
        bytemuck::cast_slice::<CubeRaw, u8>(&w.instances([0, 0, 0])).to_vec()
    };
    assert_eq!(build(&cells), build(&[cells[2], cells[0], cells[1]]));
}
```

- [ ] **Step 2 : Vérifier l'échec.** Run `<scene-test> world`. Expected : échec de compilation, `World` n'existe pas.
- [ ] **Step 3 : Implémenter `world.rs`.** `edit_region` parcourt les cases en `x`, puis `y`, puis `z` croissants et appelle `edit` une fois par case. Il rassemble les changements, compte le total final, vérifie la capacité, et seulement ensuite applique. Un changement `Keep`, ou un `Remove` sur une case vide sans cube hors grille, ne marque rien. `Priority::Budget` ne rétrograde jamais un chunk déjà `Immediate`.
- [ ] **Step 4 : Vérifier.** Run `<scene-test> world`. Expected : tout passe, seul `dead_code` peut avertir.
- [ ] **Step 5 : Commit** `feat(etib-scene): the world of blocks on the CPU`.

---

### Task 3 : Emplacements du buffer d'instances

**Files :** Create `crates/etib-scene/src/slots.rs`

**Pourquoi des morceaux :** `cull.wgsl` parcourt tous les emplacements et ne lit que `chunk_ids`, jamais `start_idx`. Un chunk peut donc être dispersé dans le buffer, ce qui évite compaction et échec par fragmentation.

**Interfaces :**
- Produces :

```rust
impl SlotAllocator {
    pub fn new(capacity: u32) -> Self;
    /// `count` emplacements, les plus bas d'abord, en un ou plusieurs morceaux.
    /// `None`, sans rien prendre, s'il en reste moins que `count`.
    pub fn alloc(&mut self, count: u32) -> Option<Vec<Range<u32>>>;
    pub fn free(&mut self, ranges: &[Range<u32>]);
    pub fn free_count(&self) -> u32;
    /// Un de plus que le dernier emplacement occupé, 0 si tout est libre.
    pub fn end(&self) -> u32;
}
```

- [ ] **Step 1 : Écrire les tests**

```rust
#[test] fn slots_are_packed_from_the_bottom() {
    let mut s = SlotAllocator::new(10);
    assert_eq!(s.alloc(3), Some(vec![0..3]));
    assert_eq!(s.alloc(2), Some(vec![3..5]));
    assert_eq!(s.end(), 5);
}
#[test] fn too_much_takes_nothing() {
    let mut s = SlotAllocator::new(2);
    assert_eq!(s.alloc(3), None);
    assert_eq!(s.free_count(), 2);
}
#[test] fn a_fragmented_buffer_still_fills_up() {
    let mut s = SlotAllocator::new(6);
    let a = s.alloc(2).unwrap();
    let _b = s.alloc(2).unwrap();
    let c = s.alloc(2).unwrap();
    s.free(&a);
    s.free(&c);
    assert_eq!(s.alloc(4), Some(vec![0..2, 4..6]));
    assert_eq!(s.free_count(), 0);
}
#[test] fn freed_neighbours_merge_and_the_end_comes_down() {
    let mut s = SlotAllocator::new(6);
    let a = s.alloc(3).unwrap();
    let b = s.alloc(3).unwrap();
    s.free(&b);
    s.free(&a);
    assert_eq!(s.end(), 0);
    assert_eq!(s.alloc(6), Some(vec![0..6]));
}
#[test] fn allocating_nothing_gives_nothing() {
    let mut s = SlotAllocator::new(2);
    assert_eq!(s.alloc(0), Some(vec![]));
}
```

- [ ] **Step 2 : Vérifier l'échec.** Run `<scene-test> slots`. Expected : échec de compilation.
- [ ] **Step 3 : Implémenter** avec un `BTreeMap<u32, u32>` des morceaux libres (début → longueur).
- [ ] **Step 4 : Vérifier.** Run `<scene-test> slots`. Expected : tout passe.
- [ ] **Step 5 : Commit** `feat(etib-scene): slots of the instance buffer, in pieces`.

---

### Task 4 : Quels chunks appliquer à cette image

**Files :** Create `crates/etib-scene/src/rebuild.rs`

**Interfaces :**
- Consumes : `ChunkKey`, `Priority`.
- Produces :

```rust
pub const DEFAULT_REBUILD_BUDGET: u32 = 50_000;

pub(crate) struct Candidate {
    pub key: ChunkKey,
    pub priority: Priority,
    /// Du centre du chunk à la caméra.
    pub distance: f32,
    /// Cubes qu'il occupe sur le GPU, et cubes qu'il aura.
    pub old_len: u32,
    pub new_len: u32,
}

/// Les chunks à appliquer, dans l'ordre d'application.
pub(crate) fn plan_rebuilds(candidates: &[Candidate], free: u32, budget: u32) -> Vec<ChunkKey>
```

**Algorithme**, que la spec fixe mais que les tests seuls ne déterminent pas :
- Le coût d'un chunk est `max(old_len, new_len)`, car effacer ses emplacements se paie aussi.
- La place libre simulée part de `free`. Appliquer un chunk fait `room += old_len - new_len`.
- Ordre de base : distance croissante avec `total_cmp`, puis clé.
- **Immédiats d'abord,** dans cet ordre, sans budget. Si l'un n'a pas la place, on insère avant lui les chunks `Budget` qui rétrécissent, les plus proches d'abord, jusqu'à ce qu'il l'ait. Ils consomment le budget, qui peut passer à zéro.
- **Puis budget :** on parcourt les chunks `Budget` restants, et on en applique tant qu'il reste du budget. Le premier chunk `Budget` de l'image passe même s'il dépasse le budget. Un chunk sans la place est mis de côté, sans rien consommer, et le parcours reprend depuis le début de ceux mis de côté dès qu'un chunk appliqué a libéré de la place : les rétrécissements passent ainsi devant lui. Le parcours s'arrête au premier chunk qui a la place mais dépasse le budget restant.

- [ ] **Step 1 : Écrire les tests**

```rust
fn chunk(x: i32, priority: Priority, old_len: u32, new_len: u32) -> Candidate {
    Candidate { key: [x, 0, 0], priority, distance: x as f32, old_len, new_len }
}
use Priority::{Budget, Immediate};

#[test] fn nothing_to_do_plans_nothing() {
    assert!(plan_rebuilds(&[], 10, 100).is_empty());
}
#[test] fn nearest_chunks_go_first_up_to_the_budget() {
    let c = [chunk(3, Budget, 0, 40), chunk(1, Budget, 0, 40), chunk(2, Budget, 0, 40)];
    assert_eq!(plan_rebuilds(&c, 1000, 100), [[1, 0, 0], [2, 0, 0]]);
}
#[test] fn immediate_chunks_ignore_the_budget_and_go_first() {
    let c = [chunk(1, Budget, 0, 10), chunk(9, Immediate, 0, 500)];
    assert_eq!(plan_rebuilds(&c, 1000, 100), [[9, 0, 0], [1, 0, 0]]);
}
#[test] fn a_chunk_over_the_budget_passes_alone() {
    let c = [chunk(1, Budget, 0, 500), chunk(2, Budget, 0, 10)];
    assert_eq!(plan_rebuilds(&c, 1000, 100), [[1, 0, 0]]);
}
#[test] fn a_budget_of_zero_still_makes_progress() {
    let c = [chunk(1, Budget, 0, 10), chunk(2, Budget, 0, 10)];
    assert_eq!(plan_rebuilds(&c, 1000, 0), [[1, 0, 0]]);
}
#[test] fn removing_a_chunk_costs_budget_too() {
    let c = [chunk(1, Budget, 80, 0), chunk(2, Budget, 80, 0)];
    assert_eq!(plan_rebuilds(&c, 0, 100), [[1, 0, 0]]);
}
#[test] fn a_full_buffer_lets_shrinking_chunks_pass_first() {
    let c = [chunk(1, Budget, 0, 50), chunk(9, Budget, 50, 0)];
    assert_eq!(plan_rebuilds(&c, 0, 1000), [[9, 0, 0], [1, 0, 0]]);
}
#[test] fn an_immediate_chunk_in_a_full_buffer_pulls_shrinking_chunks_past_the_budget() {
    let c = [
        chunk(1, Immediate, 0, 1),
        chunk(8, Budget, 500, 0),
        chunk(9, Budget, 500, 0),
    ];
    assert_eq!(plan_rebuilds(&c, 0, 10), [[8, 0, 0], [1, 0, 0]]);
}
```

- [ ] **Step 2 : Vérifier l'échec.** Run `<scene-test> rebuild`. Expected : échec de compilation.
- [ ] **Step 3 : Implémenter `plan_rebuilds`.**
- [ ] **Step 4 : Vérifier.** Run `<scene-test> rebuild`. Expected : tout passe.
- [ ] **Step 5 : Commit** `feat(etib-scene): choose which chunks to rebuild each frame`.

---

### Task 5 : Buffers à capacité fixe

**Files :** Modify `culling.rs`, `scene.rs`, `shaders/chunk_cull.wgsl`, et chaque appel à `Scene::new` : `examples/sounds.rs:55`, `examples/physics.rs:155`, `examples/pong/game.rs:137`, `examples/bench/main.rs:190`, `examples/hello/main.rs:164`, plus `lib.rs` (doctest) et les tests de `scene.rs`.

**Interfaces :**
- Consumes : `World`, `SlotAllocator`.
- Produces :

```rust
pub fn Scene::new(device, static_cubes, static_capacity: usize, dynamic_capacity, width, height) -> Self
impl StaticCubes {
    /// `None` si `world.capacity()` vaut 0.
    pub fn new(device: &Device, passes: &CullingPasses, world: &World) -> Option<Self>;
    /// Remplace le contenu GPU du chunk `key` par celui de `world`, en une fois.
    pub fn rebuild(&mut self, device: &Device, world: &World, key: ChunkKey);
    pub fn free_slots(&self) -> u32;
    /// Cubes qu'occupe le chunk sur le GPU, 0 s'il n'y est pas.
    pub fn gpu_len(&self, key: ChunkKey) -> u32;
}
```

- [ ] **Step 1 : Ajouter à `scene.rs` les tests**

```rust
#[test]
fn an_empty_chunk_row_is_never_visible() {
    let Some(device) = device() else { return };
    let mut scene = Scene::new(&device, &[white_cube(0.0, 0.0, 0.0)], 8, 1, 64, 64);
    let mut encoder = device.encoder("test");
    scene.render(&device, &mut encoder, &mut camera());
    finish(&device, encoder);
    let statics = scene.statics.as_ref().unwrap();
    assert_eq!(statics.draw.read_back(&device)[0].instance_count, 1);
}

#[test]
fn a_capacity_of_zero_makes_no_static_buffers() {
    let Some(device) = device() else { return };
    assert!(Scene::new(&device, &[], 0, 1, 64, 64).statics.is_none());
}

#[test]
#[should_panic(expected = "static capacity")]
fn more_cubes_than_capacity_panics() {
    let Some(device) = device() else { panic!("static capacity") };
    Scene::new(&device, &[white_cube(0.0, 0.0, 0.0)], 0, 1, 64, 64);
}
```

Adapter `static_cubes_are_split_into_chunks` et `cubes_kept` à la nouvelle signature. `static_count` lit `world.cube_count()`.

- [ ] **Step 2 : Vérifier l'échec.** Run `<scene-test> scene`. Expected : échec de compilation sur le nombre d'arguments de `Scene::new`.
- [ ] **Step 3 : Implémenter.**
  - `Scene` possède un `world: World`, créé dans `new`. Si `static_capacity < static_cubes.len()`, panique avec un message contenant `static capacity`, documenté en `# Panics`.
  - `instances`, `visible`, `chunk_ids` ont `capacity` éléments. `chunks` et `chunk_visible` en ont `capacity + 1`, car une scène peut avoir un cube par chunk. Le coût est d'environ 50 octets par cube de capacité, sans effet sur `cull`, qui s'arrête à la dernière ligne utilisée.
  - La ligne 0 est réservée : elle reste à `count == 0`, et les emplacements libres pointent vers elle, donc `chunk_ids` tout à zéro signifie « tout libre ».
  - `chunk_cull.wgsl` : `chunk_visible[idx] = u32(c.count > 0u && intersects_aabb(aabb));`, et le commentaire de tête explique la ligne 0. `cull.wgsl` ne change pas.
  - `rebuild` libère les morceaux du chunk et y écrit 0 dans `chunk_ids`, alloue `world.chunk_len(key)` emplacements, écrit instances et `chunk_ids` morceau par morceau, puis la ligne `ChunkRaw` (bornes de `build_chunks`, `start_idx` du premier morceau, `count`). Chunk vide : sa ligne repart à zéro et revient dans une liste de lignes libres. Une allocation impossible est une panique avec un message clair, car `plan_rebuilds` l'exclut.
  - `new` appelle `rebuild` pour chaque `world.chunk_keys()` dans l'ordre, ce qui tasse les chunks au chargement. `build_chunks` et `StaticLayout` disparaissent. Leurs tests de bornes et de déterminisme passent sur `rebuild` et `World`.
  - `cull` lance `SlotAllocator::end()` et le nombre de lignes utilisées au lieu de `count` et `chunk_count`. `chunk_count` compte les chunks présents.
  - Les exemples passent `cubes.len()` comme capacité.
- [ ] **Step 4 : Vérifier.** Run `<scene-test>` puis `cargo build --offline -p etib-examples --bins --examples`. Expected : tout passe, tout compile.
- [ ] **Step 5 : Commit** `feat(etib-scene): static buffers with a fixed capacity`.

---

### Task 6 : Modifier le décor depuis `Scene`

**Files :** Modify `scene.rs`, `lib.rs` (`pub use world::BlockEdit;`, `pub use rebuild::DEFAULT_REBUILD_BUDGET;`, réexport déjà couvert de `glam`), `ARCHITECTURE.md`.

**Interfaces :**
- Consumes : `World`, `plan_rebuilds`, `StaticCubes::{rebuild, free_slots, gpu_len}`.
- Produces : `Scene::{set_block, remove_block, block, edit_region, set_rebuild_budget}` aux signatures de la spec, et `pub fn pending_rebuilds(&self) -> usize`, chunks dont la nouvelle version n'est pas encore à l'écran. Cette dernière fonction n'est pas dans la spec, mais le critère 3 en a besoin.

- [ ] **Step 1 : Ajouter à `culling.rs` un `#[cfg(test)] pub fn live_instances(&self, device) -> Vec<CubeRaw>`** : relit `instances` et `chunk_ids`, garde les emplacements dont l'identifiant n'est pas 0, et trie par octets `bytemuck`. `_instances` et `_chunk_ids` perdent leur `_`.

- [ ] **Step 2 : Écrire les tests dans `scene.rs`**, avec un helper `render_once(&device, &mut scene)` et une scène de capacité explicite :

```rust
fn same_as_fresh(device: &Device, scene: &Scene, cubes: &[ModelCube]) {
    let mut fresh = Scene::new(device, cubes, cubes.len().max(1), 1, 64, 64);
    render_once(device, &mut fresh);
    let bytes = |s: &Scene| -> Vec<u8> {
        bytemuck::cast_slice(&s.statics.as_ref().unwrap().live_instances(device)).to_vec()
    };
    assert_eq!(bytes(scene), bytes(&fresh));
}

#[test] fn a_block_is_on_the_gpu_at_the_next_frame() { /* set_block, render_once, same_as_fresh */ }
#[test] fn a_rebuilt_chunk_matches_a_fresh_scene_after_many_edits() {
    // Pose, remplace et retire 20 blocs sur 3 chunks, puis same_as_fresh avec le résultat.
}
#[test] fn a_waiting_chunk_keeps_its_old_contents() {
    let Some(device) = device() else { return };
    let mut scene = Scene::new(&device, &[white_cube(0.0, 0.0, 0.0), white_cube(40.0, 0.0, 0.0)], 8, 1, 64, 64);
    scene.set_rebuild_budget(1);
    render_once(&device, &mut scene);
    let before = scene.statics.as_ref().unwrap().live_instances(&device);
    scene.edit_region(IVec3::ZERO, IVec3::new(40, 0, 0), |_| BlockEdit::Remove).unwrap();
    assert_eq!(scene.block(IVec3::ZERO), None);
    assert_eq!(scene.static_count(), 0);
    assert_eq!(scene.statics.as_ref().unwrap().live_instances(&device), before);
    assert_eq!(scene.pending_rebuilds(), 2);
    render_once(&device, &mut scene);
    assert_eq!(scene.pending_rebuilds(), 1);
    render_once(&device, &mut scene);
    assert_eq!(scene.pending_rebuilds(), 0);
}
#[test] fn past_capacity_a_scene_reports_it_and_changes_nothing() { /* Error::Capacity, static_count inchangé */ }
#[test] fn filling_emptying_and_refilling_in_another_order_never_fails() {
    // Capacité 64, 64 blocs sur 4 chunks, tout retirer un bloc sur deux, reposer dans l'autre sens.
    // Répéter 3 fois, render_once à chaque étape, puis same_as_fresh.
}
#[test] fn a_full_scene_can_swap_a_far_zone_for_a_near_one_and_take_a_block() {
    // Capacité 8, 8 blocs dans un chunk lointain, budget 1.
    // edit_region retire la zone lointaine, edit_region ajoute 7 blocs près de la caméra,
    // set_block pose le 8e : aucun Err. render_once puis same_as_fresh sur les 8 blocs proches.
}
#[test] fn a_scene_emptied_and_refilled_matches_a_fresh_one() {
    // Tout retirer, render_once, vérifier instance_count == 0, remettre 2 blocs, same_as_fresh.
}
#[test] fn a_scene_built_empty_can_receive_blocks() { /* Scene::new(&device, &[], 4, ...) puis set_block */ }
```

- [ ] **Step 3 : Vérifier l'échec.** Run `<scene-test> scene`. Expected : échec de compilation, `set_block` n'existe pas.
- [ ] **Step 4 : Implémenter.**
  - Les méthodes publiques délèguent à `World`, avec une documentation qui dit la différence de vitesse entre `set_block` et `edit_region`.
  - Dans `render`, avant `statics.reset` : construire les `Candidate` depuis `world.dirty()`, avec `distance` entre `camera.eye` et le centre `(key + 0.5) * CHUNK_SIZE`, `old_len = statics.gpu_len(key)` et `new_len = world.chunk_len(key)`. Appeler `plan_rebuilds`, puis `statics.rebuild` et `world.clean` pour chaque clé. Si rien n'est marqué, ne rien allouer.
  - Budget initial : `DEFAULT_REBUILD_BUDGET`.
  - Mettre à jour le commentaire de `Scene` (« Static cubes are fixed at construction »), le commentaire de tête de `culling.rs`, et `ARCHITECTURE.md` lignes 667 et 932.
- [ ] **Step 5 : Vérifier.** Run `<scene-test>`, puis `cargo test --offline --workspace` et `cargo clippy --offline --workspace --all-targets`. Expected : tout passe, clippy ne signale rien.
- [ ] **Step 6 : Commit** `feat(etib-scene): edit the scenery block by block and by region`.

---

### Task 7 : L'image la plus lente dans les résultats

**Files :** Modify `examples/bench/collect.rs`, `results.rs`, `compare.rs`

**Interfaces :**
- Produces : `Summary.max_ms: Option<f32>` avec `#[serde(default)]`, `fn max_row(label: &str, a: Option<&Summary>, b: Option<&Summary>) -> [String; 7]` dans `compare.rs`.

- [ ] **Step 1 : Écrire les tests**

```rust
// collect.rs
#[test] fn the_summary_keeps_the_slowest_sample() {
    assert_eq!(summarize(&[4.0, 1.0, 9.0, 2.0]).unwrap().max_ms, Some(9.0));
}
// results.rs
#[test] fn a_run_saved_before_max_ms_has_none() {
    let mut json = serde_json::to_value(sample_run("old")).unwrap();
    json["cpu_frame"].as_object_mut().unwrap().remove("max_ms");
    let run: RunResults = serde_json::from_value(json).unwrap();
    assert_eq!(run.cpu_frame.max_ms, None);
}
// compare.rs
#[test] fn compare_shows_the_slowest_frame_and_a_dash_when_unknown() {
    let (mut a, b) = (run("a", 4.0), run("b", 4.0));
    a.cpu_frame.max_ms = None;
    let out = compare(&a, &b);
    assert!(out.contains("frame CPU max"), "{out}");
    assert!(out.lines().find(|l| l.contains("frame CPU max")).unwrap().contains(DASH), "{out}");
}
```

Dans `sample_run`, `max_ms: Some(m * 2.0)`. Le `Summary` vide de `collect.rs` prend `None`.

- [ ] **Step 2 : Vérifier l'échec.** Run `<bench-test>`. Expected : échec de compilation sur `max_ms`.
- [ ] **Step 3 : Implémenter.** La ligne `frame CPU max` vient juste après `frame CPU` : valeur de chaque côté dans les colonnes médianes, tiret dans les colonnes p95, delta seulement si les deux valeurs existent.
- [ ] **Step 4 : Vérifier.** Run `<bench-test>`. Expected : tout passe.
- [ ] **Step 5 : Commit** `feat(bench): record and compare the slowest frame`.

---

### Task 8 : Scénarios `edit-blocks` et `edit-area`

**Files :** Modify `examples/bench/scenario.rs`, `main.rs`, `results.rs`, `compare.rs`

**Interfaces :**
- Consumes : `Scene::{set_block, remove_block, block, edit_region, pending_rebuilds}`, `BlockEdit`.
- Produces :

```rust
Scenario::{EditBlocks, EditArea}  // "edit-blocks", "edit-area"
Scenario::static_capacity(self, cubes: usize) -> usize  // cubes * 11 / 10 pour les deux nouveaux, cubes sinon
pub fn edit_blocks(frame: u32) -> [(IVec3, Vec3); 4]
pub enum AreaEdit { Remove, Restore }
pub const AREA_RADIUS: i32 = 32;
pub fn area_edit(frame: u32) -> Option<(AreaEdit, IVec3)>   // l'opération et le centre de la boule
pub fn terrain_block(cell: IVec3) -> Option<Vec3>
ScenarioInfo.rebuild_frames: u32  // #[serde(default)]
```

`edit_blocks` : le groupe de cases ne change que toutes les 2 images. Pour l'image `f`, on prend `camera_at(f & !1).0`, on y ajoute `(±20, ±20)` en X et Z, on borne à `-SIDE/2..SIDE/2`, et on prend `y = top(x, z)`, avec la couleur de `color`. `top` est extrait de `world()` sans changer le monde : le test `terrain_is_the_world_and_nothing_else` et un run de référence le garantissent.

`area_edit` : `Remove` si `f % 120 == 0`, `Restore` si `f % 120 == 60`, sinon `None`. Le centre est celui de `f - f % 120` : `camera_at(..).1`, borné au monde, à la hauteur `top`.

- [ ] **Step 1 : Écrire les tests et étendre `ALL` à 6 scénarios**

```rust
#[test] fn the_edit_scenarios_start_from_the_terrain() {
    assert_eq!(Scenario::EditBlocks.static_cubes(), world());
    assert_eq!(Scenario::EditArea.static_cubes(), world());
    assert!(Scenario::EditBlocks.models().is_empty());
}
#[test] fn the_edit_scenarios_leave_room() {
    assert_eq!(Scenario::EditBlocks.static_capacity(100), 110);
    assert_eq!(Scenario::Terrain.static_capacity(100), 100);
}
#[test] fn edit_blocks_touches_four_chunks_of_the_surface_every_frame() {
    for frame in 0..WARMUP_FRAMES + MEASURED_FRAMES + GRACE_FRAMES {
        let cells = edit_blocks(frame);
        let chunks: HashSet<_> = cells.iter()
            .map(|(c, _)| (c.div_euclid(IVec3::splat(32))).to_array()).collect();
        assert_eq!(chunks.len(), 4, "frame {frame}");
        assert!(cells.iter().all(|(c, color)| terrain_block(*c) == Some(*color)));
    }
}
#[test] fn edit_blocks_uses_the_same_cells_two_frames_in_a_row() {
    assert_eq!(edit_blocks(10), edit_blocks(11));
    assert_ne!(edit_blocks(300), edit_blocks(400));
}
#[test] fn a_zone_is_removed_then_restored_at_the_same_place() {
    let (remove, centre) = area_edit(120).unwrap();
    assert!(matches!(remove, AreaEdit::Remove));
    let (restore, back) = area_edit(180).unwrap();
    assert!(matches!(restore, AreaEdit::Restore));
    assert_eq!(centre, back);
    assert!(area_edit(121).is_none());
}
#[test] fn terrain_block_knows_the_world() {
    for cube in world().iter().step_by(997) {
        assert_eq!(terrain_block(cube.position.as_ivec3()), Some(cube.color));
    }
    assert_eq!(terrain_block(IVec3::new(0, -1, 0)), None);
    assert_eq!(terrain_block(IVec3::new(0, 200, 0)), None);
}
```

Dans `main.rs`, `a_scenario_is_chosen_by_its_name` reste, et s'ajoute `edit-area` parsé en `Scenario::EditArea`. Dans `compare.rs` :

```rust
#[test] fn compare_shows_how_long_the_rebuilds_took() {
    let mut a = run("a", 4.0);
    a.scenario.rebuild_frames = 7;
    assert!(compare(&a, &a).contains("7 rebuild frames"));
}
```

- [ ] **Step 2 : Vérifier l'échec.** Run `<bench-test>`. Expected : échec de compilation.
- [ ] **Step 3 : Implémenter `scenario.rs`.**
- [ ] **Step 4 : Vérifier.** Run `<bench-test>`. Expected : tout passe.
- [ ] **Step 5 : Jouer les modifications dans `main.rs`.**
  - `Scene::new` reçoit `scenario.static_capacity(cubes.len())`.
  - Dans `App::render`, avant `scene.render`, une fonction `apply_edits(&mut self)` :
    - `EditBlocks` : pour chaque `(cell, color)` de `edit_blocks(bench_frame)`, `remove_block` si le bloc est présent, sinon `set_block`.
    - `EditArea` : sur `area_edit`, `edit_region` sur la boîte du rayon `AREA_RADIUS`. Dans la boule, `Remove`, ou `terrain_block(cell).map_or(BlockEdit::Keep, BlockEdit::Set)` pour `Restore`. Hors de la boule, `Keep`.
    - Les `Result` passent par `.expect("the capacity covers the scenario")`.
  - Pendant les images mesurées, compter les images consécutives où `pending_rebuilds() > 0` après `scene.render`, garder la plus longue série dans `rebuild_frames`, et l'écrire dans `ScenarioInfo`.
  - `header_line` ajoute `, {} rebuild frames`. Le commentaire de tête de `main.rs` liste les deux scénarios.
- [ ] **Step 6 : Vérifier.** Run `cargo test --offline --workspace` et `cargo clippy --offline --workspace --all-targets`. Expected : tout passe.
- [ ] **Step 7 : Run court sur le pilote logiciel.** Run `edit-blocks` et `edit-area` sous `xvfb-run`, en build release, hors de `engine/bench-results`, puis supprimer les fichiers. Expected : `saved ...` pour les deux si le temps le permet, aucune panique. Noter dans le rapport si un run ne finit pas. Ses chiffres ne valent rien, seul le fait que le run se termine compte.
- [ ] **Step 8 : Commit** `feat(bench): scenarios that edit the scenery`.

---

### Task 9 : Mesures sur Mac

À faire par Corentin. La fusion dépend de cette tâche.

- [ ] **Step 1 : Les références.** Sur `bench-scenarios`, depuis `engine/`, lancer `terrain`, `scattered`, `moving` et `few-changes` avec `--save base-<scénario>`.
- [ ] **Step 2 : Le nouveau code.** Sur `editable-world`, lancer les six scénarios avec `--save new-<scénario>`.
- [ ] **Step 3 : Comparer** avec `--compare base-<s> new-<s>` pour les quatre scénarios existants, puis `--compare new-terrain new-edit-blocks` et `--compare new-terrain new-edit-area`, qui afficheront l'avertissement normal de scénarios différents.
- [ ] **Step 4 : Juger** selon les critères 1 à 3 de la spec, et suivre sa « Définition de terminé » si un seuil est manqué.
