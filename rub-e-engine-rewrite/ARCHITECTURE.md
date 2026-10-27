# etib architecture

etib is a small framework for windows whose contents are drawn on the GPU — by
a compute shader, or by rasterizing meshes with a depth buffer — and presented
through a colour-managed pipeline. It opens the window,
paces the frames, and owns the path from what a shader wrote to what the display
actually shows — including the parts that stop being simple on an HDR display.

It takes no position on *what* is drawn. The shape of an application is: write
into an offscreen texture, hand that texture to a presentation pass, optionally
paint an interface over the top — and, just as optionally, make a noise while it
does.

Built on [wgpu](https://wgpu.rs) 30 and [winit](https://github.com/rust-windowing/winit)
0.30, Rust edition 2024.

---

## 1. Crates

Seven crates, with a strict dependency rule: each knows about the layer below it
and nothing about the layer above. Two of them are optional, and three —
`etib-audio`, `etib-input` and `etib-physics` — have nothing of the workspace
below them at all. The last two sit beside the diagram: `etib-input` knows
winit and, behind its `gamepad` feature, gilrs; `etib-physics` knows only glam.
Neither draws anything, so neither needs a device.

```text
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ your application  (examples/triangle, chart, cube; etib-scene's hello)    │
  │   implements App, owns its scene passes, its state and its sounds         │
  └──────┬──────────────────┬──────────────────┬───────────────────┬──────────┘
         │                  │                  │                   │
  ┌──────▼───────────┐ ┌────▼─────────────┐ ┌──▼───────────────┐ ┌─▼─────────────────┐
  │ etib             │ │ etib-scene       │ │ etib-ui (opt.)   │ │ etib-audio (opt.) │
  │  window, frame   │ │  Camera, Scene,  │ │  egui context    │ │  Audio, Sound,    │
  │  loop, App,      │ │  cube models,    │ │  and paint, the  │ │  Voice, instru-   │
  │  Blit, Post-     │ │  GPU culling,    │ │  HDR composite   │ │  ments, the mixer │
  │  Process, Lut    │ │  skybox          │ │  pass            │ │  knows cpal, CLAP │
  │  knows winit     │ │  knows no window │ │  knows egui,     │ │  and hides them   │
  │                  │ │                  │ │  winit           │ │                   │
  └──────┬───────────┘ └────┬─────────────┘ └──┬───────────────┘ └───────────────────┘
         │                  │                  │                   nothing underneath:
  ┌──────▼──────────────────▼──────────────────▼──┐                no GPU in its tree
  │ etib-core                                     │
  │   Instance, Device, Surface, Encoder,         │
  │   passes, pipelines, textures, bindings,      │
  │   buffers, Display/Output                     │
  │   knows wgpu - and hides it                   │
  └───────────────────────────────────────────────┘
```

| Crate | Owns | Deliberately does not know about |
|---|---|---|
| `etib-core` | Every GPU object, the swapchain, colour-space detection | winit, egui, the frame loop |
| `etib` | The window, the frame loop, the `App` trait, the presentation passes | egui and cpal (behind the `ui` and `audio` features), what is being drawn or heard |
| `etib-scene` | The camera and its controller, cube models, GPU frustum culling, the skybox, the `Scene` that draws them | windows, keyboards and the frame loop — the controller is told which way to `Move`; tone mapping, which is `PostProcess`'s job |
| `etib-input` | Keyboard, mouse and gamepad state, fed from winit events and cleared once per frame | the GPU, the window, the frame loop and `App` — an application feeds it; gilrs, unless the `gamepad` feature is on |
| `etib-physics` | Rigid bodies for cubes: gravity, spin, floor and wall contacts, cube-to-cube impulses, settling onto a face | the GPU, the window and the scene — it moves a position and a rotation the caller owns |
| `etib-ui` | egui integration and the interface composite | the frame loop — it is usable with a loop of your own |
| `etib-audio` | The output device, the mixer, sounds in memory, sounds placed in the world, and CLAP instruments | everything else in the workspace, the GPU included |

**wgpu is an implementation detail of `etib-core`.** Nothing in its API takes or
returns a wgpu type, the errors are its own, and the async device handshake is
done for the caller. Every wrapper still exposes a `raw()` / `raw_mut()`
accessor, because some libraries have to be handed the real objects —
`etib-ui` is the one library in the workspace that uses them, since `egui-wgpu`
speaks wgpu directly (tests elsewhere use the device's `raw()` only to wait for
their GPU work to finish). It borrows through those accessors rather than holding its
own handles, so there is still exactly one device, one queue and one encoder per
frame.

`etib-ui` and `etib-audio` are optional, behind `etib`'s default `ui` and
`audio` features. Without one of them its backend is not compiled, linked, or in
the dependency tree, and nothing in `App`, `Frame` or `run` changes either way.

---

## 2. Startup

```text
etib::run(Config, build)                        crates/etib/src/app.rs
  │
  ├─ EventLoop::new()
  ├─ Instance::new(event_loop.owned_display_handle())
  │     the backend wants the display connection before any window
  │     exists, and it has to outlive it
  │
  └─ on winit's `resumed`:
       ├─ create_window(title, size, borderless fullscreen if asked)
       ├─ Gpu::new -> Instance::open(window, w, h, hdr, vsync)
       │     ├─ create_surface
       │     ├─ request_adapter(HighPerformance, compatible surface)
       │     ├─ request_device(no optional features, downlevel limits
       │     │                 + 8 storage buffers per stage)
       │     └─ Surface::new -> select_output() -> configure()
       │
       └─ build(&gpu) -> App          your pipelines are created here
```

`resumed` can fire again if the platform tears the window down and brings it
back, which is why `build` is `FnMut` and why everything above it is rebuilt
rather than reused.

Two deliberate choices in there:

- **No optional features, downlevel default limits plus storage buffers**
  (`open_device` in `instance.rs`). Everything is the downlevel default except
  `max_storage_buffers_per_shader_stage`, raised from 4 to 8: GPU culling binds
  five in a single compute stage. That is the minimum the engine needs rather
  than the adapter's own limits, so the same binary keeps working across
  backends, and an adapter that cannot meet it is refused at startup with an
  error naming the limit — not later, by a pipeline that fails to validate.
- **Blocking on the async handshake** (`pollster::block_on`). It resolves
  immediately on native backends, and it keeps every caller synchronous.

### Choosing the surface format and colour space

`select_output` (`crates/etib-core/src/surface.rs:228`) decides what the window
is:

```text
hdr != Off  and  Rgba16Float + EXTENDED_SRGB_LINEAR is offered
            and  (hdr == Force  or  the display reports headroom)
   → Rgba16Float, ColorSpace::ExtendedLinear      values are linear, 1.0 = SDR white
otherwise
   → the first sRGB format offered, ColorSpace::Srgb
                                                  values are linear, hardware encodes
```

`Hdr::Allow` requires *both* halves — a surface reports what it can be
configured with, not what state the display is in. Metal offers extended range
for a float surface on any display and DXGI keeps offering it while HDR is
switched off, so taking it regardless would cost a float swapchain for nothing
and would place white by a convention the platform is no longer honouring.
`Hdr::Force` exists for the backends that report no luminance at all.

Wide-gamut and PQ/HLG spaces are deliberately left alone: they change how output
has to be encoded, and there is nothing here that would encode it.

Present mode follows `Config::vsync`, with a frame latency of 2.
`Vsync::On`, the default, is `Fifo`: one image per refresh, as the engine's
`vsync: true` was, so the frame rate is capped at the display's. `Vsync::Off`
takes `Mailbox` when offered, then `Immediate` (which may tear), then `Fifo`;
measure frame rates with it off. `Config::fullscreen` opens borderless
fullscreen on the current monitor. Both are the engine's `config.json`
settings; reading such a file is the application's business.

---

## 3. The frame loop

`Runner::render` (`crates/etib/src/app.rs:145`), once per `RedrawRequested`:

```text
  dt = now - last_frame                     fps is an EMA (0.9 old / 0.1 new);
  app.update(dt)                            a raw 1/dt reads as noise
  │
  surface.acquire() ──► Acquired::Frame  draw into it
                        Acquired::Skip   occluded, timed out, or the swapchain
                                         was just rebuilt — ask for another
                        Acquired::Fatal  unusable; shut down
  │
  encoder = device.encoder("frame encoder")
  app.render(Frame { gpu, encoder, view, stats })    ← all your passes go here
  │
  device.submit(encoder)        one command buffer for the whole frame
  frame.present(&device)
  window.request_redraw()       animation is continuous, so always ask again
```

`ControlFlow::Poll` is set in `about_to_wait`: this animates continuously rather
than only on input.

Recoverable swapchain states are handled inside `Surface::acquire` — it
reconfigures itself where that is the fix, so an application only ever chooses
between drawing, skipping, and giving up.

Events reach `App::window_event` **before** the runner acts on them, and a
`true` return suppresses the runner's own handling (that is how egui captures a
keystroke before Escape would close the window).

Raw device events — relative mouse motion above all, which keeps arriving once
the cursor is captured — go to `App::device_event`. They are not tied to the
window, so nothing in the runner acts on them; an application that does
usually checks first that it has captured the cursor.

`etib_input::Input` turns both streams into state, without anything added to
`App`: the application owns one, passes it every event from `window_event` and
`device_event`, reads it in `update` (`is_key_pressed`, `is_key_just_pressed`,
`mouse_delta`, `scroll`, the gamepads), and calls `clear_frame_state` last in
`update`, so the "just" answers and the mouse motion cover exactly one frame.
Gamepad events are collected in that same call, ready for the next `update`.

---

## 4. The render pipeline

The canonical frame is three passes. Only the middle one belongs to the
framework; the first is yours and the third is optional.

```text
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 1. SCENE                                            your code        │
  │    COMPUTE: dispatch_grid over an Rgba16Float StorageTexture, or     │
  │    RASTER:  draws into an Rgba16Float RenderTexture + DepthTexture   │
  │             — etib_scene::Scene::render is one of these              │
  │    writes SCENE-REFERRED values - 1.0 is not a limit                 │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  │  sampled
  ┌───────────────────────────────▼──────────────────────────────────────┐
  │ 2. POST-PROCESS        crates/etib/src/post.rs + shaders/post.wgsl   │
  │    a fullscreen triangle; clears the target                          │
  │    exposure -> curve -> grade -> LUT -> headroom fit -> white level  │
  │    writes DISPLAY-REFERRED values in the surface's units             │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  │  drawn over
  ┌───────────────────────────────▼──────────────────────────────────────┐
  │ 3. INTERFACE (optional)                     crates/etib-ui           │
  │    egui, composited at SDR white - after the grade, never through it │
  └───────────────────────────────┬──────────────────────────────────────┘
                                  │
                       swapchain image -> present
```

`Blit` (`crates/etib/src/blit.rs`) is the alternative to pass 2: a straight 1:1
copy for content that is already display-referred. It belongs on a standard-range
surface only — on an extended-range one its values would land at a fraction of
the desktop's brightness, because there 1.0 means SDR white rather than maximum.

Both presentation passes are vertexless: `RenderPipeline::screen` draws `0..3`
and the vertex stage generates one oversized triangle from the vertex index,
which avoids the diagonal seam a two-triangle quad can produce. WGSL has no
`include`, so that vertex stage is repeated in each shader.

### The post-processing pass, stage by stage

`crates/etib/src/shaders/post.wgsl`. Everything is linear light unless stated.

| # | Stage | Why there |
|---|---|---|
| 1 | Exposure × colour filter | Before the curve: they decide what the tone mapper is given to compress |
| 2 | Tone curve (`None` / `Reinhard` / `ACES`) | Everything after this point is display-referred |
| 3 | Contrast (pivoting on 0.18) and saturation | Where a grade normally sits |
| 4 | Look-up table, if one is set | A table is a finished look, baked over an already-mapped image |
| 5 | `fit_to_headroom` | The curve does not guarantee the ceiling on its own |
| 6 | `× sdr_white` | Places the image on the surface |

The three curves differ in what they target:

- **None** — clips at the display's peak rather than at white.
- **Reinhard** — applied to *luminance*, not per channel, so the ratio between
  the channels survives and bright areas desaturate far less. It rolls off
  towards `headroom`, so it is the one curve that uses an HDR display's range.
- **ACES** (Narkowicz's fit) — saturates at SDR white by construction, so on an
  HDR surface it gives the reference SDR look at the right brightness rather
  than a brighter image.

`fit_to_headroom` brings an overshooting colour back by scaling all three
channels by one factor rather than clamping each. Clamping cuts only the
brightest channel, changes the ratios, and drags the colour towards white; an
ember at 3× maps to `(1.68, 0.25, 0.06)` on a display with 1.21 of headroom —
clamping gives a duller orange, scaling gives the same orange slightly dimmer.

---

## 5. Colour management

This is the part that most of the design is really about.

### Two kinds of image

- **Scene-referred** — what the compute pass writes. 1.0 is not a limit; a
  highlight can be at 12.0. Needs a float storage format.
- **Display-referred** — what a surface takes. 1.0 means white, and how far
  above it you may go depends on the display.

The post-processing pass is the one place that converts between them.

### `Display` and `Output`

`crates/etib-core/src/display.rs` separates two questions that are easy to
conflate:

```text
Display  — what the platform says about the monitor right now
           headroom, sdr_white_nits, peak_nits, hdr_mode; every field optional,
           all of it advisory, and all of it moves with the brightness slider

Output   — how a finished image should be written to this surface right now
           sdr_white : the value that means SDR white
           headroom  : how far above it output may go
```

`Output::detect(color_space, &display)` derives one from the other: on an sRGB
surface it is always `{ 1.0, 1.0 }`, and on an extended-range one it converts
scRGB's fixed 80-nit reference white to whatever the OS currently calls SDR
white (`nits / 80`, which is Windows-only information; on Apple EDR the scale is
already 1.0) and takes the reported headroom.

**Re-read it; don't cache it.** Those numbers move while the application runs.
Both examples poll once a second rather than every frame — on macOS the answer
lives on main-thread-only AppKit objects, and the platform *ramps* the headroom
up over a second or two after an EDR surface starts drawing, so an app that
reads it once at startup sees 1.0 and stops there.

### Where each thing sits on the surface

```text
extended-range (scRGB) surface, e.g. headroom 16, sdr_white 1.0

   0 ─────────────── 1.0 ──────────────────────────────── 16.0
   black          SDR white                            display peak
                     ▲                                       ▲
                     │                                       │
              the interface                        where Reinhard rolls
              composites here                      highlights off to
              (menus are paper white,
               not highlights)
```

### The interface on an HDR surface

egui picks its encoding from the target's format: for an sRGB format it writes
linear values and lets the hardware encode, and for anything else it writes
sRGB-encoded values directly. Neither is right for a float extended-range
surface, so handed one, egui's output comes out washed out.

So on such a surface `EguiLayer` gives the interface its own `RGBA8_SRGB`
texture — the case egui already handles correctly — and composites it with
premultiplied blending, scaling the colour by `sdr_white`
(`crates/etib-ui/src/composite.wgsl`). It costs one texture and one pass, and
only on that kind of surface; on an sRGB surface egui paints straight into the
swapchain image with `Load::Keep`.

### Look-up tables

`Lut` (`crates/etib/src/lut.rs`) is a cube of output colours: the general form
of a grade, as exported by a grading tool. `Lut::from_cube` parses Adobe `.cube`
files (3D tables over the default 0..1 domain only — a 1D table is a tone curve,
and a non-default domain is better refused than guessed at).

Three details in the shader decide whether the result is the look its author
saw:

1. **The table is indexed in gamma-encoded space, not linear.** A grading tool
   shows its user the signal a display is fed, so the table is a function of
   that. The pass encodes before the lookup and decodes after. Mid-grey is 0.18
   linear but 0.46 encoded — skipping the pair would read every entry at the
   wrong place.
2. **Texel centres.** A table of `size` entries spans 0..1 with one at each end,
   so the coordinate of the nth is `(n + 0.5) / size`; a colour of 0.0 or 1.0
   must land exactly on the first or last entry rather than half a texel
   outside, where clamping would flatten the ends of the grade.
3. **Values above SDR white.** The cube is only defined over 0..1, so the lookup
   runs on the part that fits and the excess is carried over the result — the
   graded colour with its highlight still on top — rather than squashing a
   highlight into the cube.

A shader binding cannot be left empty, so with no table set the pass binds a
neutral 2³ identity and sets `lut_size = 0.0`, which makes the shader skip the
stage entirely. `Grade::lut_strength` mixes per frame; `PostProcess::set_lut`
uploads, and is not something to call per frame.

---

## 6. Sound

`etib-audio` is the crate with nothing underneath it. It shares no types with
`etib-core`, has no GPU in its dependency tree, and works with a frame loop of
your own; in the framework it is optional, behind `etib`'s default `audio`
feature, exactly as `etib-ui` is.

An application owns its `Audio` the way it owns its pipelines — built once in
`build`, kept with its other state, played from wherever it likes. Nothing in
`App`, `Frame` or `run` mentions sound, and nothing has to happen per frame:
playback is paced by the device, not by the frame loop.

```rust
let audio = Audio::new_or_silent();   // silence rather than refusing to start
let blip = Sound::tone(Waveform::Square, 660.0, 0.08);

let voice = audio.play_with(&blip, Play { gain: 0.3, pan: -0.5, ..Default::default() });
voice.set_speed(1.5);                 // and the pitch rises with it
voice.stop();                         // faded over a block, not cut
```

A `Sound` is samples in memory and says nothing about how it is played; a `Play`
says that, and a `Voice` is the handle that can change it while it sounds.
Dropping the handle does **not** stop the sound — most effects are started and
forgotten, and cutting them off at the end of a scope would make the common case
the surprising one.

### Two threads, one of them with a deadline

The device calls back on a real-time thread every few milliseconds, and wants a
block of audio back before the one it is playing runs out. Missing that is an
audible glitch, so the whole crate is arranged around the mixing path
**allocating nothing, locking nothing, logging nothing and freeing nothing.**

```text
  the application's thread                     the device's thread
  ────────────────────────                     ───────────────────

  play()  ──── Command::Play ─────►  bounded channel, 64  ────►  Mixer
                                     (storage allocated              │
                                      when it is created)            │
                                                                     ▼
  drop it here  ◄──── the voice ───  bounded channel, 32  ◄────  finished
   (freeing is                                                    voice
    this side's job)
                                                                     ▲
  Voice::set_gain ─────────────────► atomics ──────────────────► read once
  Audio::set_gain                                                per block
```

That is also why a finished voice is *sent back* rather than dropped where it
ends: dropping the last handle on a sound would free its samples, on the thread
that cannot afford it. A stand-in is left in its place so that the `Vec` entry
being removed drops nothing that is still the last of its kind.

`Audio::silent()` is the same object with no device behind it: every call works
and nothing is heard. It is what `Audio::new_or_silent()` falls back to, so a
machine with no sound card — or a container with no audio at all — costs one
warning rather than a second code path.

### What the mixer does with a block

| # | Stage | Why it is like that |
|---|---|---|
| 1 | Take the commands that have arrived | One `try_recv` loop; nothing blocks |
| 2 | Read each voice's controls, once | Per-sample atomic loads would be paid for nothing |
| 3 | Resample, linearly interpolated | The source is rarely at the device's rate. A gentle low-pass, inaudible at the small ratios involved, and two loads and a multiply |
| 4 | Ramp the gain across the block | A level that jumps between blocks is a step in the waveform, which is a click. It is also why `stop` fades over a block rather than cutting |
| 5 | Sum into one stereo buffer | Panning is a stereo idea, and nearly every device is stereo |
| 6 | Apply the master gain and clamp | A limiter would duck everything whenever one sound peaked, which is not a decision a mixer this size should be making |
| 7 | Write it out in the device's own sample type and channel count | A mono device gets the sum of both sides; channels past the first two are left silent, because placing a sound in a surround field needs a listener orientation this does not have |

Past `MAX_VOICES` — 32 — starting another sound takes the oldest, which is what
a game engine does, because the sound that started last is the one the listener
is waiting for. It is counted in `Audio::stolen()` rather than logged, since it
is noticed on the thread that cannot log.

### Getting a sound in

`Sound::tone` generates one: sine, triangle, saw, square or noise, with five
milliseconds of fade at each end, because a waveform that starts or stops at a
non-zero sample is a step and a step is a click. The shapes other than the sine
are the plain geometric ones rather than a sum of harmonics, so they alias — the
sound eight-bit hardware made, which is what makes them useful for effects.

`Sound::from_wav` reads the other case, from a file or from `include_bytes!`:
RIFF/WAVE, uncompressed PCM at 8, 16, 24 or 32 bits or float at 32 or 64, mono
or stereo, with unknown chunks skipped and everything converted to `f32` on the
way in. Anything that would have to be guessed at rather than read — a
compressed payload, more than two channels — is refused with a message saying
which it was, on the same principle as the `.cube` parser.

`Sound::from_mp3`, behind the `mp3` feature, decodes an MP3 in full through
symphonia (its MP3 decoder only). A damaged frame is skipped, as a player would.

### Sounds in the world

The engine placed sounds through kira's spatial tracks. Here that is a layer
above `Voice`, entirely on the application's thread:

```text
  every frame, from App::update
  SoundManager::update(Listener::looking_at(camera.eye, camera.target))
    ├─ each SoundGroup: SoundStrategy::compute_emitters(cubes, listener) → position, volume
    └─ each SpatialVoice: spatialize(listener, position) → Placement { gain, pan }
         └─ Voice::set_gain(volume × gain), Voice::set_pan(pan)     two atomics
```

`spatialize` is kira's model, kept so the scenes sound as they did: the level
falls linearly in decibels from full at `Spatial::min_distance` (1) to -60 dB,
silence, at `max_distance` (100), and each ear — 0.2 apart, turned π/8 forward
— hears `(1 - strength) + strength · (facing · direction + 1) / 2` of it, with
kira's `strength` of 0.75. The two ear levels become one gain and one pan under
the mixer's pan law: the louder ear sets the gain, the ratio sets the pan.

So nothing was added to the mixing path, which still only reads a gain and a
pan per voice. A one-shot is let go when its `Voice` stops playing, and a
`SoundGroup` stops its loop when dropped, as a kira track did; a bare
`SpatialVoice`, like a `Voice`, does not.

---

### Instruments

A `Sound` is a recording. An *instrument* is a program that makes one on
demand: a [CLAP](https://cleveraudio.org) plugin, loaded from a file the
application does not ship, played with notes, and summed into the same mix as
everything else. It is behind `etib-audio`'s own default `clap` feature, and
`etib`'s `clap` feature turns it on from there; without it no plugin loader is
compiled or linked.

```rust
// Loading a plugin runs code that is not part of this program, which is why
// this is the one unsafe call in the crate.
let mut synth = unsafe { audio.load_instrument(path, &InstrumentConfig::default()) }?;

synth.note_on(0, 60, 0.8);      // channel, key, velocity
synth.note_off(0, 60);
synth.poll();                   // once a frame, on this thread
```

CLAP says which thread every call belongs on, and the split falls out the same
way the rest of this crate is already arranged:

```text
  the thread that loaded it                    the device's thread
  ─────────────────────────                    ───────────────────

  Instrument                                   Processor
    the plugin's main-thread half                its activated audio half
    its window, its timers                       its buffers and event storage
    !Send: it stays here                         moved here when it is loaded
         │                                              │
         │  note_on ─────► bounded queue, 64 ──────────►│  applied at the top
         │                                              │  of the next block
         │  Command::AddInstrument ────────────────────►│
         │                                              │
         │◄──── the processor itself ◄── on unload ─────│  destroyed where it
         │      (one slot, waited on)                   │  was made, not here
```

Two consequences an application sees:

- **`poll` has to be called**, about once a frame, from the thread that loaded
  the instrument. It is where a plugin's own callbacks, its timers and its
  window are serviced. Notes do not depend on it — an unpolled instrument still
  plays — but its editor freezes.
- **Unloading waits**, up to 250 ms, for the audio thread to hand the plugin
  back. Destroying it anywhere else would run a plugin's deallocation on the
  thread with the deadline, and CLAP asks for it not to be. That is why
  `unload` is a call that can fail rather than something that happens quietly
  in a destructor — though dropping the handle does the same thing, minus the
  error.

### Headless by default

An instrument plays without a window, and nothing creates one unless it is
asked to. `InstrumentConfig::gui` is `Gui::Headless`: the plugin's editor is
never created, and a plugin that asks to show it is told no.

`Gui::Debug` is the other setting, and it is what the name says — for hearing
what a patch is doing while building something against it, not for shipping.

Where the editor ends up is the plugin's choice more than this crate's. CLAP
has two arrangements, and only one of them is required of a plugin:

```text
  show_gui
     │
     ├─ is_api_supported(cocoa/win32/x11, floating: true)?
     │     yes → create(floating) ─ suggest_title ─ show
     │            the plugin opens its own window; nothing is owed to it
     │
     ├─ is_api_supported(…, floating: false)?
     │     yes → create(embedded) ─ get_size ─ open a window that size
     │            ─ set_parent ─ show
     │            the window is this crate's, and it resizes it when asked
     │
     └─ neither → an error saying so
```

Floating would cost this crate nothing, but plenty of real plugins — Surge XT
among them — only embed, so there has to be a window. `src/instrument/window.rs`
is the one place in the workspace that opens one without winit: an `NSWindow` on
macOS, and on every other platform an error pointing at `show_gui_in`, which
takes a window from the application instead (anything with a raw window handle,
which a winit window has). Both paths end in the same `set_parent` call.

The platform draws and clicks that window from the event loop on the thread the
instrument was loaded on, which in an `etib` application is the frame loop. A
program without one has to pump it, which is what `examples/instrument` does on
macOS so that the editor can be looked at at all.

### What a plugin may ask for, and where it is answered

A plugin calls back into the host, from whichever thread it likes. Every one of
those is recorded and acted on later, on the thread that is allowed to:

| It asks | Where it lands | What happens |
|---|---|---|
| `request_callback` | an atomic | `poll` calls it back |
| `request_restart` | an atomic | logged; this host does not rebuild a running plugin |
| `request_show` / `request_hide` | an atomic | `poll` opens or hides the editor, if it was allowed one |
| `request_resize` | an atomic | `poll` resizes the window, if the window is one this crate opened |
| `closed` | an atomic | `poll` acknowledges it and frees the editor |
| `register_timer` | the main-thread handler | `poll` ticks whatever is due, at 15 ms or slower |
| `log` | the `log` facade | forwarded wherever it came from — see below |
| `is_main_thread` / `is_audio_thread` | a stored thread id, and a thread-local | answered exactly, so a plugin's own assertions mean something |

The one thing that crosses the audio thread's line is a plugin logging from
inside its own `process`, which CLAP allows and which may take a lock on the way
out. It is forwarded anyway: the plugin has already made that choice, and
swallowing what it says would leave whoever is debugging it with nothing.

### What is not there

No presets or saved state, no MIDI beyond notes, no audio into the plugin — an
instrument's input port, if it has one, is fed silence — and only the main
output port is connected. Parameters can be listed and set; automation curves
and modulation cannot. A plugin that asks to be restarted is logged rather than
rebuilt.

`examples/test_instrument` is a CLAP instrument built from this workspace, so
that the host has something to load that nobody has to buy first: eight voices
of sine, and an editor that is a pretence — it accepts the whole protocol and
reports what it was asked, which is what makes the host's side of that
conversation testable.

---

## 7. Scenes

`etib-scene` is the cube engine on top of the core: everything it draws goes
through `etib-core`'s own types, and it knows nothing about windows or input.

```text
Scene::render(device, encoder, &mut camera)        crates/etib-scene/src/scene.rs
  │
  ├─ camera.aspect = the scene's; write CameraRaw    one uniform, Stage::All
  ├─ upload the dynamic models, if any changed       packed, group transform applied
  │
  ├─ reset the indirect draw: instance_count = 0     queued, lands first
  ├─ COMPUTE "cull"
  │    chunk pass: one thread per chunk  ──► chunk_visible[]
  │    cube pass:  one thread per cube, skips culled chunks,
  │                appends survivors to visible[], counts them atomically
  │
  └─ RENDER "scene" into Rgba16Float + depth, cleared
       static cubes   visible[] as instances, draw_indexed_indirect
       dynamic cubes  draw_indexed, no culling — there are few of them
       skybox         one triangle at the far plane, depth LessEqual, no write
```

Nothing is tone mapped here: `target_view()` is scene-referred, and the
application hands it to `PostProcess` like any other scene texture. There is one
submission per frame — the scene records into the frame's encoder.

| Part | Lives in | Notes |
|---|---|---|
| `Camera` | the application | Plain maths, no GPU. `raw()` once a frame; it remembers last frame's matrix for `prev_view_proj` |
| `CameraController` | the application | Told which way to `Move`, how far the mouse went, how far the wheel turned; which keys mean what is the application's call |
| Static cubes | the `Scene`, from construction | Sorted into 32×32 XZ chunks on the CPU once; none at all means no culling buffers are created |
| Dynamic models | the `Scene`, by id | Slots keep their ids and are reused; a fixed capacity in cubes, set at construction |
| Skybox | the `Scene`, optional | A `.hdr` is decoded on the CPU, projected onto a `TextureCube` by a compute pass at load time, then sampled linearly |

A cube is a `CubeRaw` — a model matrix and a colour, 80 bytes — read as
instance attributes at locations 3 to 7, and as a storage array by the culling
shader. The same struct is all three: the static array, the visible array the
culling writes, and the dynamic buffer.

---

## 8. Resource model

### Bindings are positional

```rust
let layout = BindLayout::build(device, "post")
    .texture(Stage::Fragment)          // @binding(0)
    .sampler(Stage::Fragment)          // @binding(1)
    .uniform::<Params>(Stage::Fragment)// @binding(2)
    .texture_3d(Stage::Fragment)       // @binding(3)
    .finish();

let bindings = layout.group(device, "post", &[source, &sampler, &uniforms, lut.view()]);
```

The nth thing declared is `@binding(n)`, and a group lists its resources in the
same order. A count mismatch panics with both numbers; an order mismatch is on
you to keep straight against the WGSL.

### Textures

| Type | Written by | Used for |
|---|---|---|
| `StorageTexture` | a compute shader (`textureStore`) | the scene. Always a linear format — storage writes are not defined for sRGB |
| `RenderTexture` | a render pass | a rasterized scene; intermediates; the interface texture |
| `DepthTexture` | a render pass with depth | the depth buffer paired with a rasterized scene. Always `Depth32Float` |
| `Texture2d` | the CPU, once at creation | images such as an HDR environment map. `Rgba16Float` |
| `TextureCube` | a compute shader, through its six-layer `storage_view` | a skybox, sampled through `view` as a `texture_cube` |
| `Texture3d` | the CPU, once at creation | lookup tables. `Rgba16Float`, filterable everywhere |

The three window-sized ones — storage, render and depth — resize with the
window and report whether the view actually changed, which is the caller's cue to rebuild bind groups. That handshake is the
one piece of protocol an application has to get right:

```rust
fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
    if self.compute.resize(&gpu.device, width, height) {   // true = new view
        self.post.rebind(&gpu.device, self.compute.target_view());
    }
}
```

### Uniforms

`UniformBuffer<T>` holds one `#[repr(C)] Pod` struct that mirrors a WGSL one,
rewritten as often as you like (`queue.write_buffer`, which lands before any
work submitted after the call, so writing just before recording the pass is
correct). WGSL's alignment rules are the caller's problem: trailing padding is
spelled out in the Rust struct, because bytemuck will not derive `Pod` for a
struct with padding it cannot see.

### Buffers

| Type | Holds | Bound as |
|---|---|---|
| `VertexBuffer<T>` | per-vertex or per-instance elements | a vertex buffer slot |
| `IndexBuffer` | `u16` or `u32` indices | the index buffer |
| `StorageBuffer<T>` | an array of `T` | `var<storage>`, read or read-write — **and** a vertex buffer slot |
| `IndirectBuffer` | `DrawIndexedArgs`, one per draw | `var<storage>` for the shader that fills it, then `draw_indexed_indirect` |

A `StorageBuffer` doubles as a vertex buffer so that a compute pass can write
the instances a later draw reads, and an `IndirectBuffer` doubles as storage so
that the same pass can write how many there are: that pair is what GPU culling
needs, and it means the CPU never waits to learn the count. Whether a shader
may write a storage binding is a property of the layout
(`storage_buffer::<T>(stage, Access::ReadWrite)`), not of the buffer.

`write` on all of them has the same semantics as on `UniformBuffer`, and
refuses — with a panic naming both numbers — a write that would run past the
end. That is also how a culling counter is put back to zero each frame:
`indirect.write(device, 0, &args)` before the pass that increments it.

### Vertex layouts and raster pipelines

`RenderPipeline::raster` takes a `RasterDesc`: the bind group layouts (the nth
is `@group(n)`), one `VertexLayout` per vertex buffer slot — a stride, a
`StepMode::{Vertex, Instance}` and `VertexAttribute { location, offset,
format }`s — plus the target format, an optional `Depth { write, compare }`,
the blend and the face culling. Counter-clockwise is the front face. These are
this crate's own types, converted to wgpu's inside, so a mesh is described
without naming the backend.

A pipeline with depth has to be drawn in a pass opened by
`Encoder::render_pass_with_depth`, and one without in a pass opened by
`render_pass`; the backend validates the pairing.

### Passes

`Encoder` is one per frame. `compute_pass` and `render_pass` return guards that
end when dropped; internally they erase the wgpu pass lifetime and re-borrow the
encoder through `PhantomData`, so the safe API keeps the borrow that wgpu's own
type would have given you. `dispatch_grid` rounds the workgroup count up and
leaves the shader to discard the overhang.

`dispatch` and `draw` set a pipeline and `@group(0)` and run in one call, which
is all a single-group pass needs. Passes with several groups or with buffers
use the long form: `set_pipeline`, `set_bind_group(index, …)`, then
`dispatch_workgroups` — or, in a render pass, `set_vertex_buffer`,
`set_index_buffer`, and `draw_indexed` / `draw_indexed_indirect`.

---

## 9. Conventions

**Errors.** `etib-core` has one error type carrying a static context string and
the backend's error as its `source`, so `{:#}` under anyhow prints the chain.
`etib-audio` has the same arrangement for the same reason — a leaf crate should
not make its callers match on cpal's errors or the filesystem's. The `etib`
layer uses anyhow directly. A shader that fails to compile *panics* rather
than returning `Err`: shaders are compiled into the binary, so a bad one is a
bug, not a runtime condition. An uncaptured GPU error is logged by a handler
installed with the device.

**Logging.** Everything goes through the `log` facade and no crate installs a
logger. `info` for adapter, surface, device and pipeline lifecycle; `debug` for
resource allocation and rebinds; `trace` for per-frame detail. Nothing is logged
from the audio thread, at any level.

```sh
RUST_LOG=etib=debug,etib_core=debug,etib_audio=debug cargo run --example chart
```

**Labels.** Every GPU object is labelled with the same string used in the logs,
so a validation message, a log line and a capture all name the same thing.

---

## 10. Writing an application

```rust
impl App for MyApp {
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool { … }
    fn device_event(&mut self, event: &DeviceEvent) { … }             // raw mouse motion
    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) { … }  // rebind here
    fn update(&mut self, dt: f32) { … }                             // no GPU work
    fn render(&mut self, frame: Frame<'_>) { … }                    // record passes
}

etib::run(Config { title, hdr: Hdr::Allow, ..Default::default() },
          |gpu| Ok(MyApp::new(gpu)))
```

Sound is not part of that: an application that makes a noise builds an `Audio`
in `MyApp::new` and keeps it in `MyApp`, and plays from `update`, from
`window_event`, or from anywhere else it likes.

`build` is `FnMut` because the platform may tear the window down and resume it,
so anything expensive it captures should be cheap to clone or rebuild.

The two examples are the reference:

- **`examples/triangle`** — a rotating triangle rasterized in a compute shader,
  driven brighter than the display can show so the tone curve has something to
  do. The smallest complete application.
- **`etib-audio/examples/beep`** — every waveform, then what the playback
  controls do to them. It opens no window, so it is also the quickest way to
  hear whether a machine's audio works at all:
  `cargo run -p etib-audio --example beep`.
- **`etib-audio/examples/spatial`** — a sound circling the listener, then
  walking away until it fades out; with headphones, the quickest check of the
  spatial layer. `cargo run -p etib-audio --example spatial`, or
  `--features mp3 -- file.mp3` to circle a file.
- **`examples/cube`** — 4096 instanced cubes, depth-tested into an
  `Rgba16Float` render texture and tone mapped by `PostProcess`. The
  smallest use of the raster side: the instances live in a `StorageBuffer` and
  the draw is indirect, which is the shape a culling pass slots in front of.
  Its window title shows the frame rate, and it needs neither `ui` nor
  `audio`.
- **`etib-scene/examples/hello`** — the port of the engine's `hello`: a
  generated terrain of 10 000 cubes (or `--static <file.model>`), a cat
  orbiting it as a dynamic model, and an HDR skybox, culled on the GPU and
  tone mapped by `PostProcess`. Keys and mouse go through `etib_input::Input`;
  `C` captures the mouse to look around. `--fullscreen` and `--no-vsync` set
  the matching `Config` fields; the title shows the frame rate and the cube
  counts. `cargo run -p etib-scene --example hello`.
- **`etib-input/examples/gamepads`** — prints what connected gamepads do,
  without a window: `cargo run -p etib-input --example gamepads --features gamepad`.
- **`examples/chart`** — four zones of known scene-referred values, so the
  differences between the curves are something you can look at rather than
  reason about. The post-processing pass is the only thing standing between
  those values and the window.

All of them take `--sdr` / `--hdr` to force the surface either way; the triangle
and the chart also take `--lut teal|<file.cube>` to grade through a table. `examples/common/lut.rs` is
shared between them by `#[path]`.

---

## 11. File map

| Path | What lives there |
|---|---|
| `crates/etib-core/src/instance.rs` | Backend connection; adapter and device opening; the limits asked for |
| `crates/etib-core/src/device.rs` | Device + queue handle, encoder creation, submission |
| `crates/etib-core/src/surface.rs` | Swapchain config, format/colour-space choice, acquire |
| `crates/etib-core/src/display.rs` | `ColorSpace`, `Hdr`, `Display`, `Output` |
| `crates/etib-core/src/texture.rs` | Formats, `StorageTexture`, `RenderTexture`, `DepthTexture`, `Texture2d`, `TextureCube`, `Texture3d`, samplers |
| `crates/etib-core/src/binding.rs` | `BindLayout` builder, `BindGroup`, `Bindable` |
| `crates/etib-core/src/buffer.rs` | `UniformBuffer<T>`, `VertexBuffer<T>`, `IndexBuffer`, `StorageBuffer<T>`, `IndirectBuffer`, `DrawIndexedArgs` |
| `crates/etib-core/src/pipeline.rs` | `ComputePipeline`, `RenderPipeline::{screen, raster}`, `RasterDesc`, vertex layouts, `Depth`, `Cull`, `Blend` |
| `crates/etib-core/src/encoder.rs` | `Encoder`, `ComputePass`, `RenderPass`, `Load`, `DepthLoad`, `Color` |
| `crates/etib-core/src/test_util.rs` | Tests only: a windowless device with the same limits, or a skip without a GPU |
| `crates/etib-core/src/shader.rs` | WGSL module compilation |
| `crates/etib-core/src/error.rs` | The single error type |
| `crates/etib/src/app.rs` | `Config`, `App`, `Frame`, `FrameStats`, `run`, the winit runner |
| `crates/etib/src/gpu.rs` | `Gpu`: window + device + surface |
| `crates/etib/src/post.rs` | `Tonemap`, `Grade`, `PostProcess` |
| `crates/etib/src/lut.rs` | `Lut`: identity, `from_rgb`, `.cube` parser |
| `crates/etib/src/blit.rs` | The 1:1 presentation pass |
| `crates/etib/src/shaders/` | `post.wgsl`, `blit.wgsl` |
| `crates/etib-scene/src/scene.rs` | `Scene`: what it owns, and the order a frame is recorded in |
| `crates/etib-scene/src/camera.rs` | `Camera`, `CameraRaw`, `CameraController`, `Move` |
| `crates/etib-scene/src/cube.rs` | The cube mesh, `Vertex`, `CubeRaw` and their vertex layouts |
| `crates/etib-scene/src/model.rs` | `.model` files: `ModelCube`, `parse_model`, `load_model`, `save_model` |
| `crates/etib-scene/src/dynamic.rs` | `DynamicModel`, and the slots the scene keeps them in |
| `crates/etib-scene/src/culling.rs` | Chunking on the CPU, the two culling passes, the static cubes' buffers |
| `crates/etib-scene/src/skybox.rs` | `.hdr` decoding, the equirectangular projection, the skybox pass |
| `crates/etib-scene/src/shaders/` | `cube.wgsl`, `cull.wgsl`, `chunk_cull.wgsl`, `skybox.wgsl`, `equirectangular.wgsl` |
| `crates/etib-input/src/input.rs` | `Input`: keyboard and mouse state, the per-frame cycle, focus loss |
| `crates/etib-input/src/held.rs` | `Held<T>`: down / just pressed / just released, shared by keys and buttons |
| `crates/etib-input/src/gamepad.rs` | `Gamepads`, `Gamepad`, the dead zone; gilrs, behind the `gamepad` feature |
| `crates/etib-physics/src/lib.rs` | `RigidBody`, `apply_cube_collision`, `GRAVITY`; the engine's constants and its cgmath slerp |
| `crates/etib-ui/src/layer.rs` | `EguiLayer`: input bridge and paint pass |
| `crates/etib-ui/src/composite.rs` | The standard-range interface texture and its composite |
| `crates/etib-audio/src/engine.rs` | `Audio`, `Play`, `Voice`; opening the device and the callback |
| `crates/etib-audio/src/mixer.rs` | The audio thread's half: voices, resampling, summing |
| `crates/etib-audio/src/sound.rs` | `Sound`, `Waveform`, tone generation |
| `crates/etib-audio/src/wav.rs` | The WAV reader |
| `crates/etib-audio/src/mp3.rs` | `Sound::from_mp3`, behind the `mp3` feature |
| `crates/etib-audio/src/spatial/mod.rs` | `Listener`, `Spatial`, `spatialize`, `SpatialVoice` |
| `crates/etib-audio/src/spatial/{strategy,group,manager}.rs` | `SoundStrategy`, `SoundGroup`, `SoundManager`: the engine's spatial sound |
| `crates/etib-audio/src/instrument/mod.rs` | `Instrument`, `InstrumentConfig`, `Gui`; loading and unloading |
| `crates/etib-audio/src/instrument/host.rs` | What a CLAP plugin may ask of this host, and on which thread |
| `crates/etib-audio/src/instrument/processor.rs` | A plugin's audio half: events in, a block out |
| `crates/etib-audio/src/instrument/window.rs` | The debug window an embedded editor is put in |
| `crates/etib-audio/src/error.rs` | The single error type |

---

## 12. Invariants worth knowing

- **Bind order is positional.** Declaration order in `BindLayout` *is* the
  `@binding` index, and the group's slice must match it.
- **Resize means rebind.** A texture that actually resized has a new view, and
  every bind group over it is stale.
- **The presentation pass clears its target**, so it must run before anything
  that paints on top.
- **The interface is drawn after the grade, never through it** — its colours are
  already display-referred.
- **`Output` is per-frame information, not startup information.** Poll it.
- **Depth goes with its colour target.** A `DepthTexture` is resized with the
  render texture it is paired with, and a pipeline created with a `Depth` is
  only drawn in a pass opened with one.
- **Eight storage buffers per stage is a requirement, not a hope.** An adapter
  that offers fewer is refused when the device is opened; nothing past that
  point checks again.
- **What a draw consumes from a compute pass stays on the GPU.** Instances in
  a `StorageBuffer` and counts in an `IndirectBuffer` are written and read in
  the same submission; nothing reads them back to decide what to draw.
- **One camera uniform, every stage.** It is laid out with `Stage::All`, so the
  culling passes, the cubes and the skybox share one bind group as `@group(0)`.
- **The culling counter is zeroed before the culling runs**, by a queued write
  that lands ahead of the frame's single submission. Recording a second frame
  into the same submission would need the reset in the encoder instead.
- **The scene clears its own target and depth**, so it is the first pass that
  touches them each frame.
- **`Input::clear_frame_state` runs once per frame, last in `update`.** Read
  before it, the "just" answers cover the frame; called twice, a press is lost.
- **No gamepad is not an error.** Gilrs failing to start leaves an empty
  `Gamepads`; nothing panics or returns an error for it.
- **Losing focus releases everything held**, keys and mouse buttons, since
  their releases go to another window; a gamepad that disconnects releases its
  buttons the same way.
- **The scene texture must be a float format** if anything is written above 1.0;
  `Rgba8Unorm` would clip before the tone curve ever sees it.
- **A LUT is indexed in gamma-encoded space**, and its entries are gamma-encoded
  too.
- **Shaders compile when the module is created, and panic on failure** — a WGSL
  typo is a startup crash, not a runtime error you can handle.
- **Spatial sound is worked out on the application's thread.** A
  `SpatialVoice` only ever sets a voice's gain and pan, once a frame; the
  listener never reaches the audio thread.
- **The audio thread never allocates, locks, logs or frees.** Anything added to
  the mixing path has to hold to that, including what it drops. A plugin's own
  `log` calls are the single exception, and they are the plugin's choice.
- **A plugin is destroyed where it was made, and stopped where it was run.**
  `stop_processing` happens on the audio thread, before the plugin is handed
  back to be deactivated and destroyed on the thread that loaded it. Plugins
  check this, and say so when a host gets it wrong.
- **An instrument is headless unless it was loaded asking for a window**, and
  a plugin asking for one anyway is refused.
- **Dropping a `Voice` does not stop the sound.** `stop` does.
- **A missing audio device is not a reason to fail.** `Audio::silent` is a
  working object that plays nothing.
