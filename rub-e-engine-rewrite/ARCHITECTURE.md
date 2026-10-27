# etib architecture

etib is a small framework for windows whose contents are drawn by a compute
shader and presented through a colour-managed pipeline. It opens the window,
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

Four crates, with a strict dependency rule: each knows about the layer below it
and nothing about the layer above. Two of them are optional, and one of those —
`etib-audio` — has nothing below it at all.

```text
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ your application  (examples/triangle, examples/chart)                     │
  │   implements App, owns its compute pass, its state and its sounds         │
  └───────────┬────────────────────────┬──────────────────────────┬───────────┘
              │                        │                          │
  ┌───────────▼──────────┐  ┌──────────▼──────────┐  ┌────────────▼───────────┐
  │ etib                 │  │ etib-ui (optional)  │  │ etib-audio (optional)  │
  │   window, frame loop │  │   egui context and  │  │   Audio, Sound, Voice  │
  │   App, Blit,         │  │   paint, the HDR    │  │   instruments, the     │
  │   PostProcess, Lut   │  │   composite pass    │  │   mixer, the device    │
  │   knows winit        │  │   knows egui, winit │  │   knows cpal, CLAP,    │
  │                      │  │                     │  │   and hides them       │
  └───────────┴──────────┘  └──────────┴──────────┘  └────────────────────────┘
              │                        │                nothing underneath it:
  ┌───────────▼────────────────────────▼──────────┐     no GPU in its tree
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
| `etib-ui` | egui integration and the interface composite | the frame loop — it is usable with a loop of your own |
| `etib-audio` | The output device, the mixer, sounds in memory, and CLAP instruments | everything else in the workspace, the GPU included |

**wgpu is an implementation detail of `etib-core`.** Nothing in its API takes or
returns a wgpu type, the errors are its own, and the async device handshake is
done for the caller. Every wrapper still exposes a `raw()` / `raw_mut()`
accessor, because some libraries have to be handed the real objects —
`etib-ui` is the one place in the workspace that uses them, since `egui-wgpu`
speaks wgpu directly. It borrows through those accessors rather than holding its
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
       ├─ create_window(title, size)
       ├─ Gpu::new -> Instance::open(window, w, h, hdr)
       │     ├─ create_surface
       │     ├─ request_adapter(HighPerformance, compatible surface)
       │     ├─ request_device(no optional features, downlevel limits)
       │     └─ Surface::new -> select_output() -> configure()
       │
       └─ build(&gpu) -> App          your pipelines are created here
```

`resumed` can fire again if the platform tears the window down and brings it
back, which is why `build` is `FnMut` and why everything above it is rebuilt
rather than reused.

Two deliberate choices in there:

- **No optional features, downlevel default limits** (`instance.rs:77`). A
  compute pass and a fullscreen draw need nothing more, and asking for nothing
  keeps the same binary working across backends.
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

Present mode is `Mailbox` when offered, otherwise `Fifo`, with a frame latency
of 2.

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

---

## 4. The render pipeline

The canonical frame is three passes. Only the middle one belongs to the
framework; the first is yours and the third is optional.

```text
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 1. COMPUTE                                          your code        │
  │    dispatch_grid over an Rgba16Float StorageTexture                  │
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

## 7. Resource model

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
| `RenderTexture` | a render pass | intermediates; the interface texture |
| `Texture3d` | the CPU, once at creation | lookup tables. `Rgba16Float`, filterable everywhere |

The first two resize with the window and report whether the view actually
changed, which is the caller's cue to rebuild bind groups. That handshake is the
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

### Passes

`Encoder` is one per frame. `compute_pass` and `render_pass` return guards that
end when dropped; internally they erase the wgpu pass lifetime and re-borrow the
encoder through `PhantomData`, so the safe API keeps the borrow that wgpu's own
type would have given you. `dispatch_grid` rounds the workgroup count up and
leaves the shader to discard the overhang.

---

## 8. Conventions

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

## 9. Writing an application

```rust
impl App for MyApp {
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool { … }
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
- **`examples/chart`** — four zones of known scene-referred values, so the
  differences between the curves are something you can look at rather than
  reason about. The post-processing pass is the only thing standing between
  those values and the window.

Both take `--sdr` / `--hdr` to force the surface either way, and
`--lut teal|<file.cube>` to grade through a table. `examples/common/lut.rs` is
shared between them by `#[path]`.

---

## 10. File map

| Path | What lives there |
|---|---|
| `crates/etib-core/src/instance.rs` | Backend connection; adapter and device opening |
| `crates/etib-core/src/device.rs` | Device + queue handle, encoder creation, submission |
| `crates/etib-core/src/surface.rs` | Swapchain config, format/colour-space choice, acquire |
| `crates/etib-core/src/display.rs` | `ColorSpace`, `Hdr`, `Display`, `Output` |
| `crates/etib-core/src/texture.rs` | Formats, `StorageTexture`, `RenderTexture`, `Texture3d`, samplers |
| `crates/etib-core/src/binding.rs` | `BindLayout` builder, `BindGroup`, `Bindable` |
| `crates/etib-core/src/buffer.rs` | `UniformBuffer<T>` |
| `crates/etib-core/src/pipeline.rs` | `ComputePipeline`, `RenderPipeline::screen`, `Blend` |
| `crates/etib-core/src/encoder.rs` | `Encoder`, `ComputePass`, `RenderPass`, `Load`, `Color` |
| `crates/etib-core/src/shader.rs` | WGSL module compilation |
| `crates/etib-core/src/error.rs` | The single error type |
| `crates/etib/src/app.rs` | `Config`, `App`, `Frame`, `FrameStats`, `run`, the winit runner |
| `crates/etib/src/gpu.rs` | `Gpu`: window + device + surface |
| `crates/etib/src/post.rs` | `Tonemap`, `Grade`, `PostProcess` |
| `crates/etib/src/lut.rs` | `Lut`: identity, `from_rgb`, `.cube` parser |
| `crates/etib/src/blit.rs` | The 1:1 presentation pass |
| `crates/etib/src/shaders/` | `post.wgsl`, `blit.wgsl` |
| `crates/etib-ui/src/layer.rs` | `EguiLayer`: input bridge and paint pass |
| `crates/etib-ui/src/composite.rs` | The standard-range interface texture and its composite |
| `crates/etib-audio/src/engine.rs` | `Audio`, `Play`, `Voice`; opening the device and the callback |
| `crates/etib-audio/src/mixer.rs` | The audio thread's half: voices, resampling, summing |
| `crates/etib-audio/src/sound.rs` | `Sound`, `Waveform`, tone generation |
| `crates/etib-audio/src/wav.rs` | The WAV reader |
| `crates/etib-audio/src/instrument/mod.rs` | `Instrument`, `InstrumentConfig`, `Gui`; loading and unloading |
| `crates/etib-audio/src/instrument/host.rs` | What a CLAP plugin may ask of this host, and on which thread |
| `crates/etib-audio/src/instrument/processor.rs` | A plugin's audio half: events in, a block out |
| `crates/etib-audio/src/instrument/window.rs` | The debug window an embedded editor is put in |
| `crates/etib-audio/src/error.rs` | The single error type |

---

## 11. Invariants worth knowing

- **Bind order is positional.** Declaration order in `BindLayout` *is* the
  `@binding` index, and the group's slice must match it.
- **Resize means rebind.** A texture that actually resized has a new view, and
  every bind group over it is stale.
- **The presentation pass clears its target**, so it must run before anything
  that paints on top.
- **The interface is drawn after the grade, never through it** — its colours are
  already display-referred.
- **`Output` is per-frame information, not startup information.** Poll it.
- **The scene texture must be a float format** if anything is written above 1.0;
  `Rgba8Unorm` would clip before the tone curve ever sees it.
- **A LUT is indexed in gamma-encoded space**, and its entries are gamma-encoded
  too.
- **Shaders compile when the module is created, and panic on failure** — a WGSL
  typo is a startup crash, not a runtime error you can handle.
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
