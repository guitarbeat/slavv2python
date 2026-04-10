# Arnis Cross-Platform Architecture Notes

Date: 2026-04-08

## Scope

This note explains how [Arnis](https://github.com/louis-e/arnis) is able to run
on Windows, macOS, and Linux, with a focus on the desktop stack: Rust, Tauri,
the native webview layer, packaging, and why it can feel like "just a terminal
command" on macOS.

## Short Answer

Arnis is primarily a native Rust application. Its desktop GUI is built with
Tauri, which means:

- the core app logic is native Rust code
- the GUI is rendered with the operating system's built-in webview
- the project is compiled separately for each target OS and CPU architecture
- the same binary can behave as either a GUI app or a CLI tool

So Arnis is not "one universal script" that happens to work everywhere. It is a
set of native binaries produced from one shared Rust codebase.

## What The Stack Looks Like

At a high level, Arnis uses these layers:

1. Rust for the main application logic
2. `clap` for the command-line interface
3. Tauri for the desktop app shell and GUI window management
4. The OS-native webview for rendering the frontend
5. GitHub Actions for building separate release binaries per platform

In practice, the stack looks like this:

```text
User runs "arnis"
    |
    +-- Rust executable starts
          |
          +-- If no CLI args: launch Tauri GUI
          |
          +-- If CLI args: run command-line workflow
                    |
                    +-- Shared Rust world-generation logic
```

## The Core Is Rust

Arnis is a Rust project with a normal Cargo manifest:

- The crate is named `arnis`.
- The default feature is `gui`.
- The `gui` feature enables `tauri`, `tauri-plugin-log`,
  `tauri-plugin-shell`, `tokio`, `rfd`, and `tauri-build`.

Relevant source:

- Arnis `Cargo.toml`:
  [https://github.com/louis-e/arnis/blob/main/Cargo.toml](https://github.com/louis-e/arnis/blob/main/Cargo.toml)

From the current manifest:

- `default = ["gui"]`
- `gui = ["tauri", "tauri-plugin-log", "tauri-plugin-shell", "tokio", "rfd", "tauri-build", "bedrock"]`
- `tauri = { version = "2", optional = true }`

This means the normal build includes the GUI by default, while a CLI-only build
can be produced by disabling default features.

The README reflects exactly that:

- GUI build: `cargo run`
- CLI build: `cargo run --no-default-features -- ...`

Relevant source:

- Arnis README:
  [https://github.com/louis-e/arnis](https://github.com/louis-e/arnis)

## Tauri Is The Cross-Platform Desktop Layer

The key cross-platform piece is Tauri.

Tauri's process model is:

- a native Rust "core process" owns OS access and app lifecycle
- one or more webview processes render the frontend
- the frontend can use HTML, CSS, and JavaScript, while Rust keeps the native
  and business-logic side

Official Tauri docs say:

- each Tauri app has a core process with full OS access
- the core process spins up webview processes for the UI
- webview libraries are provided by the operating system, not bundled into the
  executable

Relevant source:

- Tauri Process Model:
  [https://v2.tauri.app/concept/process-model/](https://v2.tauri.app/concept/process-model/)

Important implications:

- Arnis can keep its heavy logic in fast native Rust code.
- The GUI can stay cross-platform without maintaining three separate native UI
  toolkits.
- The final app can be smaller than an Electron app because it does not ship a
  full Chromium runtime.

## Which Native Webview Tauri Uses On Each OS

Tauri uses the platform's native webview implementation:

- Windows: Microsoft Edge WebView2
- macOS: `WKWebView`
- Linux: `webkitgtk`

Relevant source:

- Tauri Process Model footnote:
  [https://v2.tauri.app/concept/process-model/](https://v2.tauri.app/concept/process-model/)
- Tauri Webview Versions reference:
  [https://v2.tauri.app/reference/webview-versions/](https://v2.tauri.app/reference/webview-versions/)

This is the main reason Tauri apps feel more "native binary plus system UI
layer" than "self-contained browser application."

## Arnis Uses A Tauri App Configuration

Arnis has a `tauri.conf.json` file, which is the standard Tauri app
configuration.

Notable values visible in that config:

- `identifier: "com.louisdev.arnis"`
- `build.frontendDist: "src/gui"`
- `bundle.active: true`
- `bundle.targets: "all"`

Relevant source:

- Arnis `tauri.conf.json`:
  [https://github.com/louis-e/arnis/blob/main/tauri.conf.json](https://github.com/louis-e/arnis/blob/main/tauri.conf.json)

What that tells us:

- the frontend assets live under `src/gui`
- Tauri is configured to treat that frontend as the GUI payload
- the project is set up to support Tauri bundling across platforms

## The Same Binary Can Be GUI Or CLI

One of the most interesting parts of Arnis is that it does not fully separate
"desktop app" and "CLI tool" into different programs.

Its `main.rs` does this:

- on Windows, it tries to attach back to the parent console for CLI usage
- if built with the `gui` feature and there are no command-line arguments, it
  launches the GUI
- after that, the CLI path is still available through normal argument parsing

Relevant source:

- Arnis `src/main.rs`:
  [https://github.com/louis-e/arnis/blob/main/src/main.rs](https://github.com/louis-e/arnis/blob/main/src/main.rs)

The important behavior is:

- `let gui_mode = std::env::args().len() == 1;`
- if `gui_mode`, call `gui::run_gui()`
- otherwise continue through `run_cli()`

This explains your macOS experience very well:

- you ran a single executable from Terminal
- that executable is still a native desktop app binary
- with no args, it switches into GUI mode
- so it feels like "just a terminal command," but the command is launching a
  desktop UI through Tauri

In other words, the terminal is just one way to launch the native app.

## Why It Feels Especially Terminal-Like On macOS

There are two big reasons:

1. Arnis's release workflow packages a directly runnable macOS binary.
2. The same binary is intentionally designed to support both CLI and GUI modes.

In the current release workflow, the platform asset names are:

- `arnis-windows.exe`
- `arnis-linux`
- `arnis-mac-universal`

Relevant source:

- Arnis release workflow:
  [https://github.com/louis-e/arnis/blob/main/.github/workflows/release.yml](https://github.com/louis-e/arnis/blob/main/.github/workflows/release.yml)

In that workflow:

- Windows is uploaded as `arnis-windows.exe`
- Linux is uploaded as `arnis-linux.tar.gz`
- macOS is uploaded as `arnis-mac-universal.tar.gz`

That means the macOS release path is centered on a binary artifact inside an
archive rather than only on a traditional `.app` bundle or `.dmg`, which
naturally makes terminal-style launching feel normal.

## How Arnis Is Built For Multiple Operating Systems

Arnis uses a GitHub Actions release workflow that builds separate binaries for
separate targets:

- Windows: `x86_64-pc-windows-msvc`
- Linux: `x86_64-unknown-linux-gnu`
- macOS Intel: `x86_64-apple-darwin`
- macOS ARM64: `aarch64-apple-darwin`

Relevant source:

- Arnis release workflow:
  [https://github.com/louis-e/arnis/blob/main/.github/workflows/release.yml](https://github.com/louis-e/arnis/blob/main/.github/workflows/release.yml)

The workflow does not build one magical cross-OS artifact. It compiles a native
binary for each target. That is the normal Rust approach.

The release pipeline then:

- builds each OS target with `cargo build --release --target ...`
- renames the output to a platform-specific asset name
- packages non-Windows binaries as `.tar.gz`
- uploads build artifacts
- creates a GitHub release with Windows, Linux, and macOS assets

## macOS Universal Binary

For macOS, Arnis goes one step further:

- it builds one Intel binary
- it builds one ARM64 binary
- it combines them with `lipo -create`
- it publishes `arnis-mac-universal`

That is how one download can run on both older Intel Macs and Apple Silicon
Macs.

This is a classic macOS universal-binary workflow, not a Tauri-specific trick.
Tauri makes the app cross-platform at the framework layer; Rust plus the build
pipeline produce the architecture-specific binaries.

## Linux-Specific Notes

Linux is where the native-webview model becomes most visible.

In the release workflow, the Linux build installs:

- `libgtk-3-dev`
- `libglib2.0-dev`
- `libsoup-3.0-dev`
- `libwebkit2gtk-4.1-dev`

Relevant source:

- Arnis release workflow:
  [https://github.com/louis-e/arnis/blob/main/.github/workflows/release.yml](https://github.com/louis-e/arnis/blob/main/.github/workflows/release.yml)

That is a strong signal that the Linux GUI depends on the GTK/WebKitGTK stack
that Tauri uses on Linux.

So on Linux, cross-platform support does not mean "no native dependencies." It
means Arnis is written against Tauri's abstractions while still relying on the
Linux desktop stack underneath.

## Windows-Specific Notes

Arnis includes Windows-specific console handling in `main.rs`:

- it imports `AttachConsole` and `FreeConsole`
- it reattaches to the parent console for CLI behavior

That helps the same executable behave more naturally when used from a Windows
terminal.

The release workflow also self-signs the Windows executable during CI before it
is attached to the GitHub release.

## What Tauri Does And Does Not Do Here

Tauri does:

- provide the cross-platform desktop shell
- open native application windows
- host the frontend in the OS webview
- bridge frontend and Rust/native code
- support platform-aware packaging and bundling configuration

Tauri does not:

- compile one binary that runs unchanged on every OS
- eliminate the need for per-platform builds
- replace the operating system's native webview stack

Rust and Cargo handle native compilation. GitHub Actions handles matrix builds.
Tauri handles the desktop application shell and frontend/native integration.

## One Subtle But Important Detail

Arnis is configured like a Tauri app, but its public release workflow currently
publishes raw compiled executables rather than using a full Tauri-generated
installer flow in the GitHub release pipeline.

So there are really two related ideas here:

- Tauri as the app framework and runtime model
- GitHub Actions plus `cargo build` as the practical release mechanism

That distinction helps explain why the download can feel closer to a "binary you
run" than a traditional consumer desktop installer.

## Bottom Line

Arnis runs on Windows, macOS, and Linux because it combines:

- native Rust application code
- a Tauri desktop shell
- the OS-native webview on each platform
- separate compiled binaries for each target
- a release pipeline that builds and publishes those binaries

And it feels like a terminal command on macOS because the downloaded artifact is
a native executable that supports both CLI and GUI behavior. Running `arnis`
from Terminal is still launching the desktop app; it just happens through the
same binary that also supports command-line usage.

## Sources

- Arnis repository:
  [https://github.com/louis-e/arnis](https://github.com/louis-e/arnis)
- Arnis releases:
  [https://github.com/louis-e/arnis/releases](https://github.com/louis-e/arnis/releases)
- Arnis `Cargo.toml`:
  [https://github.com/louis-e/arnis/blob/main/Cargo.toml](https://github.com/louis-e/arnis/blob/main/Cargo.toml)
- Arnis `src/main.rs`:
  [https://github.com/louis-e/arnis/blob/main/src/main.rs](https://github.com/louis-e/arnis/blob/main/src/main.rs)
- Arnis `tauri.conf.json`:
  [https://github.com/louis-e/arnis/blob/main/tauri.conf.json](https://github.com/louis-e/arnis/blob/main/tauri.conf.json)
- Arnis release workflow:
  [https://github.com/louis-e/arnis/blob/main/.github/workflows/release.yml](https://github.com/louis-e/arnis/blob/main/.github/workflows/release.yml)
- Tauri Process Model:
  [https://v2.tauri.app/concept/process-model/](https://v2.tauri.app/concept/process-model/)
- Tauri Webview Versions:
  [https://v2.tauri.app/reference/webview-versions/](https://v2.tauri.app/reference/webview-versions/)
