# Performance — Release benchmarks (T-A2-001)

## 1. Conditions

| Item | Value |
|---|---|
| Hardware | Intel i7-12650H laptop (10 cores / 16 threads), 15 GB RAM, NVIDIA RTX 3050 Laptop (driver 595.99.02) |
| OS / governor | Ubuntu 22.04.5, kernel 6.8, CPU governor `powersave` (laptop; not changed: needs root) |
| Build | `scripts/build.sh Release` → `-O3 -DNDEBUG -DEIGEN_NO_DEBUG`, GCC 14.4.0, frozen tree (21:31) |
| Command | from `build-release/`: `./a2 ../data/<dir>/<video>.MOV --scenario=N -e` (interactive=0) |
| Rendering | `vtkEGLRenderWindow` on the NVIDIA GPU (printed at start-up) |
| Load | No other project jobs or Peers running. Desktop session active. 1-min load 4–6 during the runs, mostly from `a2`'s own threads (decode, OpenCV, writer). |
| Scope | Per-frame read + process (estimation incl. feature extraction) + plot (VTK update/render/capture) + write (enqueue to the buffered writer; encoding happens on the writer thread) |

## 2. Results (final benchmark, 2026-11-06 21:32–22:03, sequential runs)

| Scenario | Video | Frames | Wall [s] | total ms/frame mean / p50 / p95 / max | process mean / p95 | plot mean / p95 (render / capture) | write mean | Loop throughput | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| 5 tags | tag_identical.MOV 1920x1080 @29.97 | 1074 | 117.3 | 108.1 / 115.3 / 156.5 / 441.4 | 65.0 / 106.7 | 38.7 / 50.0 (20.2 / 14.9) | 4.3 | 9.24 fps | 0.66 GB |
| 6 room | room.MOV 1920x1080 @29.97 | 1470 | 158.2 | 106.9 / 102.4 / 205.1 / 461.3 | 61.8 / 135.8 | 40.2 / 64.8 (21.5 / 14.5) | 4.8 | 9.34 fps | 0.60 GB |
| 4 flight | flight.MOV 2688x1512 @59.94 | 9161 | 1541.6 | 168.0 / 157.9 / 282.1 / 400.9 | 98.5 / 196.5 | 64.3 / 89.7 (33.9 / 25.9) | 5.1 | 5.95 fps | 1.04 GB |

- Exports: `out/{tag_identical,room,flight}_out.MOV`, mp4v. Reopened with ffprobe at 3840x1080 (1074 frames),
  3840x1080 (1470 frames) and 5376x1512 (9161 frames, 59.94 fps). Frame counts and fps match the inputs.
- Per-frame CSVs: `out/<stem>_timing.csv`.
- No scenario runs in real time. S5 and S6 reach about 31 % of the video rate; S4 about 10 %.

**Quality condition (AGENTS.md: performance only counts when estimates stay adequate):**

- S4 and S5 estimates stay finite with 0 MAP failures throughout (test-report §5).
- **S6 quality is INADEQUATE**, so its timing is reported for completeness only.

## 3. Run-to-run variance (honest note)

The same S4 command took 1000 s (108.6 ms/frame) at 17:25, while three other Peers were also running. In this final
idle benchmark it took 1541.6 s (168.0 ms/frame). Breakdown of the difference:

- The plot stage rose from 46.2 to 64.3 ms/frame. Its code did not change between the two runs.
- The process stage rose from 59.2 to 98.5 ms/frame. Two things changed in between:
  - the horizon classifier was converted to square-root form (two 4x4 triangular solves per analysis pixel instead of
    one precomputed quadratic form);
  - the MAP outcome bookkeeping was added.

Because unchanged code also slowed by 40 %, a large part of the difference is machine state: laptop CPU in `powersave`,
thermals, desktop activity. **I did not isolate the cost of the square-root classifier change.** Treat the numbers as
indicative with roughly ±40 % run-to-run variance on this laptop. A controlled benchmark (performance governor, pinned
clocks, repeated runs) needs root and was NOT run.

## 4. Hotspots and optimisation decisions

| Stage | Observation | Decision |
|---|---|---|
| Plot (VTK) | Was 340–670 ms/frame through X/Mesa (Intel iGPU) when `DISPLAY` was set. | `--interactive=0` forces EGL offscreen on the NVIDIA GPU: 38–64 ms/frame. Batched landmark geometry, slot reuse (a fixed leak: plot time grew 23→55 ms over a run), SIMD flipped readback. |
| Plot capture | 15–26 ms: GPU→CPU readback of a 2W x H frame. | Kept. PBO async readback would be the next step (not done). |
| S4 flow | LK at half resolution, ~400 features, keyframe every 5 frames: ~21–29 ms. | Kept; full resolution did not improve accuracy. |
| S4 horizon | Classifier at 1/4 scale plus sub-pixel refinement: ~11–15 ms. | Kept. Could be reduced by classifying only a band around the predicted horizon (not done). |
| Time update | Full n x n STM integration was O(n³): 63.5 ms at n=300. | Block-wise 12-state integration (exact): 7.0 ms at n=300 (9x). |
| Measurement update | Dense eigendecomposition per trust-region iteration. | Cholesky fast path in `funcmin`: 1.4–2.4x faster at 30–60 landmarks. Support-restricted MAP. |
| Video I/O | Threaded buffered reader/writer; read stage < 0.2 ms mean. | Kept the architecture; fixed lost-wakeup races and added a failure contract. |

## 5. Not measured

- Interactive modes 1 and 2 (X/Mesa window): about 1 s/frame in a smoke run; not benchmarked.
- Hardware profiler (perf/VTune) traces: not collected. Hotspots above come from the per-stage timers.
- Debug timings are never reported as performance.
