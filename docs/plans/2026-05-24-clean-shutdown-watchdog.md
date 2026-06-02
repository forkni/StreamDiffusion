# Fix: clean shutdown — replace the 100ms `os._exit()` guillotine with a watchdog

## Context

The stream now launches, runs, and stops correctly. But at shutdown the console shows a
hard-exit fallback instead of a graceful exit:

```
02:03:13 - OSCHandler - INFO - Stopping OSC server...
Forcing exit...
Press any key to continue . . .
```

"Forcing exit..." means the process was killed with `os._exit(0)` mid-cleanup — `finally:`
blocks, `atexit`, sockets, and CUDA/SHM teardown were all bypassed. (`Press any key to
continue` is just the `Start_StreamDiffusion.bat` `pause` after the process dies.) We want
the normal `/stop` to exit gracefully without forcing, while keeping a safety net for the one
case that genuinely needs it.

## Root cause (confirmed by reading all three files verbatim)

`request_shutdown()` in `Scripts/streamdiffusionTD__Text__td_main__td.py:498-510` is called on
**every** OSC `/stop`. It sets `self.shutdown_requested = True` **and unconditionally** spawns a
daemon thread that sleeps 100 ms, prints "Forcing exit...", and calls `os._exit(0)`:

```python
498  def request_shutdown(self):
499      """Request application shutdown (called by OSC /stop command)"""
500      print("\n\033[31mStop command received via OSC\033[0m")
501      self.shutdown_requested = True
502
503      # Force immediate exit if stuck in model loading (can't gracefully shutdown)
504      # Use threading to allow OSC response to be sent before exit
505      def force_exit():
506          time.sleep(0.1)  # Give OSC handler time to send response
507          print("Forcing exit...")
508          os._exit(0)  # Hard exit (bypasses cleanup but works when blocked)
509
510      threading.Thread(target=force_exit, daemon=True).start()
```

The comment claims this is conditional on "stuck in model loading," but **there is no `if`
guard** — it fires on every `/stop`. Meanwhile the *intended* graceful path runs on the main
thread: setting the flag breaks `_wait_for_shutdown()`'s `while not self.shutdown_requested`
poll loop (`td_main.py:512-520`) → `start()`'s `finally:` → `shutdown()`
(`td_main.py:479-490`), which calls `manager.stop_streaming()` (a 2.0s join) then
`osc_handler.stop()` (another 2.0s join). 100 ms is never enough for that, so the force timer
beats it and kills the process right after "Stopping OSC server...".

**Key fact:** every thread in all three files is `daemon=True` (heartbeat, VRAM monitor, OSC
server, OSC batch, streaming loop, force_exit) — see audit table below. So **no thread blocks
a clean interpreter exit**. Once the main thread finishes `shutdown()` and `main()` returns,
the process exits 0 on its own; the `os._exit()` is unnecessary in the normal case. The one
case it legitimately guards: `/stop` arriving while the main thread is blocked inside
`manager.start_streaming()` building a TensorRT engine (minutes) — there the flag can't be
observed, so a fallback force-kill is needed.

### Thread/exit audit (all daemon — none block exit)

| Thread | File:line | Stop signal | Joined? |
|---|---|---|---|
| heartbeat | td_main.py:45 | `_heartbeat_running` flag | no (daemon) |
| VRAM monitor | td_main.py:94-99 | `_vram_monitor_running` flag | no (daemon) |
| force_exit | td_main.py:510 | n/a — calls `os._exit(0)` | n/a |
| OSC server | td_osc_handler.py:62 | `running=False` + `server.shutdown()` | yes, 2.0s |
| OSC batch | td_osc_handler.py:66 | `running` flag | no (bare local, never joined) |
| streaming loop | td_manager.py:198 | `streaming` flag | yes, 2.0s |

## Fix (approved approach — watchdog fallback)

### 1 — Replace the unconditional force with a conditional watchdog (primary)
File: `Scripts/streamdiffusionTD__Text__td_main__td.py`

**a.** Add a completion flag in `__init__` (near `self.shutdown_requested = False`, line 442):
```python
self.shutdown_requested = False
self._shutdown_complete = False
```

**b.** Set it at the end of `shutdown()` (after the "Shutdown complete" print, line 488), in a
`finally:` so the flag is set whether cleanup succeeds or raises:
```python
def shutdown(self):
    """Graceful shutdown"""
    print("\n\nShutting down...")
    try:
        self.manager.stop_streaming()
        self.osc_handler.stop()
        self.osc_reporter.stop_heartbeat()
        self.osc_reporter.stop_vram_monitoring()
        print("Shutdown complete")
    except Exception as e:
        print(f"Shutdown error: {e}")
    finally:
        self._shutdown_complete = True
```

**c.** Rewrite `request_shutdown()` (lines 498-510) so the daemon thread is a *watchdog* that
only force-exits if the graceful path hasn't completed within a generous deadline:
```python
def request_shutdown(self):
    """Request application shutdown (called by OSC /stop command)"""
    print("\n\033[31mStop command received via OSC\033[0m")
    self.shutdown_requested = True

    # Watchdog fallback: the graceful path runs on the main thread
    # (_wait_for_shutdown breaks -> start() finally -> shutdown()). Only force-exit
    # if that hasn't completed within the deadline — e.g. /stop arrived while the main
    # thread was blocked building a TensorRT engine and never observed the flag.
    def _force_exit_watchdog():
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self._shutdown_complete:
                return  # graceful shutdown finished; let the interpreter exit cleanly
            time.sleep(0.1)
        print("Forcing exit (graceful shutdown timed out)...")
        os._exit(0)

    threading.Thread(target=_force_exit_watchdog, daemon=True).start()
```

Result: normal `/stop` → main thread breaks the poll loop → `shutdown()` runs full cleanup
(<1s with fix #2) → sets `_shutdown_complete` → `main()` returns → interpreter exits 0, daemon
watchdog dies before firing. No "Forcing exit...". Stuck model-build → watchdog fires after 10s.

### 2 — Make the pause-wait loop responsive to stop (secondary, prevents join timeout)
File: `Scripts/streamdiffusionTD__Text__td_manager__td.py`, line 659.

The pause-mode acknowledgment wait can spin up to 5s and does **not** observe `self.streaming`,
so `stop_streaming()`'s 2.0s join can time out when the stream is paused at stop time. Add the
flag to the condition so a stop breaks the wait immediately:
```python
while not self.frame_acknowledged and self.streaming and time.time() < acknowledgment_timeout:
    time.sleep(0.001)
```

### 3 — Close the OSC socket on stop (secondary, hygiene)
File: `Scripts/streamdiffusionTD__Text__td_osc_handler__td.py`, `stop()` (lines 76-90).

`stop()` calls `server.shutdown()` + `join` but never `server_close()`, leaking the UDP
socket. Add it after the join, before the final log line:
```python
    if self.server_thread and self.server_thread.is_alive():
        self.server_thread.join(timeout=2.0)

    if self.server:
        self.server.server_close()

    logger.info("OSC server stopped")
```

### 4 — Apply each edit to BOTH the Scripts/ mirror and the on-disk runtime copy
Because this session's crash fix disabled `copy_sdtd_code()` from overwriting the four code
files, the on-disk `StreamDiffusionTD\*.py` is now canonical and is what Python actually
`import`s/runs; the `Scripts/` mirror is the externalized DAT text that live-syncs to the
running `.tox`. They must stay byte-identical. Edit both for each of the three files:

| Scripts/ mirror (syncs to .tox) | on-disk runtime (Python runs this) |
|---|---|
| `Scripts\streamdiffusionTD__Text__td_main__td.py` | `StreamDiffusionTD\td_main.py` |
| `Scripts\streamdiffusionTD__Text__td_manager__td.py` | `StreamDiffusionTD\td_manager.py` |
| `Scripts\streamdiffusionTD__Text__td_osc_handler__td.py` | `StreamDiffusionTD\td_osc_handler.py` |

## Out of scope (noted, not fixing now)
- Double `shutdown()` call on the SIGINT/SIGTERM path (`_signal_handler` calls `shutdown()`
  then `sys.exit(0)`, whose `SystemExit` re-triggers `start()`'s `finally → shutdown()`).
  Harmless — `stop_streaming()`/`stop()` are idempotent via their `if not running: return`
  guards. Cosmetic double "Shutting down..." only.
- The OSC `batch_thread` (td_osc_handler.py:66) is a bare local, never stored or joined. Benign
  (daemon, stops on `running=False`). Leave as-is unless tracking/joining is wanted later.

## Verification
1. Launch the stream (Start Stream in TD, Debugcmd on). Confirm normal run.
2. Press Stop in TD (OSC `/stop`). The console must now show the full graceful sequence —
   `Stop command received via OSC` → `Shutting down...` → `Stopping streaming...` /
   `Streaming stopped` → `Stopping OSC server...` → `OSC server stopped` → `Shutdown complete`
   → the `Press any key to continue` `pause`, with **NO** "Forcing exit..." line.
3. Time it: graceful shutdown should complete in well under the 10s watchdog deadline
   (expect <1-2s).
4. Pause-mode stop: pause the stream, then Stop — confirm it still shuts down promptly
   (fix #2) and doesn't sit out the 5s ack timeout.
5. (Optional) Confirm the watchdog still works as a safety net: it should only ever print
   "Forcing exit (graceful shutdown timed out)..." if cleanup genuinely stalls >10s.

## Notes / constraints
- `Scripts/` and `StreamDiffusionTD/` are at the **parent** `...\StreamDiffusion\` level,
  outside the cwd git repo; `Scripts/` edits sync live to the running `.tox` (no TOX rebuild).
- After approval, copy this plan into `StreamDiffusion\docs\plans\` per the
  save-plans-as-project-files convention.
