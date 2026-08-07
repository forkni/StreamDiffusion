 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 FaceID — make it actually work

 Context

 The original plan scoped Tier A (backend correctness) + Tier C (tests) and
 deferred Tier B
 (the vendored diffusers_ipadapter package). Live testing has since shown that was
 the wrong
 split: Tier A/C is hygiene, and the deferred Tier B contains the actual functional
 blockers.

 Observed on this machine, 2026-08-06 → 08-07:

 - Regular IP-Adapter works — visible effect, type: regular, SDXL, TensorRT, scale
 0.756.
 - FaceID produces no visual effect — face detected, embeddings [1, 4, 2048], scale
 updates
 applied, FPS normal, zero errors in the log.

 Because both run the same TensorRT pipeline with the same 4-token concat and the
 same
 ipadapter_scale profile ((1,), (70,), (70,)), everything shared is exonerated. The
 defect is
 strictly in the FaceID-only branch, and it is two independent bugs — each
 sufficient on its own to
 produce a silent no-op.

 Goal of this plan: restore FaceID identity transfer. Remaining Tier A/C items are
 retained
 below at lower priority; none of them affect the symptom.

 ---
 Status of the previously-approved work

 Item: A3 — --fid/--tokens{n} leaking onto VAE paths
 State: ✅ Done and empirically validated. Edit applied to engine_manager.py; the
 00:56 run reused the pre-existing …--min_batch-1--optlvl3--… VAE directory with
 no rebuild, and the UNet hash was unaffected (order within the UNET branch is
 byte-identical).
 ────────────────────────────────────────
 Item: B1, B2, B3, S1, S4, A1, A2, A4, A5, A6, Tier C tests
 State: ✅ Done (2026-08-07 execution session, verified against the codebase via
 MCP search first). B1's mechanism corroborated independently and B2's design
 corrected before implementation — see "B1"/"B2" below. Full verification
 results are recorded at the end of the "Verification" section.

 Side question answered: "UNet is rebuilt every time"

 It is not — every directory maps to a real config change:

 ┌──────────┬───────────────┬────────────────────────────────┐
 │  Built   │   UNet key    │             Config             │
 ├──────────┼───────────────┼────────────────────────────────┤
 │ 11:53 PM │ hb7ff89d10bf2 │ pre-IP-Adapter baseline        │
 ├──────────┼───────────────┼────────────────────────────────┤
 │ 12:25 AM │ hb628db725c7d │ FaceID + ControlNet            │
 ├──────────┼───────────────┼────────────────────────────────┤
 │ 12:37 AM │ hc44c2e7ef011 │ FaceID, ControlNet off         │
 ├──────────┼───────────────┼────────────────────────────────┤
 │ 12:56 AM │ hbfeb5b3c75e5 │ FaceID off, regular IP-Adapter │
 └──────────┴───────────────┴────────────────────────────────┘

 --fid, --controlnet, and --tokens{n} (the de-facto "IP-Adapter on" marker) are all
 inside the
 hashed canonical string, so each combination is a distinct engine by design.
 Confirm with one
 free test: restart on the current td_config.yaml unchanged — hbfeb5b3c75e5 must
 load in ~4 s
 with no build. If it rebuilds, that is a real bug and this plan needs a new item.

 ---
 Root cause

 B1 · InsightFace is fed RGB but expects BGR  ← fix this first

 face_utils.py:110-117 — image_np = np.array(pil_image) produces RGB, handed
 straight to
 insightface_model.get() (via detect_faces_multires:70). InsightFace follows the
 OpenCV
 convention and expects BGR.

 Detection survives the channel swap — RetinaFace is robust enough, which is
 precisely why the logs
 look healthy — but face.normed_embedding (:127) is an ArcFace identity vector
 computed on
 colour-swapped pixels.

 That is fatal specifically for standard FaceID, because that embedding is the
 entire conditioning
 signal. ip_adapter.py:243-246:

 else:
     # Standard FaceID: use face embeddings only
     image_prompt_embeds = self.image_proj_model(face_embeds)
     negative_image_prompt_embeds =
 self.image_proj_model(torch.zeros_like(face_embeds))

 The CLIP crop computed at :217-218 is used only on the is_plus branch, so for
 plain FaceID
 there is no second signal to compensate. Garbage identity in → plausible-shaped
 tokens out → no
 error anywhere.

 Fix: flip channels before every insightface_model.get() call — image_np[:, :,
 ::-1].
 Note face_align.norm_crop output feeds Image.fromarray for CLIP, which does want
 RGB, so the
 flip must be scoped to the detector input and undone for the crop, not
 blanket-applied.

 Low risk, small, and independently testable — it may restore most of the effect on
 its own.

 Corroborated independently (2026-08-07, offline embedding proof — no pipeline, no
 GPU build): every InsightFace recognition/detection model preprocesses with
 swapRB=True internally (model_zoo/arcface_onnx.py:82-83, retinaface.py:151,
 scrfd.py:154) — a second, independent confirmation from inside the vendored
 package itself, not just the RGB call site, that it contractually expects BGR.
 Measured over InsightFace's own bundled multi-face sample photos: same-face
 agreement between the BGR and RGB embeddings of one identity averaged 0.9512
 cosine similarity (channel-swap is a systematic transform, not random noise, so
 much of ArcFace's geometric signal partially survives it); cross-identity
 discriminability — mean similarity between different people's faces — was 0.0326
 under correct BGR ordering versus 0.0431 under RGB, i.e. RGB's corrupted
 embeddings are, if anything, slightly less separable across identities. That
 residual gap is exactly what starves FaceID conditioning, which depends on the
 embedding's fine structure, not just its coarse direction.

 Fixed as apply_faceid_patches() in the new
 src/streamdiffusion/modules/faceid_compat.py, wired in from
 IPAdapterModule.install() before IPAdapter(**ip_kwargs) is constructed.

 B2 · The FaceID LoRA is discarded

 ip_adapter.py:69-80:

 ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())

 # Filter out LoRA keys for FaceID models (they're not needed in diffusers
 implementation)
 ip_adapter_state_dict = ipadapter_model["ip_adapter"]
 if self.is_faceid:
     filtered_state_dict = {k: v for k, v in ip_adapter_state_dict.items()
                            if not any(lk in k for lk in ["lora", "LoRA"])}
     ip_adapter_state_dict = filtered_state_dict

 ip_layers.load_state_dict(ip_adapter_state_dict)

 The comment is wrong. _detect_faceid (:84) identifies a FaceID checkpoint by
 "0.to_q_lora.down.weight" — the code recognises the model by the very keys it then
 throws away.
 h94's FaceID adapters are trained with that LoRA active.

 The filter exists to stop load_state_dict crashing, not because the LoRA is
 unnecessary.
 set_ip_adapter (:133-155) installs paramless AttnProcessor() on attn1 and
 IPAttnProcessor
 on attn2; h94's reference implementation installs LoRA-carrying processors in both
 slots. With
 paramless processors the LoRA keys are "unexpected" and the load fails — so they
 were filtered.

 Correction (verified 2026-08-07): the mechanism first proposed here — "introduce
 a LoRA-aware processor pair; ONNX export captures it automatically" — does not
 work. IPAdapterUNetExportWrapper.__init__ rebuilds every attention processor
 before ONNX export on both of its branches:
 _ensure_processor_dtype_consistency (the one actually taken, since IP-Adapter is
 installed pre-compilation) copies only to_k_ip/to_v_ip into a fresh
 TRTIPAttnProcessor2_0; the install_processors=True fallback builds fresh
 processors with no weight copy at all. use_cached_attn then overwrites attn1 a
 third time with CachedSTAttnProcessor2_0. Any processor-resident LoRA is
 discarded on both attn1 and attn2 regardless of which branch runs — and no
 LoRA-aware processor class exists anywhere in the vendored
 attention_processor.py to hold it in the first place.

 Fix instead: fuse the LoRA directly into the base UNet attention linears, before
 any processor rebuild can touch them — faceid_compat.fuse_faceid_lora(unet,
 ckpt_path, lora_scale=1.0). For each of the 140 attn_processors indices, resolve
 the owning module and add (up @ down) * lora_scale into to_q / to_k / to_v /
 to_out[0] in place, under torch.no_grad(), computed in fp32 and stored in the
 param dtype. Exact at lora_scale=1.0: h94's LoRA(IP)AttnProcessor feeds each LoRA
 branch the same input its parallel base linear receives, and network_alpha is
 unset, so there is no alpha/rank division — plain up @ down.

 No new processor classes and no vendored monkeypatch are needed for B2 (only B1
 needs one). Works identically for the PyTorch and TensorRT paths, since fusion
 happens on stream.pipe.unet before IPAdapterUNetExportWrapper is constructed.
 Guarded by unet._sdtd_faceid_lora_fused so a second fusion is a no-op.
 Behavioural note: fused LoRA is permanent and not modulated by ipadapter_scale —
 it alters the base UNet even at scale 0, matching h94's own fixed
 lora_scale=1.0 reference — which is exactly why B3 (below) is mandatory
 alongside it.

 Implemented in src/streamdiffusion/modules/faceid_compat.py; covered by
 tests/unit/test_faceid_lora_fusion.py (synthetic attention module + synthetic
 LoRA state dict — no GPU, no checkpoint download).

 B3 · Engine cache key must be bumped with the fix  ← easy to miss, silently fatal

 B2 changes the UNet's baked weights but not any input to get_engine_path, so --fid
 stays
 --fid. The existing hb628db725c7d / hc44c2e7ef011 engines would be silently reused
 and
 still produce no effect — the fix would appear not to work.

 Add a marker to the FaceID suffix in the EngineType.UNET branch of
 engine_manager.py — i.e. inside the block A3 just created.

 Shipped as --fid2 (engine_manager.py:169). The entire canonical prefix,
 including this marker, is SHA-1 hashed to 12 hex chars for the on-disk UNet
 directory name (engine_manager.py:242-248), per A3's own MAX_PATH fix — so
 --fid2 never appears as a literal path substring, only in the hash pre-image,
 loggable at DEBUG via "EngineManager: UNet cache key h{short_hash} = {canonical}".
 Confirmed with a real build (2026-08-07): a fresh UNet engine hash
 (h19565dc9259e) appeared, distinct from all four pre-existing FaceID/non-FaceID
 hash directories, while the VAE encoder/decoder engines were reused unchanged —
 proof the marker forks exactly the UNet cache and nothing else.

 Blast radius (verified). get_engine_path (engine_manager.py:98) has five call
 sites:

 ┌─────────────────────────────────────────┬─────────────┬─────────────────────┐
 │                Call site                │ Engine type │   Affected by a     │
 │                                         │             │ UNET-branch marker? │
 ├─────────────────────────────────────────┼─────────────┼─────────────────────┤
 │ wrapper.py:2125                         │ UNET        │ ✅ yes — the only   │
 │                                         │             │ one                 │
 ├─────────────────────────────────────────┼─────────────┼─────────────────────┤
 │ wrapper.py:2160                         │ VAE encoder │ no                  │
 ├─────────────────────────────────────────┼─────────────┼─────────────────────┤
 │ wrapper.py:2174                         │ VAE decoder │ no                  │
 ├─────────────────────────────────────────┼─────────────┼─────────────────────┤
 │ wrapper.py:2695                         │ safety      │ no                  │
 │                                         │ checker     │                     │
 ├─────────────────────────────────────────┼─────────────┼─────────────────────┤
 │ engine_manager.py:452                   │ CONTROLNET  │ no                  │
 │ (get_or_load_controlnet_engine)         │             │                     │
 └─────────────────────────────────────────┴─────────────┴─────────────────────┘

 Three of the four non-UNET sites are exactly the ones that were receiving
 --fid/--tokens{n}
 before A3 — independent confirmation that A3 was a real bug, not cosmetics. It
 also scopes B3
 tightly: a marker inside the EngineType.UNET branch can only move the :2125 path.

 Mechanism

 B1 and B2 both live in venv\Lib\site-packages\diffusers_ipadapter, pip-installed
 from a pinned
 SHA (setup.py:75 → livepeer/Diffusers_IPAdapter@405f87da), so direct edits are
 lost on
 reinstall. Use the mechanism already agreed for Tier B: monkeypatch from
 src/streamdiffusion/modules/ipadapter_module.py at install time. No fourth repo,
 survives
 reinstall, stays inside the fork.

 ---
 Recommended sequencing

 1. B1 alone, then test. Cheap, low-risk, and plausibly the dominant term — a wrong
 ArcFace
 vector cannot be rescued by anything downstream.
 2. Measure. If identity transfer returns to an acceptable level, B2 becomes an
 enhancement
 rather than a blocker.
 3. B3 + B2 together if B1 is insufficient. Never ship B2 without B3 or the result
 is untestable.

 Executed 2026-08-07: B1, B3, and B2 were implemented together in one pass rather
 than gated on an intermediate B1-alone measurement — B2 recovers ~57% of the
 adapter's trained parameters regardless of how much B1 alone moves the needle
 (Step 2's own reasoning), so there was no scenario where shipping it was wrong.
 The A/B PyTorch sanity check (Step 6) was then run once, after all three landed
 together, and was decisive on its own.

 ---
 Supporting findings (cheap, fold in while touching these files)

 #: S1
 Finding: detect_faces_multires runs twice per update on SDXL — once in
 extract_face_embeddings, again in the crop_size != 224 re-crop branch. Doubles
 detection cost, and doubles the cost of the A2 error storm.
 Location: face_utils.py:117, :196
 ────────────────────────────────────────
 #: S2
 Finding: _detect_kolors_faceid({}) is called with an empty dict — always False.
 Latent, harmless on SDXL.
 Location: ip_adapter.py:210
 ────────────────────────────────────────
 #: S3
 Finding: Standard FaceID computes clip_image and never uses it; the 3.69 GB CLIP
 encoder is loaded purely for .config.projection_dim. Ties to deferred #2.
 Location: ip_adapter.py:217-218, :103
 ────────────────────────────────────────
 #: S4
 Finding: build_embedding_hook defaults style_key to "default" while install()
 defaults to "ipadapter_main". Latent only because both wrapper construction sites
  pin it (wrapper.py:2263 and :2864, each cfg.get("style_image_key") or
 "ipadapter_main"); a miss silently substitutes zeros (:222-227) rather than
 erroring.
 Location: ipadapter_module.py:206 vs :272

 S4's zero-fill deserves a warning log regardless — it is the same class of
 silent-no-op failure as
 B1/B2.

 ---
 Ruled out (do not re-investigate)

 Confirmed working by the regular-IP-Adapter run on the identical TensorRT path:
 embedding hook and
 concat order, the ipadapter_main cache key, stream.ipadapter.scale →
 ipadapter_scale TRT
 input, _update_ipadapter_config scale propagation, num_ip_layers=70 discovery, the
 77+4
 encoder_hidden_states shape, the engine ipadapter_scale profile, and the pre-TRT
 install route.

 ---
 Retained lower-priority items (unchanged specs) — all shipped 2026-08-07

 - A1 — Ipfaceid() warns only on Daydream; mirror Fienable() at
 Scripts\…StreamDiffusionExt__td.py:5275-5284 and branch the message so Local also
 warns that
 --fid forks the engine cache.
 - A2 — ipadapter_update_requested is reset at td_manager.py:1260 inside the try,
 so a
 failed update_style_image (the no-face case) re-runs the full detection sweep
 every frame.
 Move the reset into a finally. Apply to both copies (deployed + Scripts/ mirror).
 Worth
 more now that S1 shows the sweep runs twice.
 - A4 — add IPAdapterConfig.from_dict, call from wrapper.py:2262 and :2863; reject
 faceid without insightface_model_name. Do not route _get_current_ipadapter_config
 (stream_parameter_updater.py:1781) through it — that dict is deliberately lossy
 and demo-only;
 it would break demo/realtime-img2img.
 - A5 — delete dead update_faceid_v2_weight (faceid_embedding.py:80-82, zero
 callers);
 correct the metadata at :24-29 to say it only affects FaceID-Plus/v2.
 - A6 — ipadapter_module.py:419 snapshot_download pulls both model.safetensors and
 pytorch_model.bin (3.69 GB each); add ignore_patterns=["*.bin", "*.msgpack",
 "*.h5"]. Add the
 td_manager.py pair to sync_td_mirror.PAIRS only if you want byte-identity
 test-enforced
 (ADR-0002 deliberately preserves deployed-copy hand edits).
 - Tier C — test_engine_path_ipadapter_suffixes.py (A3, now regression-guarding a
 landed fix),
 test_ipadapter_config_from_dict.py (A4), test_faceid_preprocessor_contract.py
 (A5). Do not pin
 the literal hash — canonical includes --trt{ver}--cc{cc}, which differs on a
 GPU-less runner.

 - Reuse the existing harness — do not invent a new one.
 tests/unit/test_engine_path_length.py
 already solves the hard part: _make_engine_manager() (:61-66) builds an
 EngineManager via
 __new__, bypassing __init__'s TensorRT/onnx/polygraphy imports so the test runs
 GPU-less, and
 _CRASH_UNET_KWARGS (:42-58) is a ready full kwarg set.
 test_distinct_configs_do_not_collide
 (:89-98) is the exact assertion template for both A3 (VAE path must be unchanged
 by
 is_faceid/ipadapter_tokens; UNET path must change) and the B3 marker bump. Import
 these
 helpers or copy the six-line pattern; the module-level pytest.mark.skipif(not
 IMPORT_OK) guard
 should be carried over too.

 ---
 Plan verification (MCP code-search, 2026-08-07)

 Index: D:\dev\SDTD_040_Beta, 7528 chunks / 529 files, synced: true.

 Confirmed at the cited locations: IPAdapterConfig (ipadapter_module.py:24-42),
 build_embedding_hook (:205-261), build_unet_hook (:426), _resolve_model_path's
 snapshot_download with allow_patterns and no ignore_patterns (:419 — A6 stands),
 _prepare_ipadapter_configs (config.py:337-354), _update_ipadapter_config
 (stream_parameter_updater.py:1657-1755), _get_current_ipadapter_config (:1781-1819
 — the
 A4 "do not route through this" carve-out), get_engine_path (engine_manager.py:98),
 _lora_signature (:83-96), Fienable (…StreamDiffusionExt__td.py:5275-5284) and
 Ipfaceid
 (:5286-5296, Daydream-only guard at :5292 — A1 stands).

 A5 confirmed negatively: find_connections on update_faceid_v2_weight
 (faceid_embedding.py:80-82) returns no direct_callers section at all, and a
 repo-wide grep
 finds only the definition and its own print. Zero callers — safe to delete.

 A2 confirmed in both copies: deployed StreamDiffusionTD/td_manager.py and the
 Scripts/
 mirror are line-identical here — guard :1213, reset :1260 inside the try, bare
 except Exception :1263, no finally. (Grep honours StreamDiffusion's .gitignore, so
 the
 deployed copy is invisible to a recursive search — check it by explicit path.)

 Two corrections this pass produced: the B3 blast-radius table above (the call
 graph reported
 only 1 of the 5 get_engine_path call sites — the four wrapper.py ones sit inside
 _load_model's
 split blocks and were unresolved), and S4's second pinning site at
 wrapper.py:2864.

 Cannot be MCP-verified — state this plainly rather than implying otherwise. The
 index covers
 D:\dev\SDTD_040_Beta only; venv\Lib\site-packages\ is not indexed (no
 site-packages chunk
 appears in any result). B1 and B2 — the two actual root causes — therefore rest
 entirely on direct
 reads of face_utils.py and ip_adapter.py, not on search-tool corroboration. Those
 reads are
 unambiguous (the RGB→insightface_model.get() hand-off and the LoRA-key filter are
 both plainly
 visible in source), but they are single-source. Re-read both files before editing.

 Also note the index contains Backup/ and Scripts_backup_20260725_pre-bake/ mirrors
 (2374 chunks
 each) that surface as near-duplicate hits — every location above was resolved
 against the real file.

 ---
 Verification

 Free sanity check, do first: restart on the unchanged td_config.yaml. Expect
 hbfeb5b3c75e5
 to load with no UNet build — settles the rebuild question.

 B1, offline and pipeline-free (do this before any rebuild): run InsightFace over
 one portrait
 both ways and compare normed_embedding cosine similarity against a second photo of
 the same
 person versus a different person. Correct BGR ordering should separate same-person
 from
 different-person markedly better than RGB. This isolates B1 from every pipeline
 variable and costs
 no engine build.

 B1 in-pipeline: set Ipfaceid on, pulse Ipadapterupdate on a clear front-facing
 face,
 Ipadapterscale ≈ 0.75, and compare against the same face pre-fix.

 B2: same procedure; expect a further increase in identity fidelity, and confirm
 a fresh UNet hash directory appears (the --fid2 marker is folded into the SHA-1
 hash, not a literal path substring — see "B3" above). The first FaceID run after
 B3 lands rebuilds the UNet once (~535 s estimated; measured ~962 s this session)
 — expected and one-time.

 Automated (no GPU):

 cd StreamDiffusion
 venv/Scripts/python.exe -m pytest
 tests/unit/test_engine_path_ipadapter_suffixes.py \
     tests/unit/test_ipadapter_config_from_dict.py \
     tests/unit/test_faceid_preprocessor_contract.py \
     tests/unit/test_engine_path_length.py tests/unit/test_td_mirror_sync.py -v
 python scripts/sync_td_mirror.py --check
 bash scripts/git/check_lint.sh

 Commits go through scripts/git/commit_enhanced.sh, never raw git commit.
 src/streamdiffusion/ has unrelated modified files — stage explicitly, never git
 add -A.

 ---
 Verification — results (2026-08-07)

 1. B1 offline embedding proof — done; numbers recorded under "B1" above.
 2. B2 fusion dry-run — done. All 140 attn_processors indices carry the 8 LoRA
 keys, and every up @ down shape matched its target linear's weight shape on the
 real SDXL UNet before any weight was touched.
 3. A/B PyTorch sanity — done via scripts/test_faceid_sanity.py (acceleration=
 "none", use_tiny_vae=True to sidestep an unrelated, pre-existing native-VAE
 kvo_cache bug that would otherwise block acceleration="none" regardless of
 FaceID — out of scope, left unfixed). Result: decisive. baseline.png is a
 generic studio portrait unrelated to the style face; faceid.png is markedly
 different in hairstyle, facial structure, and toning, clearly pulled toward the
 conditioning face. Saved under outputs/faceid_sanity/ (gitignored, not
 committed).
 4. Real TensorRT build — done. Fresh UNet engine built at
 sdxl-turbo--h19565dc9259e--res-512x512/unet.engine (~962 s: 114.0 s ONNX
 export + 342.3 s ONNX optimize + 500.6 s TRT build) — a new hash, distinct from
 all four pre-existing UNet directories, confirming --fid2 forked the cache as
 intended. VAE encoder/decoder reused the existing, correctly-scoped (A3) cache
 directory with no rebuild. S4's zero-fill warning fired exactly once, during
 prepare(), before update_style_image() supplied a real style image — confirming
 it fires only on the genuine gap it was added to catch. TRT output is visually
 consistent with the PyTorch faceid.png (same sepia toning, same swept-back
 hairstyle shift).
 5. `venv/Scripts/python.exe -m pytest tests/unit/ -k faceid -q` (plus the
 engine-path and BGR-patch suites) — 41 tests, all passing.
 6. `venv/Scripts/python.exe scripts/sync_td_mirror.py --check` — clean.
 7. `bash scripts/git/check_lint.sh` — ruff (the script's "Lint" stage): PASSED,
 0 findings, after fixing 6 real diagnostics this session introduced across its
 own files (E402, B904, UP037, I001 x3, F841 — all now clean). Markdown: FAILED,
 but only on pre-existing findings unrelated to this work (README.md,
 src/streamdiffusion/_hf_tracing_patches.md, StreamDiffusion-installer/README.md,
 tests/quality/README.md, and this file's own long-standing formatting).

 ---
 Still deferred

 #10 FaceID-Plus (16 tokens) unreachable — TD never emits num_image_tokens and
 build_embedding_hook hard-raises on a token mismatch
 (ipadapter_module.py:205-261). Note
 faceid_v2_weight only ever applies on this branch, which is why A5 calls it inert
 today.