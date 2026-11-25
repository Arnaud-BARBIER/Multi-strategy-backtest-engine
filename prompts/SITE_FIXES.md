# FORGE demo recovery and final captures

Work in `/Users/arnaudbarbier/Quant reaserch` and speak French with Arnaud.
Complete the tasks below, using focused reads and checks to limit credit usage.

## Goal

Restore a working copy of the website demo strategy, remove the duplicate stacked
allocation chart from the joined editor/terminal view, add a real demonstration of
trial accounting and change history, and loop custom charts and independent editor /
terminal theme changes on the conclusion screen.

## Read first, narrowly

- This prompt.
- `forge_demo_recovery/2026-11-25/README.txt`.
- Applicable `AGENTS.md` before working in the engine repository.
- Relevant excerpts of `Multi-strategy-backtest-engine/prompts/SITE_FORGE_DIRECTOR.md`
  and `SITE_K5_PROMPT.md`, especially the terminal layout requirement. Their old status,
  scratch paths, server state and browser instructions may be stale; verify them.

## Current website

Repository: `Multi-strategy-backtest-engine`, branch `main`, GitHub Pages from `docs/`.
French source: `docs/next/index.html`; English source: `docs/next/en/index.html`.
Generate both home pages with `python3 tools/site/make_home.py` after source edits.
Do not edit generated home pages independently.

Already implemented: new hero copy, native four-stage concept diagram, updated page 3
text, looping page 3 customisation, proposed agent architecture slide immediately before
contact, parked editor/terminal behind that slide and tree-to-code return on contact.
The concept diagram has 2px outlines and a blue halo on its active stage.
Do not revert these changes or redo the opening slides.

Current French hero lead (one paragraph only):
“Planifiez et composez vos futurs investissements, comparez vos options de placement
et testez vos propres mécanismes de protection du capital et de sécurisation des gains,
sans que la complexité technique ne limite vos ambitions.”

Preserve the Umami script and website ID, Open Graph metadata, FR/EN links and indexability.
Do not add noindex to either home page.

## Recovered evidence: do not search for it again

Persistent recovery directory:
`/Users/arnaudbarbier/Quant reaserch/forge_demo_recovery/2026-11-25/`

- `recovered_state/`: 554 files copied from a surviving scratch state and verified
  against their source with `sha256_manifest.json`. Includes `Mon portefeuille`, its
  saved v1 Git repository, private draft, saved run, report layout and trial ledgers.
  IMPORTANT: this is the PAGE 3 customisation variant, not the final main demo:
  six assets, lookback 60, max_weight .35, shrinkage .15, no trend filter. Its stored
  draft reports annual return ~16.35%, volatility ~18.72%, Sharpe ~.903, drawdown ~-30.99%.
- `main_demo_code_from_capture/`: imports.py, features.py, allocator.py, view.py,
  specs.py and combined_source.py extracted from `docs/next/states/ed3_box.html`.
  All four functional boxes match the AST of the original final AI answer.
- `main_demo_code_from_capture/settings_from_ai_answer2.toml`: original AI settings,
  not a recovered runtime snapshot. lookback 126, max_weight .20, shrinkage .20,
  trend_filter 1, SMA fast 50, SMA slow 200. Verify against the actual frozen states.
- `original_ai_answers/answer1.md` and `answer2.md`: original two AI iterations.
- `t*.json`: preparation scripts for page 3, especially t17.json. Inspect and adapt;
  their temporary paths are stale. Do not execute them blindly.

The original a718eba9 scratchpad is gone. The original main demo v1/v2/v3 history has
NOT been recovered. Do not invent old commits, dates, trial counts or transitions.

## 1. Restore in an isolated, persistent working state

Use a NEW copy under `forge_demo_recovery/`, not /tmp. Keep recovered_state untouched.
Do not replace the user's main `Github/Backtest_Git/showcase_app/_state`.
Engine checkout: `Github/Backtest_Git`; Python: workspace `.venv/bin/python`.
Follow its applicable instructions. Do not trigger generated-kernel compilation without
Arnaud's authorization. Do not install or upgrade packages.

Inspect data paths and code compatibility before starting the application. Use the
captured main-demo code and original settings to restore the final main-demo strategy.
Preserve the separate page 3 variant for custom-chart references.

Main-demo study-period reference metrics: annual return 17.6%, volatility 16.1%,
Sharpe 1.09, maximum drawdown -26.7%. Compare assets, dates, fees, clocks and settings
before claiming numerical parity. Document any mismatch; never adjust displayed numbers
just to match the website. No backtest or compilation has been run during recovery.

## 2. Fix the terminal beside the editor

Affected frozen terminal: `docs/next/states/term23.html`.
Joined scene: editor on the left third, terminal on the right two thirds.
Its original capture geometry is 1333 x 1250 before the existing .8 display scale.

Remove the duplicate bottom stacked weights/rebalance chart. The previously agreed
bottom Allocation section contains:
- Impact des frais d'allocation.
- Positions investies par actif, devise principale.
- Registre des flux de trésorerie (D).

Inspect the recovered report.toml and the relevant saved layout. Configure the isolated
copy and capture actual charts, fully drawn. Keep the main demo's strategy and result.
Do not refreeze the entire early walkthrough unnecessarily.

## 3. Add a final real trial/history step

Place it after the code demonstration and before the proposed agent architecture slide.
Suggested sequence: focus the trial counter near DSR, make a deliberate new setting
choice, show the actual recorded effect, open stored settings/runs, then the version
history and a genuine code difference. Keep it short and legible.

Verified current application mechanisms:
- Summary contains `dsr`, `universe-edits`, `trials` in that default order. The Trials
  tile displays D.n_trials; DSR also explains the count. Interface preferences can hide tiles.
- `Réglages sauvegardés` contains explicitly saved settings, not every attempted setting.
- Stored runs can be compared; `/api/strategy/runs` groups them by version with trial totals.
- Code history is separate from parameter settings and run history.
- The trial counter is not automatically the raw number of button presses. Inspect its
  accounting/deduplication rules before designing the increment shown in the recording.

Use fresh, real history when needed. Do not present reconstructed history as the lost old
history, or substitute the page 3 trial totals for those of the main strategy.

## 4. Loop personalisation on the conclusion screen

Use the corrected joined editor/terminal, not the terminal with duplicate stacked charts.
Recall page 3 by introducing the same custom views (risk contribution vs weights,
efficient frontier, relevant execution/indicator panes), then changing theme colours.
Change BOTH editor and terminal themes, visibly and independently.

Keep readable pauses and a clean return to the original layout before repeating.
Preserve the parked-scene transition during the agent slide and its return on contact.
Stop all timers when leaving the step. Respect reduced motion by displaying a useful
static state. Use real captured states; do not manufacture results.

## Verification and delivery

- Keep FR and EN equivalent; regenerate both home pages.
- Check at approximately 1288 x 723 CSS pixels and mobile 375–393px.
- Confirm full diagrams can scroll on mobile without accidentally skipping steps.
- Check forward/backward/rapid navigation, departure to free scroll, and reduced motion.
- Preserve Umami, og:image, local paths, languages and absence of noindex on home pages.
- Run targeted checks; no engine-wide tests for website-only changes.
- Never stage unrelated existing changes in history/*.py, docs/CAPTURES_RESEARCH_ALGO.md,
  or other prompt files. Check git status before edits and commits.
- Do not push the new capture work unless Arnaud requests it in the resumed session.
- Report what was actually restored, metrics verified, captures changed, and limitations.

## Recovery execution — 25 November 2026

- Restored `forge_demo_recovery/2026-11-25/main_demo_work/state` with separate copied data (36 hashed files); all 554 recovered files remain byte-identical. Main user state untouched.
- Captured source AST restored; original seven assets and study dates 2017-02-01 → 2023-08-16 retained. Terminal controls supply geometric capital, USD 100000, 5 bps execution and 10 bps FX; strategy clock is 21 bars.
- Verified without JIT: annual return 17.6030388%, volatility 16.1122511%, Sharpe 1.08740352, drawdown -26.6643950%, capital 285391.4361, fees 2102.6435, 71 rebalances. Matches published rounding; compilation 0 seconds.
- New native v1–v4 history, never reconstructed old history. Recorded 20% → 24% cap choice: 11 → 12 trials and annualised Sharpe 1.0874 → 0.9381. Earlier identical-request check stayed 4 → 4. Final verified ledger: 13 trials.
- Genuine v3 code difference changes visual bands from 2 to 2.5 standard deviations; v4 restores captured code. Original settings/code and corrected report are saved in isolated state.
- Replaced only existing `term23`; added nine frozen states for trials/settings/stored runs/code difference, custom views and independent themes. Allocation contains cost impact, invested positions and cash ledger, fully drawn.
- Custom risk/frontier charts run on the same main-demo result; frontier explicitly uses the complete universe. Execution view includes native price indicators and individual-Sharpe pane.
- Added FR/EN history step before agents and contact personalisation cycle; preserved parked agent scene, tree return, original hero/concept, reduced-motion static state and timer cleanup.
- Regenerated both home pages. Metadata, Umami script/ID, language links and indexability remain unchanged. Targeted JS syntax and ten-state integrity checks pass.
- Browser checks: 1288×723 desktop, 393×852 mobile; rapid/backward navigation, free-scroll departure, timer cancellation and reduced motion checked. Separate 375×723 diagram-end checks recorded in `main_demo_work/qa_scroll.log`.
- Review screenshots: `main_demo_work/qa_joined_final.png`, `qa_contact_independent_final.png`; logs and exact responses remain beside them. CUA browser host unavailable; used the prompt's supplied local capture driver.
- Native history UI currently labels per-bar Sharpe (~0.07); terminal Sharpe is annualised (1.09). Fresh DSR/trial totals intentionally differ from lost historical totals. No engine source edit, package install, compilation, commit or push. Levers: not concerned.

**For the director**
- Local implementation and captures complete; review the screenshots and native Sharpe-label limitation before publication.
- Recovery scripts, immutable evidence, run receipts and QA logs are persistent in `forge_demo_recovery/2026-11-25/`.

## Arnaud review revision — 25 November, 14:43 screenshot

- Page 2 now describes investment strategies, AI-generated Python versus directly written Python, and automatic execution; removed “simplify”. Millisecond wording explicitly refers to the engine calculation in this demonstration, not total interface/startup time.
- Removed the standalone recovery trials/history gallery and its six unused site states. Trial highlight now uses original a2/a2t before Compare; its real counter is 2649 (a1t: 2645). The caption explains dataset-wide cumulative accounting, including earlier searches.
- Found three genuine version nodes already preserved in original ed_tree: v1/v2/v3, 21 November. New ed_history reveals that original native panel after Tree. No lost Git repository/commit was reconstructed; no fresh recovery versions substituted.
- Agent background now uses the exact schema camera/dimming of pages 2/3. Compact layout plus desktop fitting keeps the whole diagram visible; mobile retains native scrolling.
- Editor join measured at 1173 ms, maximised code at 1819 ms, settled at 1921 ms. Contact personalisation phases shortened to 2100 ms and now replay Full screen → Escape → Back to tree → Open allocator with attached action cues.
- Fixed mobile camera preservation after resize for focused steps; original version labels land at x44, y132/201/271 on 375×723. Commit caption no longer covers the version list on desktop.
- FR/EN regenerated. Four-page metadata/scripts/language/indexability/JS checks pass; five active frozen-state integrity checks pass; mobile375 and desktop1288 QA pass, including rapid navigation and reduced motion.
- Agent camera matches schema exactly; desktop diagram bounds y68–669 on 723px viewport. Mobile scroll reaches the explanatory note without changing step or horizontal overflow. Conclusion timers cancel on departure.
- Review artifacts: forge_demo_recovery/2026-11-25/revision_1443/{agents,commits_final,commits_mobile_final,trials_final,contact_fullscreen}.png; qa*.log. No engine changes/runs, compilation, commit or push during this revision.

**For the director**
- Latest review replaces the earlier gallery/history placement; preview remains available on 127.0.0.1:8790.

## Deferred to a separate session — cross-asset timing refresh

- Arnaud will optimise his machine before the next measurement. Do not rerun cross-asset or compile during the current visual-fix session.
- The current recording shows about 149 seconds total (reported by Arnaud). After the latest engine changes and machine optimisation, his target/expectation is 25–27 seconds; this is unverified.
- In the next authorised session, inspect the original cross-asset protocol and preserve its actual basket/sample, settings, costs and study period. Measure actual end-to-end time, noting warm/cold state and machine configuration.
- Recapture the genuine run and replace the cross-asset video/state only after measurement. Never replace the displayed number with the expected target or infer a runtime from playback speed.

## History interaction correction — same-document opening

- Removed the derived ed_history state. Tree and Versions steps now share the original ed_tree iframe, without navigation/replacement.
- Restored the original History button toggle in that document: cursor travels to it, presses it, then its click handler opens the existing native history panel and updates aria-expanded/active styling. Paste and tree remain in the same native flow.
- Verified before click: same document, closed panel, visible cursor, zero click events. After: identical document and src, exactly one click, expanded panel and all three original version nodes.
- Leaving/rapid navigation closes the panel and cancels pending clicks. Reduced motion opens it immediately with no timers. FR/EN regenerated; metadata/JS/integrity checks pass.
- Proof: revision_1443/qa_history_click.log and history_{cursor_before_click,native_open}.png. No engine run, compilation or push.

## Editor sequence refinement — one downward action per interaction

- Replaced intro/tree/versions/code ordering with four explicit phases: aim at History; open History; close History and aim at Allocator/show its explanation; click Allocator and continue the existing box-flight/join/fullscreen animation.
- Each phase waits for the next scroll/arrow; timers only animate the current action. History does not open or close itself while the visitor reads. No explanatory bubble is displayed over the open history panel.
- The same ed_tree document persists; native History button receives one click to open, then one to close. The allocator target is remeasured only after the panel closes/reflows.
- Direct code navigation, leaving the editor sequence, rapid navigation and free-scroll cancel pending actions and close History. Reduced motion performs each requested state immediately; mobile focus retained.
- Actual ArrowDown QA verified all four phases, stable waiting states, two native toggle events, no simultaneous history/explanation, original code animation and zero lingering timers. Desktop1288/mobile375 and English reduced-motion pass; see revision_1443/qa_sequence.log and sequence_*.png.
- FR/EN homes regenerated; metadata/scripts/language/indexability/JS and frozen-state integrity checks pass. No engine run, compilation, commit or push.

## Agent architecture context and framing

- Added a separate introduction step immediately before the diagram in FR/EN: “Votre IA propose. FORGE exécute et garde la trace.” It explains the transition from the demonstrated AI workflow to a proposed API-driven agent research loop, using the existing editor-intro layout.
- The visitor advances explicitly from the introduction to the diagram. Outgoing content becomes hidden immediately, preventing title/diagram overlap during forward/back/rapid navigation. The same receded editor/terminal camera is preserved.
- Agent cards now follow the concept diagram’s frame hierarchy: visible 2px outer strokes and 1px internal dividers, square corners, existing colours preserved. Fitting compensates border width so reduced desktop diagrams retain the same visible strokes.
- QA: desktop1288x723 outer2.000px/divider1.000px, diagram bottom666px; mobile375 border2px and no horizontal overflow. Introduction fits mobile; full diagram remains scrollable. FR/EN, reduced motion, explicit arrow advance and rapid exit checked.
- Artifacts: revision_1443/agent_intro.png, agent_intro_mobile.png, agent_diagram_framing.png; qa_agent_context.log and qa_agent_strokes.log. Both homes regenerated; metadata/JS/integrity checks pass. No engine run, compilation or push.

## Unified continuation prompt

- All walkthrough steps now use the same original outlined “Appuyez sur ↓ pour continuer” component. Removed mobile “Suivant”, the separate hero hint and the oversized/rewritten hero notice. Hero uses the same button above its ticker; mobile uses the standard bottom position.
- FR/EN regenerated. Desktop hero/concept/editor and mobile agent intro verified to share the identical button DOM, 14px type, 8px/14px padding and 1px border. Full wording fits375px; English uses “Press ↓ to continue”.
- QA: revision_1443/qa_prompt.log and prompt_{hero,concept,mobile}.png. Metadata, JS and integrity checks pass; no engine work or push.

## Development outlook copy, settled fullscreen and diagram entrance

- Replaced agent-introduction copy verbatim with Arnaud’s “PERSPECTIVES DE DÉVELOPPEMENT / Une infrastructure pour la recherche financière agentique.” text; supplied English equivalent. “Règles d’exécution documentées” is an actual hyperlink back to demo page2 (concept), verified by mouse click.
- Main code animation now waits1100ms after joining the panes, then950ms for cursor travel before fullscreen click. Observed join992ms, click3046ms, maximise3229ms: click2054ms after join. Preserve this settle/travel separation; the previous460ms gap was too early.
- Guarded all pending code-animation callbacks and delayed fly cleanup with entry identity; leaving during the pause produces no late fullscreen click. Existing contact-loop timing remains unchanged.
- Added a one-shot staggered card/link/footer entrance lasting under0.9s to the agent diagram. It restarts on entry, cancels on exit and remains fully visible/static under reduced motion; no continuous animation.
- QA: actual fullscreen timings, link→page2, fifteen running entrance animations then five fully visible cards, zero animations on exit/reduced motion, zero late fullscreen clicks. Desktop1288/mobile375 copy readable with no horizontal overflow.
- Proof: revision_1443/qa_outlook_timing.log; outlook_text.png, outlook_mobile.png and agent_reveal_{early,complete}.png. FR/EN regenerated; metadata/JS/integrity checks pass. No engine run, compilation or push.


## Opening cue, continuous code playback and measured agent arrows

- Restored the first-entry prominent opening cue, retaining the original canonical wording/component. The normal cue is still visible before the Allocator click and returns only after code playback completes; it stays hidden during the whole opening/join/fullscreen sequence.
- Rebuilt code playback as guarded sequential phases. The flying code crop uses the actual outer editor box; all click rectangles/cursor/toast marks are cleared before switching views. The fullscreen cursor targets the live visible native button rather than stale capture coordinates. A1100ms join-settle pause plus950ms cursor travel precedes the click.
- Retained the outgoing editor underneath the incoming panes during the join to remove the blank frame; transparent joining container stacks above that temporary editor. Handoff is cleared after the join, on navigation and when leaving the demo. Fullscreen no longer leaves a floating “Plein écran” label over the code.
- Agent arrows are SVG paths measured from actual node borders at each layout/resize, with explicit arrowheads and separate directions/colours. Disabled the old pseudo-element connectors. Desktop, tablet and mobile routes follow their respective card layouts. Entrance is opacity-only so arrows remain attached during the reveal.
- QA: fullscreen projected cursor matched the native button geometry; no continue cue during clicks; final ide_max has continue=true and toast/press/fly/handoff=false. Rapid exit and English reduced motion pass. Agent desktop/tablet/mobile paths and no horizontal overflow inspected. FR/EN sources and generated homes pass integrity/metadata/JS checks and diff whitespace checks.
- Evidence: revision_1443/qa_playback_links.log, qa_code_final.log, code_sequence_result.json, code_sequence_fixed.mp4, opening_notice_large.png and agent_links_{measured,wide,tablet,mobile}.png. No engine run, compilation, package installation, commit or push.
