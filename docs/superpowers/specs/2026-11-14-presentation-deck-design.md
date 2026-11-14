# Presentation deck: Market IDS (A07), CCTV control-room style

Date: 2026-11-14. Owner: "แบบแรกสวยดีแต่อยากให้ตัวโลโก้ใช้สีจริงๆ ... ถ้ามั่นใจแล้วเริ่มทำมาได้เลย"

## Intent

A 10-minute talk to the instructors, most likely the progress presentation on 19 Nov 2026 (the Notion calendar has 19 Nov progress and 14 Dec final). It should score A on all four rubric rows:
- topic and concept;
- progress, obstacles and fixes;
- teamwork;
- plan.

It must look modern and cool, and must not look AI-made. The deck lives in the owner's Canva account and every element can be edited there.

## Approaches considered

1. **Canva AI generation from an outline (`create-design`).** Fast, but Canva picks its own look, which loses the approved style. Rejected.
2. **Build a .pptx with real text boxes, shapes and logo images, then upload it to Canva.** Canva converts it into an editable design. This keeps the approved look and every piece stays editable. **Chosen.**
3. **HTML import.** It needs a public URL for a generated file, which is not allowed. Rejected.

## Design

- **Style 1, CCTV control room**, as approved in `scratchpad/hud-mock/deck-styles.html`:
  - background `#05070d`, neon cyan `#00e5ff`, red `#ff2d55`, text `#e6f1ff`, muted `#8aa0b8`;
  - corner brackets, mono tags such as `CAM-0x · SECTION`, a REC dot;
  - no gradient fills.
- **Logos in their real brand colours** (the owner's ask). They sit on white rounded tiles so dark logos (GitHub, Notion, Render) stay visible on the dark slides. Sources: devicon "original" colour SVGs where they exist, otherwise simple-icons in the brand hex, rendered to PNG.
- **Fonts:** Bai Jamjuree for headings, IBM Plex Sans Thai for body, JetBrains Mono for tags. All three are in Canva, with Kanit for the big numbers.
- **13 slides, 16:9**, each with Thai speaker notes and a time cue (about 45 s each):
  1. Cover
  2. Hook: real pump numbers (median of 200 real pumps: +15 % price, ×739 volume)
  3. Problem, target users and scope
  4. What is new compared with existing work
  5. How it works (pipeline)
  6. Tools, with logos
  7. Live demo (site screenshot and URL)
  8. Results so far: 74 % of real pumps flagged at 2 alerts per coin-day; the targets met and not met are stated honestly
  9. Progress: tasks done and milestones
  10. Problems and how we fixed them
  11. Teamwork: roles and tools
  12. Plan to 14 Dec: work breakdown and milestones
  13. Thanks and Q&A
- **Facts:** only facts from the repo (README, `docs/REPORT.md`, git) and the Notion export. Words like "ผิดปกติทางสถิติ" stay as they are. A p-value is never called a probability, and there is no accuracy metric.

## Left to the builder

- Exact layout per slide.
- Icon art for the pipeline boxes.
- How each role maps to a member name. Notion shows the handles "Fuscosucof" and "Kum Ch", and the owner is asked which members they are.

## Non-goals

- No Canva AI restyle.
- No made-up numbers.
- Nothing that accuses a person or a coin group.
