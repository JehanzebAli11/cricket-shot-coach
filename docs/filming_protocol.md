# Filming protocol (v0.1)

This protocol is how we film batting clips for the shot classifier and the
form checks. The aim is clips that are consistent enough to compare and
varied enough that the model doesn't learn one net, one light or one batter.

**Before any filming:** every batter has a signed consent form
(`consent_form_en.md` / `consent_form_ur.md`). For anyone under 18, a parent
or guardian has signed too. No form, no filming. Check this at the start of
every session, not afterwards.

## 1. Cameras

Two phones on two tripods, filming at the same time.

| | Side-on camera | Front-on camera |
|---|---|---|
| Position | Square of the batter on the **off side**, level with the popping crease | Behind the bowler's arm, straight down the pitch (or behind the net's bowling end) |
| Distance | Far enough to fit the batter head-to-toe plus the full backlift and follow-through, roughly 6–10 m | Same rule: full body plus bat in frame at all times |
| Height | Lens at about **hip height** (≈1 m) | Lens at about **chest height** (≈1.3–1.5 m) |
| What it's for | Front knee bend, head over front knee, stride, backlift, trunk lean | Head position, stance width, shoulder tilt, bat path (straight vs across) |

* Use tripods. Handheld shake ruins the angle measurements.
* Lock focus and exposure on the batter (long-press on most phones).
* Keep the batter roughly in the middle third of the frame, with no part of
  the body or bat cut off at any point in the shot.
* **Never film through the net mesh if you can avoid it.** If you must,
  put the lens right up against a gap.
* Mark the tripod spots with tape, so camera position is the same across
  sessions at a venue. Note the distances on the session sheet.

## 2. Settings

* **60 fps**, **landscape**, 1080p. 4K is not needed and fills phones quickly.
* Turn off beauty filters, HDR video and "action/stabilisation" crop modes.
* Leave the phone's clock set automatically. We sync the two cameras with a
  clap (below), not by clock.
* Before each set: one clear **hand clap** visible to both cameras, so the two
  views can be lined up later.

## 3. What to keep in each clip

Record continuously during a set, then cut the clips afterwards (the batch
script processes the cut clips):

* **Start 1–2 s before the ball is released** (the batter is in their stance).
* **End 1 s after the follow-through finishes.**
* One delivery per clip. Leave or abandon the clip if the batter isn't
  playing a real shot (e.g. walking off, fixing a pad), or label it `none`.

## 4. Variety we need

Aim for a spread on every one of these, and log them in the labels:

* **Handedness**: right- and left-handed batters. Lefties are rarer, so ask for them.
* **Light**: morning, midday, evening, floodlights, indoor halls.
* **Venue**: nets, open grass, indoor school hall, concrete/matting strips.
* **Clothing**: whites, coloured kits, school uniform with pads.
* **Level and age**: beginners through to strong club players (with guardian
  consent for under-18s).
* **Ball type**: hard ball, tape ball, tennis ball.

## 5. Held-out academy

Pick **one whole academy (or club)** at the start and do not use any of its
clips for training or tuning. It is the honest test of whether the model works
somewhere it has never seen. Record the choice in the dataset README
before you look at any results, and don't swap it later.

## 6. Labels (one row per clip)

Fill in `labels.csv` the same day, while you still remember. Columns:

| Column | Example | Notes |
|---|---|---|
| `clip_id` | `acA_2026-11-14_s02_d07_side` | Same as file name without extension |
| `pair_id` | `acA_2026-11-14_s02_d07` | Links the side-on and front-on clips of one delivery |
| `view` | `side` / `front` | |
| `academy` | `acA` | Code, not the full name, in shared files |
| `batter_id` | `B017` | Pseudonym from the consent register. **Never the real name.** |
| `handedness` | `R` / `L` | |
| `age_band` | `U13`, `U16`, `U19`, `adult` | Not exact age |
| `shot` | `cover_drive`, `straight_drive`, `pull`, `cut`, `sweep`, `defence`, `leave`, `other` | Agreed with the coach |
| `foot` | `front` / `back` | |
| `outcome` | `middled`, `edge`, `miss`, `bowled`, `unclear` | |
| `light` | `day`, `overcast`, `evening`, `floodlight`, `indoor` | |
| `venue_type` | `nets`, `outdoor`, `indoor` | |
| `ball` | `hard`, `tape`, `tennis` | |
| `public_ok` | `yes` / `no` | From the consent form, part B. Default `no`. |
| `blur_face` | `yes` / `no` | From the consent form |
| `labeller` | initials | |
| `notes` | free text | e.g. "keeper in frame", "net pole hides back foot" |

## 7. Storage on the day

* Copy clips off the phones the same day to the encrypted project drive, then
  delete them from the phones.
* Don't post clips in WhatsApp groups, even "just for the coach". Use the
  shared project folder.
* If a batter or guardian withdraws, delete their clips and keypoints and log
  the deletion (see the consent form).

## 8. Never

* No broadcast, streaming or social media footage of anyone, professional or not.
* No filming of anyone who has not consented, including people in the
  background. Reframe or pause.
* No filming in changing rooms or toilets, or anywhere near them.
