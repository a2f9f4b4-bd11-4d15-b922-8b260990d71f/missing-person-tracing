# Missing Person Tracing — Product Spec

Oct 8, 2026 · @Thai

## Overview

Given one photo of a missing person, the system finds where and when that person appeared across several cameras and shows it as a timeline. It is a personal AI project (topic #15 from the Mini projects list), built on a Raspberry Pi 4 camera node and a desktop processing server.

**Problem.** When someone goes missing, the first hours matter, but reviewing CCTV is manual: people scrub through hours of video from each camera, one at a time. Faces in CCTV are often small, blurry or turned away, so face recognition alone misses many sightings.

**Solution.** Match people by whole-body appearance (clothing, build, colours) across cameras, with face matching as an extra signal when a face is visible. Return a ranked, time-ordered list of sightings so a reviewer checks minutes of clips instead of hours of footage.

## Goals and non-goals

**Goals**

- Find a person across at least 2 cameras from a single reference photo.
- Show sightings as a timeline per camera, with clips a reviewer can open.
- Work on recorded footage (MVP) and on live streams (final).
- Run on hardware already owned: a Raspberry Pi 4 as the camera node, a desktop as the processing server.
- Report honest accuracy numbers on a public benchmark and on own footage.

**Non-goals**

- Identifying unknown people, or building a database of everyone who passes a camera.
- City-scale deployment, many cameras, or integration with real CCTV networks.
- Mobile app, user accounts, multi-tenant access.
- Running the AI models on the Pi itself (too slow on a Pi 4 without an accelerator).
- Production-grade security hardening.

## Users and use cases

The primary user is a **reviewer**: someone responsible for a site's cameras (building security, a school, a mall operator) who is asked to help find a missing person.

| Use case | Input | Output |
| --- | --- | --- |
| Find a missing person in recorded footage | Reference photo + time window | Ranked sightings with clips, per camera |
| Reconstruct a route | Reference photo | Timeline of sightings in time order, across cameras |
| Watch live for a person | Reference photo, search left running | Alert when a new match appears (final, stretch) |
| Search by description | Text such as "red shirt, backpack" | Ranked sightings (final, stretch) |

For this project, the demo operator is the builder, and the people filmed are volunteers who agreed to take part.

## Requirements

The MVP proves the search works on recorded clips; the final release adds live cameras and the timeline.

| ID | Requirement | Release |
| --- | --- | --- |
| R1 | Import recorded video files and tag each with a camera name and start time | MVP |
| R2 | Detect and track every person in each video; save each continuous sighting with a thumbnail | MVP |
| R3 | Upload a reference photo and get the top 10 matching sightings, ranked by similarity | MVP |
| R4 | Each result shows camera, time, similarity score, thumbnail and a link to the clip | MVP |
| R5 | Filter results by time window and camera | MVP |
| R6 | Drop matches that are impossible in time (two cameras, too close together) | MVP |
| R7 | Ingest live video from the Pi 4 webcam and a phone camera at the same time | Final |
| R8 | New sightings become searchable within 10 s of the person leaving the frame | Final |
| R9 | Timeline view: one lane per camera, sightings in time order, click to play | Final |
| R10 | Reviewer can confirm or reject a match; rejected matches leave the timeline | Final |
| R11 | Footage and sightings auto-delete after a set number of days | Final |
| R12 | Face matching as an extra score when a face is visible | Final, stretch |
| R13 | Search by text description instead of a photo | Final, stretch |

Only one of R12 and R13 is planned; pick it on 17/10 based on time left.

## User flow

A search takes the reviewer from one photo to a checked route in five steps.

1. **Upload a reference photo** of the person, full body if possible. The system shows the detected person so the reviewer can confirm it picked the right one.
2. **Set the scope**: time window (for example, the last 6 hours) and which cameras.
3. **Review ranked matches**: thumbnails with camera, time and score, best first.
4. **Confirm or reject** matches. Confirmed sightings form the route.
5. **Read the timeline**: one lane per camera, confirmed sightings in time order, last known location highlighted.

In live mode (final), steps 3–5 keep updating as new sightings arrive.

## Success metrics

The project succeeds if the right person is the top match in most searches on own footage, and the live demo shows a route across two cameras within seconds.

| Metric | Target | Measured on |
| --- | --- | --- |
| Top-1 accuracy (right person is the first result) | ≥ 80% | Own test clips, 3–4 people |
| Top-5 accuracy | ≥ 95% | Own test clips |
| Rank-1 accuracy, public benchmark | Report the number; no target | Market-1501 |
| Time from person leaving frame to searchable | ≤ 10 s | Live demo (final) |
| Search response time | ≤ 2 s | 1 hour of footage per camera |
| Reviewer time to build a route | Minutes, not hours | Demo walkthrough |

The targets for own footage are a first guess; adjust them after the first measurement on 11/10.

**Acceptance**

- **MVP (11/10):** on recorded clips from 2 spots, a photo of person A returns A's sightings from both spots at the top, in correct time order.
- **Final (18/10):** a volunteer walks past the Pi camera and then the phone camera; both sightings appear on the timeline within 10 s, and the search ranks them first.

## Privacy, ethics and consent

The system only searches for a person someone has a photo of; it never names or profiles anyone else it sees.

- **Consent:** only film volunteers who agreed, and put cameras where no one else is recorded (a room, a corridor at home). Public datasets (Market-1501) are used under their research terms.
- **Search starts from a photo:** no browsing "everyone seen today", no identity labels on non-matches.
- **Retention:** footage and sightings auto-delete after a set number of days (R11). Reference photos are deleted when a search is closed.
- **Human in the loop:** results are suggestions with scores; a person confirms each match (R10).
- **Local only:** all processing on own machines; no cloud APIs receive video or photos.
- **Known bias:** re-ID accuracy drops with lighting, camera angle and clothing changes. The report states this with the measured numbers.

## Milestones

Two releases: the MVP on Sunday 11 October and the final on Sunday 18 October 2026. The day-by-day plan is in the technical spec.

| Milestone | Date | Scope | Done when |
| --- | --- | --- | --- |
| MVP | Oct 11, 2026 | R1–R6: search on recorded clips, minimal web UI | MVP acceptance passes; demo video recorded |
| Final | Oct 18, 2026 | R7–R11 + one of R12/R13: live cameras, timeline | Final acceptance passes; report with evaluation table |

The webcam must arrive by 12/10 for the live work; if it is late, two phones act as cameras and the Pi relays their streams.

## Risks and open questions

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Accuracy drops with clothing, light or angle changes | Wrong person ranked first | Record test clips that include these cases; report numbers honestly; human confirms matches |
| Webcam arrives late | Live demo slips | Phones as cameras, Pi relays streams |
| Schedule overrun (evenings only on weekdays) | Final scope cut | Drop the stretch feature first, then own-footage evaluation; keep live demo and timeline |
| Too few volunteers | Weak evaluation | At least 3 people; vary clothes across clips to simulate more |

**Open questions**

- [ ] Who are the 3–4 volunteers, and when can they be filmed?
- [ ] Which stretch feature for the final: face matching (R12) or text search (R13)?
- [ ] How many days should footage be kept (R11)?
