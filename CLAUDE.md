# Lock In

## Project Overview

Lock In is an AI-powered productivity web application designed to detect when a user is distracted and intervene.

The system has three major components:

1. Computer vision attention detection
2. LLM-generated interventions
3. Device/browser activity tracking

The primary goal is to produce a working demo for the UCSD SPIS final project.

## Development Priority

Work in this order:

1. Computer vision system
2. LLM intervention system
3. Text-to-speech
4. Integrate the above systems
5. Browser/activity tracking
6. UI and polish

A working simple implementation is more important than an ambitious incomplete implementation.

## Computer Vision

The vision system uses the user's webcam.

Its purpose is NOT to determine whether someone is objectively "productive."

It should detect simple observable behaviors correlated with distraction, such as:

- looking significantly left or right
- looking down
- eyes closed for an extended period
- face disappearing from the camera

Prefer robust, understandable techniques over unnecessarily complicated ML models.

The system should avoid triggering from brief normal movements.

Use temporal smoothing / thresholds rather than making decisions from individual frames.

## Intervention System

When sustained distraction is detected:

vision detection
→ distraction event
→ LLM
→ generated reminder
→ text-to-speech

The LLM should generate short, humorous but useful reminders encouraging the user to return to their task.

Avoid excessive API calls.

Use cooldowns so repeated detections do not continuously trigger interventions.

## Architecture

Keep major systems modular.

Vision code, LLM code, audio/TTS code, browser tracking, and application/UI logic should not become one giant file.

Prefer simple interfaces between components.

## Development Philosophy

This is a one-week student project.

Optimize for:

1. working demo
2. reliability
3. understandability
4. development speed
5. sophistication

Do not introduce unnecessary infrastructure or abstractions.

Do not rewrite working components merely to make them theoretically cleaner.

## Working With Me

I am a CS student and want to understand the important architecture and algorithms.

When implementing something substantial:

- briefly explain the approach first
- explain important design decisions
- tell me which files you changed
- mention important libraries being introduced
- point out anything I should understand before presenting the project

Do not explain basic programming syntax unless I ask.

## Agent Behavior

Before making large architectural changes, inspect the existing project.

For complex features, create a plan before implementation.

When debugging:

1. reproduce or identify the failure
2. determine the likely root cause
3. make the smallest reasonable fix
4. run the relevant code/test again
5. continue until the issue is resolved or explain what blocks further progress

Do not claim something works unless it has actually been tested when testing is possible.

Never delete large amounts of working code without explaining why.

Do not commit, push, deploy, or modify external services without explicit permission.

## Documentation

Keep README.md reasonably current as the architecture stabilizes.

Prefer comments explaining WHY something exists instead of comments restating obvious code.