# Build log

Measured numbers and scope decisions, newest last. All times IST.

## 31 Oct

- 13:00. Picked YC "AI for City Hall", plan review. Probe on BME Building F (4 floors, old scan, truth by eye): SAM word and click prompts failed on line drawings. SAM video tracker with floors stacked as frames and box prompts matched 15 of 15 floor and staircase pairs and reported no mask on a fully erased staircase. Spark alone found the same staircases per sheet.
- 13:20. User approved the build and asked for deep research and many experiments. Round 1 spend cap: 60 USD of 148.
- 13:45. Three security reviews of the first pushes flagged 10 upload and access issues. All fixed with tests (24 passing): body limit enforced while receiving, page, pixel and disk limits, two workers and eight waiting at most, the API always needs a key (a random one per run if none is set), cross-site posts refused, and PDFs read in a child process that is killed after 15 minutes and reports through a file.
