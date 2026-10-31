# Build log

Measured numbers and scope decisions, newest last. All times IST.

## 31 Oct

- 13:00. Picked YC "AI for City Hall", plan review. Probe on BME Building F (4 floors, old scan, truth by eye): SAM word and click prompts failed on line drawings. SAM video tracker with floors stacked as frames and box prompts matched 15 of 15 floor and staircase pairs and reported no mask on a fully erased staircase. Spark alone found the same staircases per sheet.
- 13:20. User approved the build and asked for deep research and many experiments. Round 1 spend cap: 60 USD of 148.
- 13:45. Two security reviews of the first pushes flagged 8 upload and access issues. All fixed with tests (26 passing): body limit enforced while receiving, page and pixel limits, two workers and eight waiting at most, access key required off this machine, local means the socket address and no forwarding headers, cross-site posts refused, and PDFs read in a child process that is killed after 15 minutes.
