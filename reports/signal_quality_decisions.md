# Signal Quality Investigation — Initial Findings

## Dataset
- Task: VerbalWorkingMemory
- Participants inspected: sub-001 and sub-002
- Sampling frequency: 500 Hz
- Channels: 19 EEG + 2 EOG

## Finding
In sub-001, all 21 channels reach approximately -10,239.69 µV. During the interval from 332.5 to 333.5 seconds, each channel contains many samples at or below -10,239 µV. The longest consecutive runs are approximately 0.48–0.52 seconds.

The same boundary was not observed in sub-002 during this interval.

## Interpretation
The repeated boundary across all channels is suspicious for clipping, saturation, or another shared digital/data-representation issue. The cause has not been confirmed.

## Decision
- Preserve the original EEG files.
- Do not manually replace extreme values.
- Do not exclude sub-001 solely on the basis of this initial check.
- Investigate the recording's data representation and source documentation before deciding participant eligibility.
- Keep signal-quality decisions separate from event-structure and response-label audits.

## Limitation
The comparison covers one one-second interval. It does not establish that sub-002 is artifact-free or that the remainder of sub-001 is invalid.
