# Match Signal Probe

## Core coverage

Store A:
- Rows: 233,199
- UPC-like token rows: 1,122 (0.48%)
- Brand coverage: 54.13%
- Parsed size coverage: 58.05%
- Inferred private label: 7,193 (3.08%)

Store B:
- Rows: 55,516
- UPC-like token rows: 366 (0.66%)
- Brand coverage: 90.00%
- Parsed size coverage: 85.21%
- Inferred private label: 8,064 (14.53%)

## Cross-store overlap

- `brand_norm` shared unique values: 2,438 (A 18,563, B 5,635)
- `size_bucket` shared unique values: 716 (A 1,293, B 769)
- `info_category_0_norm` shared unique values: 0 (A 43, B 11)
- `info_category_1_norm` shared unique values: 4 (A 890, B 105)
- `info_category_2_norm` shared unique values: 47 (A 4,118, B 541)
- `info_category_3_norm` shared unique values: 62 (A 6,081, B 610)

## Blocking estimates

- `brand_norm`: 2,438 shared blocks, A coverage 14.8%, B coverage 51.7%, estimated candidate pairs 1,070,709
- `brand_norm`, `size_bucket`: 3,750 shared blocks, A coverage 4.2%, B coverage 23.5%, estimated candidate pairs 57,938
- `brand_norm`, `size_bucket`, `info_category_2_norm`: 288 shared blocks, A coverage 0.3%, B coverage 1.7%, estimated candidate pairs 3,611
- `size_bucket`, `info_category_2_norm`: 614 shared blocks, A coverage 2.3%, B coverage 7.9%, estimated candidate pairs 93,834
- `is_private_label_inferred`, `size_bucket`, `info_category_2_norm`: 604 shared blocks, A coverage 2.2%, B coverage 7.1%, estimated candidate pairs 77,064

## Top shared brands

- hallmark: A 1,028, B 2
- disney: A 776, B 6
- goya: A 228, B 376
- hot wheels: A 575, B 4
- maybelline: A 129, B 386
- l oreal paris: A 26, B 423
- hershey s: A 207, B 218
- revlon: A 105, B 301
- wilton: A 201, B 194
- dove: A 206, B 184
- bluey: A 363, B 3
- tide: A 285, B 52
- gerber: A 244, B 88
- spring valley: A 298, B 10
- mccormick: A 60, B 237
- reese s: A 137, B 141
- marvel: A 269, B 2
- covergirl: A 107, B 153
- lindt: A 106, B 140
- barbie: A 243, B 1
- suave: A 178, B 59
- pillsbury: A 122, B 108
- neutrogena: A 120, B 106
- minecraft: A 225, B 1
- command: A 185, B 27