# Size Parser & Brand Alias Audit

## Store A
- Rows: 233,199
- Multipack indicator in `name`: 67,475
- Multipack indicator in size source: 57,982
- Leading-decimal size in size source: 382
- Trivial `count:1` buckets: 1,562
- BUG: multipack name with bucket missing pack multiplier: 31,358
- BUG: leading-decimal size but bucket empty: 0
- Empty bucket rows overall: 97,818

## Store B
- Rows: 55,516
- Multipack indicator in `name`: 2,449
- Multipack indicator in size source: 6,245
- Leading-decimal size in size source: 0
- Trivial `count:1` buckets: 39
- BUG: multipack name with bucket missing pack multiplier: 1,891
- BUG: leading-decimal size but bucket empty: 0
- Empty bucket rows overall: 8,212

## Brand alias near-matches
- Total candidate aliases (token-set-ratio >= 90): 83
- Score >= 95: 77

### Top 25 alias candidates
- B `e l f` (n=223) ~ A `e l f cosmetics` (n=151) score=100
- B `amy s` (n=75) ~ A `amy s kitchen` (n=20) score=100
- B `l oreal` (n=66) ~ A `l oreal paris` (n=26) score=100
- B `nature s path organic` (n=47) ~ A `nature s path` (n=13) score=100
- B `bigelow` (n=40) ~ A `bigelow tea` (n=39) score=100
- B `so delicious dairy free` (n=35) ~ A `so delicious` (n=12) score=100
- B `bush s best` (n=32) ~ A `bush s` (n=29) score=100
- B `stonyfield organic` (n=28) ~ A `stonyfield` (n=15) score=100
- B `toll house` (n=27) ~ A `nestl toll house` (n=11) score=100
- B `bell and evans` (n=26) ~ A `bell` (n=14) score=100
- B `lindt lindor` (n=25) ~ A `lindt` (n=106) score=100
- B `good to go` (n=24) ~ A `good` (n=17) score=100
- B `hero` (n=22) ~ A `hero cosmetics` (n=12) score=100
- B `diamond cosmetics` (n=21) ~ A `diamond` (n=13) score=100
- B `u by kotex` (n=21) ~ A `kotex` (n=48) score=100
- B `mars wrigley` (n=21) ~ A `mars` (n=50) score=100
- B `rachael ray nutrish` (n=20) ~ A `nutrish` (n=40) score=100
- B `clif` (n=19) ~ A `clif bar` (n=18) score=100
- B `coors` (n=19) ~ A `coors light` (n=14) score=100
- B `voortman bakery` (n=18) ~ A `voortman` (n=12) score=100
- B `mt olive` (n=18) ~ A `mt olive pickle` (n=15) score=100
- B `suave essentials` (n=15) ~ A `suave` (n=178) score=100
- B `new york shuk` (n=14) ~ A `york` (n=13) score=100
- B `feel good foods` (n=14) ~ A `good` (n=17) score=100
- B `c4` (n=13) ~ A `c4 energy` (n=19) score=100