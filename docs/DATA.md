# Data

The datasets, the conditions written on them, where the correct answers come from, the licenses, and what the
call logs contain.

## The datasets

| | WDC Products | Abt-Buy | CUAD | IBM ArgQ-30k | arXiv | IBM ArgKP |
|---|---|---|---|---|---|---|
| domain | product offers | product offers | contract clauses | debate arguments | paper metadata | debate arguments |
| records kept | 2,841 offers of 500 products | 2,173 records from two shops | 4,383 clauses from 341 contracts | 11,904 arguments from 39 topics | 1,530 papers from 17 categories | 1,740 arguments from 17 topics |
| fields shown | title, description, brand, price | title, description, price | agreement, clause | topic, argument | title, abstract, submission date | topic, argument |
| matches known | offers of one product | labeled pairs | none | none | none | arguments of one key point |
| license | not stated | not stated | CC BY 4.0 | CC BY-SA 3.0 | CC0 1.0 (metadata) | CDLA-Sharing-1.0 |
| cite | Peeters, Der and Bizer, EDBT 2024 | Köpcke, Thor and Rahm, PVLDB 2010; Mudgal et al., SIGMOD 2018 | Hendrycks et al., NeurIPS 2021 | Gretz et al., AAAI 2020 | arXiv API | Bar-Haim et al., ACL 2020 |

- **WDC Products:** the variant in which 80% of the products have at least four textually similar products, so that
  offers of different products are hard to tell apart.
- **ArgQ-30k:** only arguments with a unanimous stance vote.
- **ArgKP:** only arguments that annotators matched to exactly one key point; arguments with the same key point
  count as the same argument. 1,174 of its 1,740 arguments also appear in ArgQ-30k.

Each dataset's conversion is recorded in `data/<dataset>/PROVENANCE.json`: the download it read and its SHA-256.

## The conditions

One condition pair per dataset (`data/<pair>/dataset.toml`, which also holds each negation, both orders of the
conjunction and the map's instruction):

| Pair | p | q |
|---|---|---|
| WDC-A | The product is a data storage or memory product (for example a hard drive, an SSD, a RAM module, a USB flash drive or a memory card). | The listed price is above 100, in whatever currency is shown. |
| Abt-A | The product is an audio device (for example speakers, headphones, a receiver, an amplifier or a turntable). | The product is made by Sony. |
| CUAD-A | The clause restricts a party from competing, from dealing with anyone other than the counterparty (exclusivity), or from soliciting or hiring the counterparty's customers or employees. | The agreement's title contains one of the words distribution, distributor, distributorship, reseller or supply. |
| ArgQ-A | The argument supports the position stated in the topic. | The topic proposes to ban, prohibit or abolish something. |
| arXiv-A | The paper's main field is computer science (for example machine learning, computer vision, language processing or computer security). | The paper's title contains a colon. |
| ArgKP-A | The argument supports the position stated in the topic. | The argument is more than 20 words long. |

The rewordings of each condition are in `harness/configs/ext-placebo-<pair>.toml` (ArgKP-A's in
`grid/configs/`). The repository also holds the definitions of a second pair per dataset (B), which the code can
run; the paper reports the A pairs only, and the B pairs' runs are not part of this repository.

## The truth

Where a dataset decides the correct answer, both queries of an identity are scored for accuracy.

- **A gold label** that comes with the dataset: CUAD's clause categories, assigned by law students and reviewed by
  attorneys (p is true exactly for Non-Compete, Exclusivity, No-Solicit Of Customers or No-Solicit Of Employees);
  ArgQ-30k's and ArgKP's stance (p is true for arguments marked pro); arXiv's primary category (p is true in
  computer science).
- **A fixed rule** on the fields the model sees: the price, a word or a colon in a title, the topic list, the
  number of words.
- **No truth** for a judgment call such as whether a product is a data storage product: the p of the two product
  datasets.

So negation, batch neighbors, position, batching and COUNT are scored on CUAD, ArgQ-30k, arXiv and ArgKP; fusion and
conjunct order on the same four (q is a fixed rule there); symmetry on all three matching datasets (the known
matches); placement and renamed columns on ArgKP. Idempotence has no truth.

## What the repository holds

- **Open datasets:** `data/{cuad,argq,arxiv,argkp}/records.csv`, each with its license notice (`LICENSE.md`).
- **Product datasets:** not redistributed, since neither download page states a license. To rebuild them, download
  WDC Products (`80multi.zip` from [webdatacommons.org](https://webdatacommons.org/largescaleproductcorpus/wdc-products/))
  and Abt-Buy (the textual dataset of [DeepMatcher](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md)),
  then run, from `harness/`:

  ```bash
  python3 -I scripts/convert_wdc_products.py <dir with the WDC files> ../data/wdc-products
  python3 -I scripts/convert_abt_buy.py <dir with the Abt-Buy files> ../data/abt-buy
  ```

  The conversion is deterministic; compare the input with `PROVENANCE.json`. The analyses of identities and
  rewordings do not need these records; accuracy needs them only for symmetry on these two datasets, and skips
  them when they are absent.

## The call logs

`harness/runs/<run>/<model>.jsonl.gz` holds one compressed JSON record per call, for the runs the paper's results
read (`grid/coverage.json`, under `sources`; `rgrid-*` are the reasoning runs). Each record keeps:

- the task (identity, condition, item, repeat), the model and the request settings;
- the model's reply, its parsed answer, and the raw API response;
- the request hash, tokens, cost, the serving host and timing.

Every prompt is removed, because a prompt contains a record's text; the field `redacted` says what was removed. The
prompts are rebuilt from `grid/units/` and the configs by the same builders (`python3 grid/run.py check`). Some runs
were reused from earlier collections of the same requests, so a few logs also hold calls the paper does not use.
