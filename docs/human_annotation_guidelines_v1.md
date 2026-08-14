# SHERLOC human annotation guidelines v1

Version: `1.0.0`  
Reliability sample: `sherloc-reliability-sample-v1`  
Sentence splitter: `sherloc_sentence_splitter_v1`  
AMP ontology: `sherloc-legacy-amp-v1`

## 1. Purpose and scope

These instructions govern two independent, narrative-only annotations of the
same 100 SHERLOC cases. The goal is to measure whether a reader can recover
trafficking Acts, Means, Purposes, geographic Form, victim multiplicity, and
child/minor involvement from the English Fact Summary alone.

The annotation is not a correction of SHERLOC and is not a legal finding about
facts outside the supplied narrative. Reviewers must record what the English
Fact Summary supports, including uncertainty and abstention. A human label may
therefore disagree with a hidden SHERLOC value without either source being
treated as a transcription error.

This is annotation version 1. It does not cover Sector of Exploitation, exact
victim counts, defendant-level linkage, label normalization, modeling, or
train/validation/test splitting.

## 2. Blinding and permitted information

Each reviewer receives only blinded annotation material:

- a neutral `reliability_case_id` such as `HRV1-001`;
- a blank `reviewer_id` field for the project-assigned reviewer code;
- non-substantive integrity metadata: sentence-splitter version, sentence count,
  and a hash of the numbered text;
- the English Fact Summary with stable sentence IDs; and
- empty annotation fields.

The reviewer-facing file intentionally omits search rank, title, jurisdiction,
URL, sampling bucket, all Legacy and Sidebar values, provisional multiplicity
and child labels, prior audit judgments, and selection reasons. Jurisdiction is
omitted because it can encourage geographic-Form inference that is not grounded
in the narrative. Place names that occur naturally in the Fact Summary remain
part of the evidence.

Reviewers must not:

1. open the management sample or researcher-only reference key;
2. search SHERLOC, the web, the repository, or another source for a case;
3. infer a label from a remembered case or its likely jurisdiction;
4. discuss individual cases with the other reviewer during Stage 1; or
5. inspect the other reviewer's file before both independent files are locked.

General language knowledge may be used to understand the text. For geographic
Form, however, use only country relationships or geographic scope made explicit
in the supplied narrative. Do not classify a city-to-city route from personal
geographic knowledge, and do not look up an unfamiliar place. If the relevant
same-country or cross-country relationship is not explicit, choose `UNKNOWN` or
`PARTIAL` as specified below.

The internal files
`reliability_sample_100.csv` and
`reliability_sample_100_reference_key.csv` are project-management artifacts,
not reviewer materials. They must remain inaccessible until both independent
annotations are complete.

## 3. Unit of annotation and general evidence policy

The unit is one complete English Fact Summary. Annotate every focal trafficking
episode described in that summary and return one case-level label set or class
for each target.

Use the following rules across all six targets:

- Select only labels affirmatively supported by the supplied narrative.
- Do not infer a label merely because it is common in trafficking cases.
- Do not use absence of a statement as evidence that the opposite is true.
- Treat explicit numbers, ages, routes, actions, quotations, and attributed
  findings as stronger evidence than contextual implication.
- Keep actors and roles straight. A defendant, recruiter, smuggled migrant,
  client, witness, plaintiff, family member, or undercover officer is not
  automatically a trafficking victim.
- A focal victim is a person described as the object of the trafficking scheme
  or its intended exploitation. Do not count incidental people.
- A description of charges or allegations may support a narrative label when
  it affirmatively attributes the conduct to the focal trafficking theory.
  Note `ALLEGED` when the qualification matters.
- An explicitly described attempted act or intended exploitation may support
  the relevant Act or Purpose; note `ATTEMPT`. An attempted transaction with an
  undercover officer does not establish an actual child victim or a victim
  count.
- If the summary expressly says conduct was not proved, did not happen, or
  involved no actual victim, do not silently convert that rejected proposition
  into a positive factual label. Use the answerability and notes fields.
- For trafficking involving a child, the legal Means element may not be
  required, but this annotation task still records any Means actually described
  in the narrative. Do not add a Means label solely because the victim is a
  child.

When the summary describes several episodes, select the union of all supported
AMP labels. For Form, use `BOTH` only under the rule in Section 8. Do not invent
episode-to-label links that the summary does not make.

## 4. Stable sentence IDs and evidence entry

The benchmark retains the original English Fact Summary separately. The
reviewer copy is deterministically segmented by
`sherloc_sentence_splitter_v1`:

1. convert `CRLF` and bare `CR` line endings to `LF`;
2. treat a blank-line paragraph break as a hard sentence boundary;
3. treat a line-leading list marker as a hard boundary: `-`, `*`, `•`, or a
   one-to-three-character alphanumeric item marker followed by `.` or `)`;
4. collapse other whitespace runs within a segment to one ASCII space;
5. within a paragraph/list unit, split after `.`, `?`, or `!` (including a following
   closing quote or bracket) when the next nonspace character is an uppercase
   letter, or an opening quote/bracket followed by an uppercase letter;
6. protect common titles, legal abbreviations, citation abbreviations, and
   initials, including `Mr.`, `Mrs.`, `Ms.`, `Dr.`, `Prof.`, `Sr.`, `Jr.`,
   `St.`, `No.`, `Nos.`, `Art.`, `Arts.`, `Sec.`, `Secs.`, `para.`,
   `paras.`, `v.`, `vs.`, `e.g.`, `i.e.`, `etc.`, `U.S.`, `U.K.`, and
   single-letter initials; and
7. number the resulting nonempty segments in order as `[S1]`, `[S2]`, and so
   on. A nonempty summary that produces no punctuation boundary is `[S1]`.

The splitter version, source text, and neutral case ID together determine the
sentence IDs. Reviewers must not renumber, edit, or resegment the supplied text.

Evidence fields contain semicolon-separated IDs in ascending order, for
example `S2;S5`. Cite the smallest set of sentences that makes the label
defensible. Do not paste long quotations into an evidence field.

For a multi-label target, the evidence field may support different labels in
different sentences. If the mapping would otherwise be unclear, explain it
briefly in notes, for example:

```text
S2 supports ACT_RECRUITMENT; S4 supports ACT_TRANSPORTATION.
```

For `UNKNOWN`, `PARTIAL`, or a conflict, cite the sentence that creates the
uncertainty when one exists. Leave evidence blank if the narrative contains no
target-relevant statement. Notes are required whenever an answer is `PARTIAL`,
whenever a nonempty label set is based on an allegation or attempt, and whenever
an auxiliary label is `UNKNOWN` for a reason more specific than total silence.

## 5. Shared answerability scale

Answerability is target-specific. A case may be `YES` for Purpose and `NO` for
Means.

| Value | Operational definition |
|---|---|
| `YES` | The narrative contains enough direct information to assign the target confidently. For AMP, at least one label is supported and there is no apparent unresolved target-relevant omission or contradiction that would materially change the label set. |
| `PARTIAL` | At least some target information is recoverable, but one or more material labels, roles, time points, routes, or alternatives remain unresolved, or the summary expressly provides only part of the relevant facts. Record the supported labels and explain the limitation. |
| `NO` | The narrative supplies no defensible target label, contains only procedural/reference material, or is so incomplete or contradictory that the target cannot be assigned. AMP labels must be empty; auxiliary labels must be `UNKNOWN`. |

`PARTIAL` does not mean low annotator confidence about an otherwise complete
answer. It means the text itself is only partly sufficient. If the text is
complete but a reviewer is personally unsure, reread the definitions, select
the best rule-governed answer, and describe the ambiguity in notes.

For Act, Means, and Purpose, an empty label list is permitted only with
answerability `NO`. For Form, Multiplicity, and Child, `UNKNOWN` is a real output
class rather than a blank value. Ordinarily:

- `YES` accompanies a non-`UNKNOWN` auxiliary label;
- `NO` accompanies `UNKNOWN`; and
- `PARTIAL` may accompany either a supported non-`UNKNOWN` label whose scope is
  incomplete or `UNKNOWN` when conflicting/partial evidence prevents a class.

## 6. Act: multi-label

Enter zero or more of the following machine IDs in ontology order, separated by
semicolons. The quoted SHERLOC strings are the exact Legacy ontology labels;
they are not shown as case-specific reference values.

| Machine ID | Exact Legacy label | Narrative annotation rule |
|---|---|---|
| `ACT_RECRUITMENT` | `Recruitment` | The actor solicits, lures, induces, enlists, hires, or arranges for a person to enter the trafficking or exploitation process. A false job or relationship offer can support Recruitment when it is used to obtain the person. Later exploitation alone does not prove Recruitment. |
| `ACT_TRANSPORTATION` | `Transportation` | The victim is physically moved, carried, driven, flown, escorted, or has travel arranged and completed from one place to another as part of the scheme. No international border is required. Mere presence at a location does not prove Transportation. |
| `ACT_TRANSFER` | `Transfer` | Control, custody, possession, or responsibility for the victim is handed, sold, exchanged, or delivered from one actor to another. Do not use Transfer as a synonym for geographic movement when no change of control is described. |
| `ACT_HARBOURING` | `Harbouring` | The actor houses, shelters, accommodates, conceals, confines, keeps, or provides a controlled place for the victim as part of the scheme. A location mentioned only as the scene of an offence is insufficient. |
| `ACT_RECEIPT` | `Receipt` | An actor accepts, buys, takes custody of, or otherwise receives control of a victim from another person. Receiving money, services, or criminal proceeds is not Receipt of a person. |

Important distinctions:

- Travel may support Transportation without Transfer.
- A sale may support Transfer for the person relinquishing control and Receipt
  for the person acquiring control when both roles are described.
- Harbouring is continued placement/control at a location; Receipt is acquiring
  the person. They may co-occur but neither implies the other.
- Generic language such as “was trafficked” is insufficient to select a
  specific Act unless the summary states what happened.

Record:

- `act_labels`
- `act_answerability`
- `act_evidence_sentence_ids`
- `act_notes`

## 7. Means: multi-label

Enter zero or more machine IDs in the order below.

| Machine ID | Exact Legacy label | Narrative annotation rule |
|---|---|---|
| `MEANS_THREAT_FORCE_OR_COERCION` | `Threat or use of force or other forms of coercion` | Explicit violence, threatened violence, physical restraint, intimidation, document confiscation used for control, debt coercion, threats to relatives, or another stated form of compelled compliance. Poor conditions alone are insufficient without coercive use. |
| `MEANS_ABDUCTION` | `Abduction` | Kidnapping, forcible taking, seizure, or carrying away without lawful consent. Ordinary transport, even exploitative transport, is not automatically Abduction. |
| `MEANS_FRAUD` | `Fraud` | A materially fraudulent scheme, transaction, document, identity, contract, or legal/financial misrepresentation is described. Do not add Fraud to every deceptive promise. |
| `MEANS_DECEPTION` | `Deception` | False promises, lies, misrepresentation, concealment of the real work or conditions, or another described device that causes the person to agree or comply. |
| `MEANS_ABUSE_OF_POWER_OR_VULNERABILITY` | `Abuse of power or a position of vulnerability` | The actor exploits authority, dependency, poverty, disability, insecure immigration status, family control, youth, isolation, or another stated vulnerability such that the person's realistic alternatives are constrained. Vulnerability must be textually stated or concretely demonstrated; do not infer it from nationality or victim status alone. |
| `MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL` | `Giving or receiving payments or benefits to achieve the consent of a person having control over another person` | Money or another benefit is given to, or received by, a parent, custodian, controller, recruiter, or other person to secure that person's consent/control over the victim. Payment to the victim, payment of travel costs, wages, purchase of services, or receipt of exploitation proceeds alone does not qualify. |

Fraud and Deception remain distinct because the frozen ontology keeps both raw
Legacy labels. Select both only when the narrative independently supports both
rules. A false job promise normally supports Deception; forged passports or a
fraudulent contract may additionally support Fraud.

Record:

- `means_labels`
- `means_answerability`
- `means_evidence_sentence_ids`
- `means_notes`

## 8. Purpose: multi-label

Purpose concerns the intended or realized exploitation. Enter zero or more
machine IDs in ontology order.

| Machine ID | Exact Legacy label | Narrative annotation rule |
|---|---|---|
| `PURPOSE_SEXUAL_EXPLOITATION` | `Exploitation of the prostitution of others or other forms of sexual exploitation` | Compelled or exploitative prostitution, commercial sex, sexual services, pornography, or another explicit form of sexual exploitation. A consensual adult sexual relationship without exploitation is insufficient. |
| `PURPOSE_FORCED_LABOUR_OR_SERVICES` | `Forced labour or services` | Work or services are exacted through force, threat, coercion, deception/control, or inability to leave. A poor wage or labour-law violation alone is insufficient unless compelled labour/services are described. |
| `PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES` | `Slavery or practices similar to slavery` | The summary explicitly describes slavery, slave-like ownership/control, sale as a person, debt bondage, or a clearly stated slavery-like practice. Do not infer this label from harsh work alone. |
| `PURPOSE_SERVITUDE` | `Servitude` | The summary explicitly describes servitude or sustained compelled service under domination/dependency from which the victim cannot realistically escape. Do not use it as an automatic synonym for all forced labour. |
| `PURPOSE_REMOVAL_OF_ORGANS` | `Removal of organs` | The scheme has the intended or realized removal, procurement, sale, or transplant of a victim's organ. Distinguish a trafficked donor/victim from a voluntary self-sale prosecution with no controller or focal trafficking victim. Do not extend this frozen label to tissue removal. |
| `PURPOSE_OTHER` | `Other` | An explicit exploitation purpose falls outside the five preceding Legacy categories, such as forced begging, compelled criminal activity, forced marriage, or another clearly described exploitation. `Other` is not a fallback for missing, vague, or unclassifiable facts. Explain the specific purpose in notes. |

Multiple Purpose labels may co-occur. A false promise of restaurant, domestic,
or factory work is a Means cue, not proof that the promised sector was the
actual exploitation Purpose. Use the intended or realized exploitative outcome,
not the recruitment pretext.

Record:

- `purpose_labels`
- `purpose_answerability`
- `purpose_evidence_sentence_ids`
- `purpose_notes`

## 9. Geographic Form

Choose exactly one:

- `INTERNAL`
- `TRANSNATIONAL`
- `BOTH`
- `UNKNOWN`

Apply these rules:

### `INTERNAL`

The narrative affirmatively places the focal trafficking process within one
country. This can be stated as internal/domestic trafficking or shown by an
explicit origin and destination that the supplied text identifies as being in
the same country.
Movement is not required if the narrative explicitly calls the trafficking
internal. Do not infer `INTERNAL` merely because no international movement is
mentioned.

### `TRANSNATIONAL`

The narrative affirmatively describes a focal trafficking route or trafficking
process crossing a national border, or explicitly uses terms such as
international, transnational, trafficked abroad, or brought from one named
country to another. Foreign nationality, immigration status, or earlier travel
does not by itself prove that the trafficking was transnational. A victim who
was already visiting or residing abroad may subsequently be trafficked only
internally.

### `BOTH`

The narrative describes both a within-country trafficking route/process and a
cross-border trafficking route/process. The two forms may concern one extended
episode or distinct focal episodes. Do not choose `BOTH` merely because a
transnational route necessarily contains local travel segments; the summary
must independently support internal trafficking as well as transnational
trafficking.

### `UNKNOWN`

Use `UNKNOWN` when the summary lacks route/scope information, when only a place
of investigation or exploitation is named, when foreign nationality is the only
cross-border cue, or when classifying named cities/regions would require
personal geographic knowledge, outside research, or a contestable geopolitical
assumption.

Examples of common traps:

- “rescued in Manila” does not prove Internal;
- “a Romanian woman in Italy” does not prove Transnational;
- travel to the country before recruitment does not prove transnational
  trafficking; and
- multiple defendants or an organized group says nothing by itself about
  geographic Form.

Record:

- `form_label`
- `form_answerability`
- `form_evidence_sentence_ids`
- `form_notes`

## 10. Victim multiplicity

Choose exactly one:

- `SINGLE`
- `MULTIPLE`
- `UNKNOWN`

This is a coarse case-level target, not exact victim counting.

### `SINGLE`

The narrative identifies one actual focal trafficking victim and remains
consistently individual-specific. An explicit “one victim,” one named victim,
or a sufficiently detailed consistently singular account can support `SINGLE`.
Do not choose `SINGLE` only because the first sentence says “the victim”; the
summary may later mention others.

### `MULTIPLE`

The narrative explicitly identifies at least two actual focal trafficking
victims through a number, distinct individuals, plural victims, or a clearly
defined group. A single collective description can support `MULTIPLE`; database
record count is irrelevant to the reviewer.

### `UNKNOWN`

Use `UNKNOWN` when no actual victim is established, victim and non-victim roles
cannot be separated, singular/plural scope is unclear, a plaintiff is not
clearly a trafficking victim, or the text speaks only generically about victims.
Undercover officers, clients, defendants, migrants in a smuggling account, and
family members do not count unless the narrative makes them focal trafficking
victims.

Do not infer an exact number and do not use the number of defendants, charges,
plaintiffs, or database person records as a proxy.

Record:

- `multiplicity_label`
- `multiplicity_answerability`
- `multiplicity_evidence_sentence_ids`
- `multiplicity_notes`

## 11. Child/minor involvement

Choose exactly one:

- `TRUE`
- `FALSE`
- `UNKNOWN`

The time point is the focal trafficking or exploitation episode, not arrest,
trial, appeal, or database publication.

### `TRUE`

At least one actual focal trafficking victim is affirmatively under 18 at the
relevant time. Strong evidence includes an age from 0 through 17 or an explicit
term such as child, minor, juvenile, underage boy, or underage girl tied to the
victim and episode.

### `FALSE`

The narrative provides sufficient affirmative evidence that every actual focal
trafficking victim is an adult at the relevant time. Strong evidence includes
explicit adult status or ages 18 and above for all focal victims. This is an
adult-only conclusion, not the absence of child language.

### `UNKNOWN`

Use `UNKNOWN` when ages are missing; only vague terms such as young are used;
some victims have known adult ages but others are unaged; age is disputed;
current age cannot be mapped to the trafficking period; a person's historical
childhood is mentioned; or the only children are a victim's children, an
employer's children, or fictitious/undercover identities. An intended offence
against a purported child with no actual child victim is `UNKNOWN`, not `TRUE`
or adult-only `FALSE`.

Do not automatically treat `woman`, `man`, `female`, `male`, migrant, worker,
student, wife, husband, mother, or father as proof of adult status. Conversely,
do not treat `young woman` or `young man` as proof of minority.

Record:

- `child_label`
- `child_answerability`
- `child_evidence_sentence_ids`
- `child_notes`

## 12. Case-level narrative sufficiency

Record one optional overall judgment:

| Value | Rule |
|---|---|
| `HIGH` | Act, Means, and Purpose are all `YES`; at least two of Form, Multiplicity, and Child are `YES`; and no target is `NO`. |
| `MODERATE` | The narrative provides useful target evidence but does not meet `HIGH` or `LOW`, including mixed `YES`, `PARTIAL`, and `NO` results. |
| `LOW` | At least four targets are `NO` and none of Act, Means, or Purpose is fully `YES`, or the summary expressly states that underlying facts are unavailable. |

Use `annotation_notes` for a concise case-level issue that affects multiple
targets, such as an attempted offence, conflicting ages, a purely procedural
summary, or uncertainty over who the focal victim is. Do not repeat all
task-specific notes.

## 13. Reviewer file format

Do not add free-text labels. Use the exact capitalization shown in this guide.
Multi-label and evidence lists use semicolons without duplicate values. AMP
labels must appear in ontology order, not in narrative order.

| Column | Allowed entry |
|---|---|
| `reviewer_id` | Enter the project-assigned reviewer code, not a name |
| `reliability_case_id` | Supplied neutral ID; do not edit |
| `sentence_splitter_version` | Supplied integrity metadata; do not edit |
| `sentence_count` | Supplied integrity metadata; do not edit |
| `numbered_text_sha256` | Supplied integrity metadata; do not edit |
| `fact_summary_numbered` | Supplied numbered narrative; do not edit |
| `act_labels` | Semicolon-separated Act IDs, or blank only with `NO` |
| `act_answerability` | `YES`, `PARTIAL`, `NO` |
| `act_evidence_sentence_ids` | Semicolon-separated sentence IDs or blank |
| `act_notes` | Free text, optional except under the rules above |
| `means_labels` | Semicolon-separated Means IDs, or blank only with `NO` |
| `means_answerability` | `YES`, `PARTIAL`, `NO` |
| `means_evidence_sentence_ids` | Semicolon-separated sentence IDs or blank |
| `means_notes` | Free text |
| `purpose_labels` | Semicolon-separated Purpose IDs, or blank only with `NO` |
| `purpose_answerability` | `YES`, `PARTIAL`, `NO` |
| `purpose_evidence_sentence_ids` | Semicolon-separated sentence IDs or blank |
| `purpose_notes` | Free text |
| `form_label` | `INTERNAL`, `TRANSNATIONAL`, `BOTH`, `UNKNOWN` |
| `form_answerability` | `YES`, `PARTIAL`, `NO` |
| `form_evidence_sentence_ids` | Semicolon-separated sentence IDs or blank |
| `form_notes` | Free text |
| `multiplicity_label` | `SINGLE`, `MULTIPLE`, `UNKNOWN` |
| `multiplicity_answerability` | `YES`, `PARTIAL`, `NO` |
| `multiplicity_evidence_sentence_ids` | Semicolon-separated sentence IDs or blank |
| `multiplicity_notes` | Free text |
| `child_label` | `TRUE`, `FALSE`, `UNKNOWN` |
| `child_answerability` | `YES`, `PARTIAL`, `NO` |
| `child_evidence_sentence_ids` | Semicolon-separated sentence IDs or blank |
| `child_notes` | Free text |
| `overall_narrative_sufficiency` | `HIGH`, `MODERATE`, `LOW` |
| `annotation_notes` | Free text |

Do not add external case identifiers to notes.

## 14. Recommended annotation sequence

For each case:

1. Read the full numbered Fact Summary once without labeling.
2. Identify the focal trafficking episode(s), victim role(s), and time point.
3. Annotate Act, then Means, then Purpose using only affirmative evidence.
4. Annotate geographic Form from explicit route/scope evidence.
5. Annotate Multiplicity from actual focal victims, not grammatical or database
   record proxies.
6. Annotate Child from age at the trafficking episode.
7. Assign target-specific answerability and minimal evidence IDs.
8. Assign overall narrative sufficiency.
9. Reread the entire summary and confirm that later sentences do not contradict
   an earlier singular, adult-only, or domestic-only conclusion.

Reviewers may revise their own earlier rows during Stage 1, but must not see the
other reviewer's answers or the hidden key.

## 15. Two-reviewer and adjudication workflow

### Stage 1: independent annotation

- Reviewer A annotates all 100 cases.
- Reviewer B independently annotates the same 100 cases.
- Both use the same frozen template, sentence IDs, ontology, and guide.
- Neither sees the other's answers or any structured SHERLOC target value.
- Each reviewer runs the completion checklist below and submits a locked copy.

### Stage 2: agreement analysis — later task

Only after both files are locked, a later reproducible script will calculate:

- raw agreement for categorical tasks;
- Cohen's kappa where its assumptions are appropriate;
- exact-set agreement and Jaccard similarity for multi-label AMP;
- per-label agreement;
- answerability agreement; and
- disagreement lists for adjudication.

No agreement statistic is calculated during this benchmark-construction task.
Prevalence-sensitive metrics will be interpreted alongside raw class counts and
agreement, especially for Child and rare AMP labels.

### Stage 3: adjudication — later task

After Stage 2, reviewers inspect only disagreements, discuss the cited narrative
evidence, and record one adjudicated label set/class, answerability value, and
evidence set. The hidden SHERLOC reference may be examined only after the
reviewers have explained their independent narrative judgments. SHERLOC does
not automatically override the adjudicated narrative-grounded result.

## 16. Completion and quality-control checklist

Before submission, each reviewer confirms:

- exactly 100 neutral case IDs are present and unique;
- supplied IDs and numbered narratives are unchanged;
- every target has an allowed answerability value;
- AMP labels use only the 5/6/6 ontology IDs and are in ontology order;
- no AMP label list is empty when answerability is `YES` or `PARTIAL`;
- every auxiliary target contains exactly one allowed class;
- auxiliary `NO` entries use `UNKNOWN`;
- every evidence ID exists in that case's numbered narrative;
- evidence IDs are unique and ascending;
- `PARTIAL`, allegations, attempts, conflicts, and nontrivial `UNKNOWN` cases
  have notes;
- `FALSE` Child cases contain affirmative adult-only evidence;
- `INTERNAL` was never assigned merely because no border crossing was stated;
- Multiplicity does not count defendants, clients, records, or incidental
  people; and
- no hidden management/reference file or external case source was consulted.

Questions about a definition should be logged without revealing the case or a
proposed answer to the other reviewer. If the project lead issues a rule
clarification, it must be written, versioned, and provided identically to both
reviewers. A rule change after annotation begins requires recording which rows
were revisited.
