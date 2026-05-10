# Generated Needle Graph Example

## Collapsed RLM View

This is what the run looks like if recursive calls collapse to strings:

```text
root
  call_llm("scan first third")  -> not found
  call_llm("scan middle third") -> decoy, no code
  call_llm("scan final third")  -> candidate code 84721
  call_llm("verify candidate")  -> 84721 matches the requested needle
  final answer                  -> 84721
```

## Sequence View

This is the same run as calls and returns:

```mermaid
sequenceDiagram
    participant root as root
    participant root_chunk_0 as root.chunk_0
    participant root_chunk_1 as root.chunk_1
    participant root_chunk_2 as root.chunk_2
    participant root_chunk_2_candidate_a as root.chunk_2.candidate_a
    participant root_chunk_2_candidate_b as root.chunk_2.candidate_b
    participant root_verify as root.verify
    root->>+root_chunk_0: delegate Scan first third for the hidden secret code.
    root->>+root_chunk_1: delegate Scan middle third for the hidden secret code.
    root->>+root_chunk_2: delegate Scan final third for the hidden secret code.
    root_chunk_0-->>-root: not found
    root_chunk_1-->>-root: decoy, no code
    root_chunk_2->>+root_chunk_2_candidate_a: delegate Inspect candidate window A.
    root_chunk_2->>+root_chunk_2_candidate_b: delegate Inspect candidate window B.
    root_chunk_2_candidate_a-->>-root_chunk_2: decoy: the code is not 12345
    root_chunk_2_candidate_b-->>-root_chunk_2: needle: the secret code is 84721
    root_chunk_2-->>-root: candidate code 84721
    root->>+root_verify: delegate Verify candidate code 84721 against the origi...
    root_verify-->>-root: 84721 matches the requested needle
    root-->>root: done 84721
```

## Steppable Graph Snapshots

### 1. Root parks after spawning parallel children

```mermaid
flowchart TD
    n_fcae719846a6["root<br/><i>query</i>"]:::query
    n_fcae719846a6 --> n_f4dcc74e0f10
    n_f4dcc74e0f10["root<br/><i>action</i>"]:::action
    n_f4dcc74e0f10 --> n_c92b3987e89e
    n_c92b3987e89e["root<br/><i>supervising</i>"]:::sup
    n_c92b3987e89e --> n_c27f13f211e5
    n_c27f13f211e5["root.chunk_0<br/><i>query</i>"]:::query
    n_c92b3987e89e --> n_6cfdfd261ef1
    n_6cfdfd261ef1["root.chunk_1<br/><i>query</i>"]:::query
    n_c92b3987e89e --> n_8d6e2d5d2b7a
    n_8d6e2d5d2b7a["root.chunk_2<br/><i>query</i>"]:::query
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```

### 2. First children finish while chunk_2 keeps working

```mermaid
flowchart TD
    n_fcae719846a6["root<br/><i>query</i>"]:::query
    n_fcae719846a6 --> n_f4dcc74e0f10
    n_f4dcc74e0f10["root<br/><i>action</i>"]:::action
    n_f4dcc74e0f10 --> n_c92b3987e89e
    n_c92b3987e89e["root<br/><i>supervising</i>"]:::sup
    n_c92b3987e89e --> n_c27f13f211e5
    n_c27f13f211e5["root.chunk_0<br/><i>query</i>"]:::query
    n_c27f13f211e5 --> n_44fcfb74a5d5
    n_44fcfb74a5d5["root.chunk_0<br/><i>action</i>"]:::action
    n_44fcfb74a5d5 --> n_21c7de897284
    n_21c7de897284["root.chunk_0<br/><i>result</i><br/>not found"]:::result
    n_c92b3987e89e --> n_6cfdfd261ef1
    n_6cfdfd261ef1["root.chunk_1<br/><i>query</i>"]:::query
    n_6cfdfd261ef1 --> n_ae921b6a0f83
    n_ae921b6a0f83["root.chunk_1<br/><i>action</i>"]:::action
    n_ae921b6a0f83 --> n_7c42c10975a9
    n_7c42c10975a9["root.chunk_1<br/><i>result</i><br/>decoy, no code"]:::result
    n_c92b3987e89e --> n_8d6e2d5d2b7a
    n_8d6e2d5d2b7a["root.chunk_2<br/><i>query</i>"]:::query
    n_8d6e2d5d2b7a --> n_ba73a874f021
    n_ba73a874f021["root.chunk_2<br/><i>action</i>"]:::action
    n_ba73a874f021 --> n_b304b8d53fd9
    n_b304b8d53fd9["root.chunk_2<br/><i>supervising</i>"]:::sup
    n_b304b8d53fd9 --> n_a0e4a44d92b2
    n_a0e4a44d92b2["root.chunk_2.candidate_a<br/><i>query</i>"]:::query
    n_b304b8d53fd9 --> n_3288505316ef
    n_3288505316ef["root.chunk_2.candidate_b<br/><i>query</i>"]:::query
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```

### 3. chunk_2 resumes from candidate readers

```mermaid
flowchart TD
    n_fcae719846a6["root<br/><i>query</i>"]:::query
    n_fcae719846a6 --> n_f4dcc74e0f10
    n_f4dcc74e0f10["root<br/><i>action</i>"]:::action
    n_f4dcc74e0f10 --> n_c92b3987e89e
    n_c92b3987e89e["root<br/><i>supervising</i>"]:::sup
    n_c92b3987e89e --> n_c27f13f211e5
    n_c27f13f211e5["root.chunk_0<br/><i>query</i>"]:::query
    n_c27f13f211e5 --> n_44fcfb74a5d5
    n_44fcfb74a5d5["root.chunk_0<br/><i>action</i>"]:::action
    n_44fcfb74a5d5 --> n_21c7de897284
    n_21c7de897284["root.chunk_0<br/><i>result</i><br/>not found"]:::result
    n_c92b3987e89e --> n_6cfdfd261ef1
    n_6cfdfd261ef1["root.chunk_1<br/><i>query</i>"]:::query
    n_6cfdfd261ef1 --> n_ae921b6a0f83
    n_ae921b6a0f83["root.chunk_1<br/><i>action</i>"]:::action
    n_ae921b6a0f83 --> n_7c42c10975a9
    n_7c42c10975a9["root.chunk_1<br/><i>result</i><br/>decoy, no code"]:::result
    n_c92b3987e89e --> n_8d6e2d5d2b7a
    n_8d6e2d5d2b7a["root.chunk_2<br/><i>query</i>"]:::query
    n_8d6e2d5d2b7a --> n_ba73a874f021
    n_ba73a874f021["root.chunk_2<br/><i>action</i>"]:::action
    n_ba73a874f021 --> n_b304b8d53fd9
    n_b304b8d53fd9["root.chunk_2<br/><i>supervising</i>"]:::sup
    n_b304b8d53fd9 --> n_a0e4a44d92b2
    n_a0e4a44d92b2["root.chunk_2.candidate_a<br/><i>query</i>"]:::query
    n_a0e4a44d92b2 --> n_b63f8488577e
    n_b63f8488577e["root.chunk_2.candidate_a<br/><i>action</i>"]:::action
    n_b63f8488577e --> n_0e98ffff4096
    n_0e98ffff4096["root.chunk_2.candidate_a<br/><i>result</i><br/>decoy: the code is not 12345"]:::result
    n_b304b8d53fd9 --> n_3288505316ef
    n_3288505316ef["root.chunk_2.candidate_b<br/><i>query</i>"]:::query
    n_3288505316ef --> n_ffeeabaca0fb
    n_ffeeabaca0fb["root.chunk_2.candidate_b<br/><i>action</i>"]:::action
    n_ffeeabaca0fb --> n_b2c78d9af6bc
    n_b2c78d9af6bc["root.chunk_2.candidate_b<br/><i>result</i><br/>needle: the secret code is 84721"]:::result
    n_b304b8d53fd9 --> n_60fd6a74241e
    n_60fd6a74241e["root.chunk_2<br/><i>result</i><br/>candidate code 84721"]:::result
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```

### 4. Root resumes and returns the answer

```mermaid
flowchart TD
    n_fcae719846a6["root<br/><i>query</i>"]:::query
    n_fcae719846a6 --> n_f4dcc74e0f10
    n_f4dcc74e0f10["root<br/><i>action</i>"]:::action
    n_f4dcc74e0f10 --> n_c92b3987e89e
    n_c92b3987e89e["root<br/><i>supervising</i>"]:::sup
    n_c92b3987e89e --> n_c27f13f211e5
    n_c27f13f211e5["root.chunk_0<br/><i>query</i>"]:::query
    n_c27f13f211e5 --> n_44fcfb74a5d5
    n_44fcfb74a5d5["root.chunk_0<br/><i>action</i>"]:::action
    n_44fcfb74a5d5 --> n_21c7de897284
    n_21c7de897284["root.chunk_0<br/><i>result</i><br/>not found"]:::result
    n_c92b3987e89e --> n_6cfdfd261ef1
    n_6cfdfd261ef1["root.chunk_1<br/><i>query</i>"]:::query
    n_6cfdfd261ef1 --> n_ae921b6a0f83
    n_ae921b6a0f83["root.chunk_1<br/><i>action</i>"]:::action
    n_ae921b6a0f83 --> n_7c42c10975a9
    n_7c42c10975a9["root.chunk_1<br/><i>result</i><br/>decoy, no code"]:::result
    n_c92b3987e89e --> n_8d6e2d5d2b7a
    n_8d6e2d5d2b7a["root.chunk_2<br/><i>query</i>"]:::query
    n_8d6e2d5d2b7a --> n_ba73a874f021
    n_ba73a874f021["root.chunk_2<br/><i>action</i>"]:::action
    n_ba73a874f021 --> n_b304b8d53fd9
    n_b304b8d53fd9["root.chunk_2<br/><i>supervising</i>"]:::sup
    n_b304b8d53fd9 --> n_a0e4a44d92b2
    n_a0e4a44d92b2["root.chunk_2.candidate_a<br/><i>query</i>"]:::query
    n_a0e4a44d92b2 --> n_b63f8488577e
    n_b63f8488577e["root.chunk_2.candidate_a<br/><i>action</i>"]:::action
    n_b63f8488577e --> n_0e98ffff4096
    n_0e98ffff4096["root.chunk_2.candidate_a<br/><i>result</i><br/>decoy: the code is not 12345"]:::result
    n_b304b8d53fd9 --> n_3288505316ef
    n_3288505316ef["root.chunk_2.candidate_b<br/><i>query</i>"]:::query
    n_3288505316ef --> n_ffeeabaca0fb
    n_ffeeabaca0fb["root.chunk_2.candidate_b<br/><i>action</i>"]:::action
    n_ffeeabaca0fb --> n_b2c78d9af6bc
    n_b2c78d9af6bc["root.chunk_2.candidate_b<br/><i>result</i><br/>needle: the secret code is 84721"]:::result
    n_b304b8d53fd9 --> n_60fd6a74241e
    n_60fd6a74241e["root.chunk_2<br/><i>result</i><br/>candidate code 84721"]:::result
    n_c92b3987e89e --> n_10a45ae36119
    n_10a45ae36119["root<br/><i>resume</i>"]:::resume
    n_10a45ae36119 --> n_7d2f49a5b7e1
    n_7d2f49a5b7e1["root<br/><i>action</i>"]:::action
    n_7d2f49a5b7e1 --> n_4e8630aeb5f3
    n_4e8630aeb5f3["root<br/><i>supervising</i>"]:::sup
    n_4e8630aeb5f3 --> n_289c76c7ee0f
    n_289c76c7ee0f["root.verify<br/><i>query</i>"]:::query
    n_289c76c7ee0f --> n_7dcca8b45141
    n_7dcca8b45141["root.verify<br/><i>action</i>"]:::action
    n_7dcca8b45141 --> n_df7775fdad79
    n_df7775fdad79["root.verify<br/><i>result</i><br/>84721 matches the requested needle"]:::result
    n_4e8630aeb5f3 --> n_d7b27692fa78
    n_d7b27692fa78["root<br/><i>result</i><br/>84721"]:::result
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```
