# Reconstructing Doxygen comments from NVIDIA's published API reference

This folder holds NvMedia API headers whose Doxygen comments were stripped before
they were vendored in. The published DriveOS Linux SDK reference still has the
original commentary, so we reconstruct the comments by scraping those pages and
re-emitting them in-tree.

This guide is for future agents (or future-you) doing the same thing on more
headers, or refreshing existing ones if NVIDIA bumps the SDK version.

## The setup

Each header in this folder has a marker comment near the top:

```c
// API reference: https://developer.nvidia.com/docs/drive/drive-os/<ver>/...nvmedia__foo_8h.html
```

That URL is the file-level Doxygen page. Two related URL shapes matter:

- `nvmedia__foo_8h.html` — file page. Lists every struct, enum, function, and
  macro with brief one-liners.
- `structNvMediaXxx.html` — per-struct page. The only place per-field
  descriptions live.
- `group__nvmedia__foo__api.html` — group/module page. Where per-function
  detail (params, returns, preconditions, usage considerations) lives.

The file page alone is not enough. You need the struct pages for field-level
docs and the group page for function-level docs.

The group-page URL is not always exactly `group__nvmedia__<foo>__api.html` —
some modules use a `group__x__nvmedia__<foo>__api.html` shape (note the
leading `x__`). The 2D module is one example: its group page is
`group__x__nvmedia__2d__api.html`. Don't guess the URL — the file page links
to the real group page; follow that link rather than constructing it.

## The output style

Look at `nvmedia_common_encode.h` and `nvmedia_iep.h` for canonical examples
(both done across this work). The convention:

```c
/**
 * \file
 * \brief NVIDIA Media Interface: <copy upstream brief>
 *
 * <copy upstream description>
 */

/** \brief <one-line macro description>. */
#define NVMEDIA_FOO_BAR    42U

/**
 * \brief <struct purpose>.
 */
typedef struct {
    int16_t x;          /**< Inline tail comment when short. */
    /** Multi-line block comment when the description doesn't fit on one line. */
    uint32_t verbose_field_with_a_long_description;
} NvMediaFoo;

/**
 * \brief <function purpose>.
 *
 * \param[in] foo  <copy upstream description>
 * \param[out] bar <copy upstream description>
 *
 * \return
 * - \ref NVMEDIA_STATUS_OK if successful.
 * - \ref NVMEDIA_STATUS_BAD_PARAMETER if foo is NULL.
 *
 * \pre <copy upstream precondition>
 *
 * Usage considerations
 * - Allowed context for the API call
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Thread-safe: Yes
 *   - Re-entrant: No
 *   - Async/Sync: Sync
 * - Required privileges: None
 * - API group
 *   - Init: Yes
 *   - Runtime: No
 *   - De-Init: No
 */
NvMediaStatus NvMediaFooBar(...);
```

Direction tags `[in]`, `[out]`, `[in,out]` are part of NVIDIA's published docs —
preserve them as-is.

## The non-negotiable rules

These rules drive every fidelity decision. When in doubt, fall back on them.

1. **Verbatim text only.** Copy upstream wording byte-for-byte. If upstream
   says "Holds a copy" don't write "Stores a copy". If a sentence has a typo
   ("cordinate", "consituent", "Huffmann", "an Video", missing periods,
   lowercase sentence starts), preserve the typo. NVIDIA's docs have many of
   them; reproducing them keeps a local `doxygen` run byte-identical to the
   published reference.

2. **Don't invent.** If a field, parameter, or enumerator has no upstream doc,
   leave it undocumented. Better silent than fabricated.

3. **Don't touch C declarations.** Struct bodies, types, names, ordering,
   whitespace, `// coverity[...]` markers, History blocks at file end — every
   non-comment character must remain identical. You are *only* adding comment
   blocks between existing declarations.

4. **Don't add or remove `#include`s.** Some upstream headers reference types
   without including the declaring header (e.g. `nvmedia_iep_output_extradata.h`
   uses `NvMediaVideoCodec` without including `nvmedia_common_encode_decode.h`;
   `nvmedia_tensormetadata.h` uses `uint32_t` without including `<stdint.h>`).
   Real consumers include parent headers first, so it works in practice. Don't
   "fix" this — it's an upstream property.

## The process

For a single header:

1. **Read the local header** to map every macro, type, struct, enum, and
   function that needs a doc block. Note the order and exact declarations.

2. **Fetch the file-level page** with WebFetch. This gives you struct/enum/
   function brief lines and macro descriptions. Useful but rarely sufficient.

3. **Fetch the per-struct pages** for any struct with member-level docs.
   These are at `structNvMediaXxx.html`.

4. **Fetch the group page** for function-level detail (params, returns,
   `\pre`, Usage considerations). Often at
   `group__nvmedia__foo__api.html`. The file page links to it.

5. **Write the file** using the Write tool — full rewrite is simpler than
   inserting comment blocks one at a time.

6. **Verify with clang.** Run:
   ```bash
   clang -fsyntax-only -x c -std=c11 -I/home/tegra/hmdcam \
         -I/home/tegra/hmdcam/nvmedia /home/tegra/hmdcam/nvmedia/<file>.h
   ```
   Expected pre-existing errors (not caused by your edits):
   - `'nvscibuf.h' file not found` / `'nvscisync.h' file not found` — these are
     external system headers from the NVIDIA SDK and aren't installed on this
     machine.
   - `unknown type name 'NvMediaVideoCodec'` in `nvmedia_iep_output_extradata.h`
     — pre-existing missing-include in upstream.
   - `unknown type name 'uint32_t'` in `nvmedia_tensormetadata.h` — same.

   Anything *new* — unterminated comment, stray brace, parse error mid-struct —
   is from your edits and must be fixed.

## Fighting WebFetch paraphrase

This is the single biggest gotcha. WebFetch routes the page through a small
summarizer LLM, and that model loves to paraphrase prose, especially in
function-detail blocks. You'll get text like "Returns the device handle" when
upstream actually says `"An NvMedia DLA device handle."`. That's a fidelity
break.

**Symptoms** of paraphrase: text that's not in quote marks; rephrased openings
like "This function does...", "Sets up a..."; consolidated bullet lists; missing
"Usage considerations" tables.

**Counter-prompt** (use this verbatim or close to it inside the WebFetch
`prompt` argument):

> CRITICAL: This is a transcription task, not a summarization task. You must
> return the EXACT verbatim text from the page, character by character, in
> quotation marks. Do not rephrase. Do not summarize. Do not "clean up" wording.
> Do not skip sentences. Do not consolidate "Usage considerations" rows.
>
> For EACH function/struct on the page, return:
> 1. The exact `\brief` description string in quotation marks.
> 2. EVERY `\param` with its EXACT direction tag and EXACT description text.
> 3. The EXACT `\returns` text including each enumerated NVMEDIA_STATUS_* value.
> 4. Any `\pre`, `\post`, `\note`, `\implements`, `\sa` blocks — verbatim.
> 5. The "Usage considerations" table rows verbatim.
>
> If you find yourself starting to paraphrase ("This function does X", "Sets up
> the Y"), stop and copy the source text instead. Wrap every transcribed string
> in quote marks "" so I can tell it from your own commentary.

If a fetch still comes back paraphrased, re-fetch with even tighter wording
focused on the specific items that came back rephrased. Anything *not* inside
literal quote marks in the response is the LLM's words, not NVIDIA's — discard
or re-fetch.

`nvmedia_dla_nvscisync.h` was redone using this counter-prompt after a first
pass returned only paraphrased detail; the rerun produced clean verbatim text
for all 12 functions.

## Delegation pattern for batches

When doing several headers at once, delegate one header per general-purpose
agent so each agent's WebFetch context stays focused on a single file. Run
agents in parallel (single message, multiple `Agent` tool calls) — wall time
becomes the slowest single header rather than the sum.

Brief each agent with:
- Explicit path to the local file.
- Explicit reference URL.
- Pointer to `nvmedia_common_encode.h` as a concrete style example.
- Inline list of every type/function the agent must document (so the agent
  can't silently miss one).
- The "Fighting WebFetch paraphrase" counter-prompt above.
- A clang verify step + report-back template.

Verify each agent's result yourself with `clang -fsyntax-only` after it
reports. Agents in this sandbox have been observed to fail on the bash
permission for `clang` — don't trust their self-reported "should compile"
claims.

## Coverage status (as of this writing)

These headers were annotated as part of this work (DriveOS 6.0.10 reference):

- `nvmedia_2d.h`
- `nvmedia_2d_sci.h`
- `nvmedia_common_encode.h`
- `nvmedia_common_encode_decode.h`
- `nvmedia_common_decode.h`
- `nvmedia_core.h`
- `nvmedia_dla.h`
- `nvmedia_dla_nvscisync.h`
- `nvmedia_ide.h`
- `nvmedia_iep.h`
- `nvmedia_iep_input_extradata.h`
- `nvmedia_iep_output_extradata.h`
- `nvmedia_ijpe.h`
- `nvmedia_ijpd.h`
- `nvmedia_iofa.h`
- `nvmedia_tensor.h`
- `nvmedia_tensor_nvscibuf.h`
- `nvmedia_tensormetadata.h`

If NVIDIA refreshes the SDK and you re-do any header, the version number in
each `API reference: ...` URL is the only thing that needs to change to point
at a newer reference.
