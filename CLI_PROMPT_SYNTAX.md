# OwlScript Syntax - Quick Guide

Owlsight understands two special syntaxes in the **How can I assist you?** field:

| Purpose | Syntax | Quick Example |
|---------|--------|---------------|
| Insert a Python value | `{{ expression }}` | `The answer is {{a * 2}}` |
| Attach / control extras | `[[tag:payload || option=value || ...]]` | `[[image:cats/kitten.jpg]]` |

---

## 1. `{{ … }}` – Python interpolation

* **What happens?** The expression is evaluated in the *current* Python interpreter; the result is stringified and inserted before the prompt is sent to the model.

* **Rules**  
  • Any single-line Python expression.  
  • Nested braces allowed (`file_{{idx}}.txt`).  
  • Exceptions abort the prompt and show a traceback.

* **Example**

```text
python > x = 21
How can I assist you? > Double it → {{x * 2}}
```
Model sees: `Double it → 42`

---

## 2. `[[tag:payload]]` – square-bracket tags

`tag` must be one of:

| Tag   | What it does                                | Payload           |
|-------|---------------------------------------------|-------------------|
| image | Send an image to a multimodal model         | file path / URL   |
| audio | Send an audio clip                          | file path / URL   |
| video | (reserved) future video support             | file path / URL   |
| load  | Load another configuration JSON mid-chat    | path to .json     |
| chain | Inline-edit config parameters               | `key=value` pairs |

Media tags may include **options** after `||`:

```text
[[image:cat.png||width=512||height=512]]
```

### Examples

```text
# Multimodal question
What is this? [[image:photos/dog.jpg]]

# Switch model then ask
[[load:qa-model.json]] How many teeth does a shark have?

# Adjust temperature
[[chain:generate.temperature=0.7]]
```

---

## 3. Order of operations

Owlsight expands a prompt in **four strict passes**.

1. **Literal text** – anything outside the special markers is copied verbatim.
2. **Python interpolation (`{{ … }}`)** – every placeholder is evaluated in the running Python interpreter and substituted by its `str()` (unless the *whole* prompt is a single placeholder, in which case the raw value may be returned).
3. **Square-bracket tags (`[[tag:payload || k=v …]]`)**
   * While a tag is being parsed, its `payload` and every `option=value` loop back to step&nbsp;2, so they may themselves contain `{{ … }}`.
   * When finished, the tag is replaced by an internal sentinel such as `__MEDIA_0__`, and a `MediaObject` (or `DoubleBracketsTag`) with the resolved data is stored for the backend call.
4. **Left-over curly braces** – after tag substitution the whole string is scanned once more for any remaining `{{ … }}` that were outside tags.

---

### Example

```text
How can I assist you? > Compare [[image:{{folder}}/frame_{{idx}}.png||width={{size}}]] with the result {{score + 1}}
```

Assuming `folder="imgs"`, `idx=7`, `size=512`, `score=5`, the processing yields:

```text
Compare __MEDIA_0__ with the result 6
```

with `__MEDIA_0__ → {tag: "image", path: "imgs/frame_7.png", options: {width: "512"}}`.


---

## 4. Handy shortcuts

* Type `{{` then **ESC + V** → autocomplete Python variables.
* Type `[[` then **ESC + V** → autocomplete tag names (`image:`, `audio:` …).

---

## 5. Common error messages

| Message | Likely cause |
|---------|--------------|
| Error evaluating ‘…’ | Python expression raised an exception |
| Invalid media tag | Tag not in `image audio video load chain` |
| Media path cannot be empty | Missing payload after the colon |
| Invalid option format | Options must be `key=value` |

---

## TODO – Future language design

* Replace regex-based parsing with a **formal grammar + AST** (PEG/LL) to improve maintainability.
* Add an explicit *section layer* using `--- python` / `--- prompt` blocks.
  * Code inside a `python` section executes once; its functions & vars become reusable in later prompt sections.
* Processing pipeline to be clarified in docs & code:
  1. **Lex/tokenise** → plain tokens.
  2. **Parse** into `Prompt`, `Section`, `PyBlock`, `PyExpr`, `MediaTag`, `Literal` nodes.
  3. **Static analysis** to build a shared namespace and catch undefined names early.
  4. **Evaluate** blocks/expressions; then render media tags.
* Keep legacy syntax as fallback while unit-tests are migrated.
* Benefits: scalability, richer DSL (loops/conditionals), precise error messages, reusable code.

---

Happy prompting! 🎉
