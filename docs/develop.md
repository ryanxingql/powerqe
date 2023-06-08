# Development Document

## BPG compression on Mac

First install libbpg via homebrew:

```bash
brew install libbpg
# brew list
```

Then replace the `BPGENC_PATH` and `BPGDEC_PATH` lines of [the Bash script in the V3 document](v3.md#example-compress-the-div2k-dataset):

```bash
BPGENC_PATH="/opt/homebrew/bin/bpgenc"
BPGDEC_PATH="/opt/homebrew/bin/bpgdec"
```

Finally run the Bash script.

## Debug

Edit `configs/debug.py`.

Then run:

```bash
conda activate powerqev4 &&\
  PYTHONPATH=./\
  python tools/train.py\
  configs/debug.py
```

With `PYTHONPATH=./`, the `powerqe` dataset can be found as a module by Python.

## Markdown heading text anchors

For example, the anchor (ID) of this paragraph is `markdown-paragraph-id`. One can jump from another Markdown file to this paragraph by `docs/develop.md#markdown-paragraph-id`.

Note that Markdown will convert the heading text to lowercase, remove any non-alphanumeric characters, and replace spaces with hyphens. For example, if we have two paragraph named `Markdown paragraph: ID` and `Markdown paragraph ID`, then their ID are `#markdown-paragraph-id` and `#markdown-paragraph-id-1`, respectively.

It's worth noting that the exact algorithm for generating the unique identifier can vary between Markdown renderers and may depend on the specific implementation details of the renderer. We may check ID by GitHub or VS Code TOC.
