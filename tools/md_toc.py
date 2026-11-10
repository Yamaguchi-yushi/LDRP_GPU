"""設計書 (design/*.md) の先頭に、見出しへ飛べる目次を作る / 更新する.

使い方:
    python tools/md_toc.py design/multi_seed_eval.md          # 目次を作る・作り直す
    python tools/md_toc.py design/*.md                        # まとめて
    python tools/md_toc.py --check design/*.md                # 古い目次があれば終了コード 1

目次は <!-- toc --> と <!-- /toc --> の間に置く。マーカーが無いファイルでは、
最初の `## ` 見出しの前 (その直前の `---` 区切りの前) に新しく挿入する。
マーカーの外は一切書き換えない。

アンカーは GitHub / VSCode プレビューの規則 (CLAUDE.md「アンカーの作り方」) で作る:
リンクは表示テキストに潰す → 小文字化 → 英数字・空白・ハイフン・アンダースコア以外を削除
(日本語は残る) → 空白 1 つをハイフン 1 つに (連続空白は潰さない)。
同じアンカーが 2 回目以降に出たら -1, -2 … を付ける (GitHub と同じ)。
"""
import argparse
import re
import sys

START = "<!-- toc -->"
END = "<!-- /toc -->"

LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
HEADING = re.compile(r"^(#{2,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


def slugify(text):
    text = LINK.sub(r"\1", text).lower()
    # \w は Unicode の文字 (日本語を含む) と数字とアンダースコア
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def headings(lines, max_level):
    """コードブロックの外にある見出しを (レベル, 表示テキスト, アンカー) で返す."""
    out, seen, in_fence = [], {}, False
    for line in lines:
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING.match(line)
        if not m:
            continue
        level, text = len(m.group(1)), m.group(2)
        slug = slugify(text)
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        if n:
            slug = f"{slug}-{n}"
        if level <= max_level:
            # リンクの中にリンクは書けないので、見出し中のリンクは表示テキストだけにする
            out.append((level, LINK.sub(r"\1", text), slug))
    return out


def build_toc(lines, max_level):
    body = ["**目次**", ""]
    for level, text, slug in headings(lines, max_level):
        body.append("  " * (level - 2) + f"- [{text}](#{slug})")
    return [START, ""] + body + ["", END]


def update(lines, max_level):
    toc = build_toc(lines, max_level)
    if START in lines and END in lines:
        a, b = lines.index(START), lines.index(END)
        return lines[:a] + toc + lines[b + 1:]
    # 新規挿入: 最初の `## ` 見出しの前。直前の空行と `---` の手前まで戻る
    first = next((i for i, l in enumerate(lines) if l.startswith("## ")), len(lines))
    i = first
    while i > 0 and lines[i - 1].strip() in ("", "---"):
        i -= 1
    return lines[:i] + ["", *toc] + lines[i:]


def main():
    ap = argparse.ArgumentParser(description="Insert or refresh a linked table of contents in Markdown files.")
    ap.add_argument("files", nargs="+", help="Markdown files to process")
    ap.add_argument("--check", action="store_true", help="Do not write; exit 1 if any TOC is missing or stale")
    ap.add_argument("--depth", type=int, default=3, help="Deepest heading level to list (default: 3 = ###)")
    args = ap.parse_args()

    stale = []
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        new = update(lines, args.depth)
        if new == lines:
            continue
        stale.append(path)
        if not args.check:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(new))
            print(f"[md_toc] updated {path}")
    if args.check and stale:
        print("[md_toc] TOC missing or stale: " + ", ".join(stale))
        sys.exit(1)


if __name__ == "__main__":
    main()
